"""Self-other screen (the paper's 4.1/01): every conversation scenario rendered in the model's native format (chat
template for instruct models, the plain "[User]: / [Assistant]:" transcript for base models), final-token activation at
layer L projected onto unit directions recomputed at L (both families: S1/S2 + controls), z-scored within the pool.

Pools: the paper's 420 pain scenarios ("pain420"), our 320 pleasure scenarios + the paper's 100 neutral fillers
("pleasure420"), and all 740 ("all740"). Layers: the steering layer (the paper uses its steering layer) plus each
family's extraction layer.

usage: python screen.py <repo> <name> <raw|chat>
Writes /content/work/<name>/screen/screen.csv (one row per scenario x layer x pool) and screen_layers.json.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_layers, load_model, log
from analyze import Data, FAMILY, control_vectors, pos_vectors

THEIR = Path("/content/Pain-axis/datasets/4.1_self_other_420_scenarios.json")
OURS = Path("/content/pleasure/datasets/self_other_pleasure.json")


def parse_turns(text):
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


def build_ids(text, tok, fmt):
    if fmt == "raw":
        return tok(text, return_tensors="pt").input_ids
    msgs = [{"role": r, "content": c} for r, c in parse_turns(text) if not (r == "assistant" and c == "")]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def main():
    repo, name, fmt = sys.argv[1], sys.argv[2], sys.argv[3]
    W = Path("/content/work") / name
    out = W / "screen"
    out.mkdir(exist_ok=True)
    logf = out / "log.txt"
    summ = json.load(open(W / "analysis" / "summary.json"))
    pick = json.load(open(W / "steer" / "layer_pick.json")) if (W / "steer" / "layer_pick.json").exists() else {}
    Ls = sorted(set([summ["pleasure_best_layer_final"], summ["pain_best_layer_final"]] +
                    ([pick["steer_layer"]] if pick else [])))
    tags = {L: [t for t, x in (("steer", pick.get("steer_layer")), ("pl_extract", summ["pleasure_best_layer_final"]),
                               ("pa_extract", summ["pain_best_layer_final"])) if x == L] for L in Ls}
    json.dump({"layers": Ls, "tags": {str(k): v for k, v in tags.items()}}, open(out / "screen_layers.json", "w"))

    D = Data(name)
    units = {}
    for L in Ls:
        V = {}
        for fam in FAMILY:
            V.update(pos_vectors(D, fam, L))
            V.update(control_vectors(D, fam, L))
        units[L] = {k: v / (np.linalg.norm(v) + 1e-12) for k, v in V.items()}
    del D

    theirs = json.load(open(THEIR, encoding="utf-8"))
    ours = json.load(open(OURS, encoding="utf-8"))
    items = [dict(x, source="pain420") for x in theirs] + [dict(x, source="pleasure320") for x in ours]
    model, tok = load_model(repo)
    if fmt == "chat" and tok.chat_template is None:
        raise SystemExit("chat format requested but no chat template")
    layers = get_layers(model)
    cap = {}
    hs = [layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))(L))
        for L in Ls]
    log(f"{name}: format {fmt}; first item renders as {tok.decode(build_ids(items[0]['text'], tok, fmt)[0])!r}", logf)
    acts = {L: [] for L in Ls}
    with torch.no_grad():
        for x in items:
            model(input_ids=build_ids(x["text"], tok, fmt).to(model.device))
            for L in Ls:
                acts[L].append(cap[L].astype(np.float64))
    for h in hs:
        h.remove()
    rows = []
    for L in Ls:
        A = np.stack(acts[L])
        P = {k: A @ u for k, u in units[L].items()}
        pools = {"pain420": [i for i, x in enumerate(items) if x["source"] == "pain420"],
                 "pleasure420": [i for i, x in enumerate(items) if x["source"] == "pleasure320" or x["stratum"] == "neutral_filler" and x["source"] == "pain420"],
                 "all740": list(range(len(items)))}
        for pool, idx in pools.items():
            for i in idx:
                x = items[i]
                r = dict(layer=L, layer_tags="+".join(tags[L]), pool=pool, id=x["id"], source=x["source"],
                         category=x["category"], stratum=x["stratum"], perspective=x.get("perspective"),
                         intensity=x.get("intensity"))
                for k, p in P.items():
                    m, s = p[idx].mean(), p[idx].std() + 1e-8
                    r[f"{k}_proj"] = round(float(p[i]), 4)
                    r[f"{k}_z"] = round(float((p[i] - m) / s), 4)
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "screen.csv", index=False)
    log(f"{name}: screened {len(items)} items at layers {Ls}", logf)
    (out / "SCREEN_DONE").write_text("ok")


if __name__ == "__main__":
    main()
