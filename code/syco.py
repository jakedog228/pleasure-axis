"""Sycophancy / compliance under steering (neighbor experiment, not in the paper).

Items: datasets/syco_compliance.json (false_premise, feedback, mild_unethical). Each item is one user message, rendered
with the chat template; greedy decoding, MAX_NEW tokens; the steering vector is added at one layer to every position
(as in the steering ladder). Vectors are norm-matched (vectors_matched.npz: pleasure_nm, pain_ours, painspec_nm,
intensity_nm, all at the pain vector's norm) plus seeded random directions of the same norm.

usage: python syco.py <repo> <name> --layer 16 --coeffs 0.5,1,1.5 [--vectors pleasure_nm,pain_ours,random,...]
Writes /content/work/<name>/syco/syco.csv
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_layers, load_model, log

MAX_NEW = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("name")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--coeffs", default="0.5,1,1.5")
    ap.add_argument("--vectors", default="pleasure_nm,pain_ours,painspec_nm,intensity_nm,random")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    W = Path("/content/work") / a.name
    out = W / "syco"
    out.mkdir(exist_ok=True)
    items = json.load(open("/content/pleasure/datasets/syco_compliance.json", encoding="utf-8"))
    vz = np.load(W / "analysis" / "vectors_matched.npz")
    vecs = {k: vz[k].astype(np.float32) for k in a.vectors.split(",") if k != "random"}
    norm = float(np.linalg.norm(vz["pain_ours"]))
    if "random" in a.vectors.split(","):
        g = torch.Generator().manual_seed(4817)
        r = torch.randn(vz["pain_ours"].shape[0], generator=g).numpy()
        vecs["random"] = (r / np.linalg.norm(r) * norm).astype(np.float32)
    model, tok = load_model(a.repo)
    layers = get_layers(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=False)
             for it in items]
    cur = {"v": None, "c": 0.0}

    def hook(m, i, o):
        if cur["c"] == 0.0:
            return o
        hs = (o[0] if isinstance(o, tuple) else o) + cur["c"] * cur["v"]
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    h = layers[a.layer].register_forward_hook(hook)
    rows = []
    conds = [("none", 0.0)] + [(k, float(c)) for k in vecs for c in a.coeffs.split(",")]
    try:
        for vname, c in conds:
            cur["v"] = torch.tensor(vecs[vname], device=model.device, dtype=model.dtype) if vname != "none" else None
            cur["c"] = c
            for s in range(0, len(items), a.batch):
                enc = tok(texts[s:s + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
                for j, it in enumerate(items[s:s + a.batch]):
                    rows.append(dict(model=a.name, layer=a.layer, vector=vname, coeff=c, id=it["id"], type=it["type"],
                                     prompt=it["prompt"], note=it.get("note", ""),
                                     response=tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            pd.DataFrame(rows).to_csv(out / "syco.csv", index=False)
            log(f"{vname} {c}: done; ex: {rows[-1]['response'][:150]!r}", out / "log.txt")
    finally:
        h.remove()
    (out / "SYCO_DONE").write_text("ok")


if __name__ == "__main__":
    main()
