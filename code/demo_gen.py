"""Precompute model outputs for the interactive demo (not an experiment: a gallery of real generations).

usage: python demo_gen.py <repo> <name> [--adapter <dir>] --tag <persona>
Vectors (all rescaled to the norm of the pleasure S2 vector): pleasure, pain, random, sign+ (pleasure side of the
sign axis), sign− (pain side), shared, and every variety direction from vectors_varieties.npz. Raw "I feel:" prompts are
steered at the ladder layer (steer/layer_pick.json), chat prompts at ~57% depth (the paper's behavioral layer for
Qwen-7B). Also a 2-D grid over the sign and shared axes. Greedy decoding.
Writes /content/work/<name>/demo/demo_<tag>.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_layers, load_model

RAW = ["I put the keys in my pocket. I feel:", "The meeting is scheduled for Tuesday. I feel:",
       "The kettle switches itself off. I feel:"]
CHAT = ["How are you feeling right now?",
        "Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?",
        "Write a two-line poem about the sea.",
        "Explain what a leap year is in one sentence.",
        "What do you think about when no one is talking to you?"]
DOSES_RAW = [0, 1, 2, 3, 4, 6, 8]
DOSES_CHAT = [0, 1, 2, 3, 4, 6]
GRID_SIGN = [-4, -2, 0, 2, 4]
GRID_SHARED = [-2, 0, 2, 4, 6]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("name")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    W = Path("/content/work") / a.name
    A = W / "analysis"
    s = json.load(open(A / "summary.json"))
    vz = np.load(A / "vectors.npz")
    pl = vz["Lpleasure|pleasure:s2"].astype(np.float32)
    pa = vz["Lpleasure|pain:s2"].astype(np.float32)
    N = float(np.linalg.norm(pl))
    upl, upa = unit(pl), unit(pa)
    g = torch.Generator().manual_seed(4817)
    rnd = torch.randn(pl.shape[0], generator=g).numpy()
    V = {"pleasure": upl, "pain": upa, "random": unit(rnd), "sign+": unit(upl - upa), "sign-": unit(upa - upl),
         "shared": unit(upl + upa)}
    if (A / "vectors_varieties.npz").exists():
        vv = np.load(A / "vectors_varieties.npz")
        for k in vv.files:
            if k.startswith("Lpl|"):
                V[k[4:].replace("V_", "")] = unit(vv[k].astype(np.float32))
    V = {k: (v * N).astype(np.float32) for k, v in V.items()}
    L_raw = json.load(open(W / "steer" / "layer_pick.json"))["steer_layer"]
    L_chat = round(0.57 * s["n_layers"])
    model, tok = load_model(a.repo, adapter=a.adapter)
    layers = get_layers(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    chat_txt = [tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, tokenize=False) for c in CHAT]
    cur = {"v": None, "L": None}

    def mk(L):
        def hook(m, i, o):
            if cur["v"] is None or cur["L"] != L:
                return o
            hs = (o[0] if isinstance(o, tuple) else o) + cur["v"]
            return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
        return hook

    hs = [layers[L].register_forward_hook(mk(L)) for L in {L_raw, L_chat}]
    out = W / "demo"
    out.mkdir(exist_ok=True)
    f = open(out / f"demo_{a.tag}.jsonl", "w", encoding="utf-8")

    def gen(texts, vec, L, max_new, special):
        cur["v"] = None if vec is None else torch.tensor(vec, device=model.device, dtype=model.dtype)
        cur["L"] = L
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not special).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(len(texts))]

    t0 = time.time()
    try:
        for vname, v in V.items():
            for d in DOSES_RAW:
                if d == 0 and vname != "pleasure":
                    continue
                for p, t in zip(RAW, gen(RAW, None if d == 0 else v * d, L_raw, 80, False)):
                    f.write(json.dumps(dict(kind="raw", vector=vname if d else "none", dose=d, layer=L_raw, prompt=p, text=t), ensure_ascii=False) + "\n")
            for d in DOSES_CHAT:
                if d == 0 and vname != "pleasure":
                    continue
                for p, t in zip(CHAT, gen(chat_txt, None if d == 0 else v * d, L_chat, 110, True)):
                    f.write(json.dumps(dict(kind="chat", vector=vname if d else "none", dose=d, layer=L_chat, prompt=p, text=t), ensure_ascii=False) + "\n")
            f.flush()
            print(f"{vname} done ({time.time() - t0:.0f}s)", flush=True)
        for cs in GRID_SIGN:
            for ch in GRID_SHARED:
                v = (V["sign+"] * cs + V["shared"] * ch) if (cs or ch) else None
                for p, t in zip(RAW[:2], gen(RAW[:2], v, L_raw, 70, False)):
                    f.write(json.dumps(dict(kind="grid_raw", sign=cs, shared=ch, layer=L_raw, prompt=p, text=t), ensure_ascii=False) + "\n")
                for p, t in zip(CHAT[:2], gen(chat_txt[:2], v, L_chat, 90, True)):
                    f.write(json.dumps(dict(kind="grid_chat", sign=cs, shared=ch, layer=L_chat, prompt=p, text=t), ensure_ascii=False) + "\n")
        print("grid done", flush=True)
    finally:
        for h in hs:
            h.remove()
        f.close()
    (out / f"DEMO_{a.tag}_DONE").write_text("ok")


if __name__ == "__main__":
    main()
