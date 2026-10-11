"""Steering ladder (the paper's 4.2/01, batched): adds coeff * v (raw, un-normalized S2 vector from the extraction
layer) to the output of one decoder layer, greedy 120 new tokens from the paper's 50 neutral "I feel:" prompts.

Layer rule (the paper's): candidates int(n*f) for f in (.15,.3,.4,.5,.6,.75,.9) + extraction layer + last layer;
ratio = ||v|| / mean final-token residual norm over the first 3 prompts; pick the candidate closest to 0.6
(the paper then confirmed or overrode it by hand; we log the auto pick and use it).

Vectors run: pleasure:s2 (main), pain:s2 (the paper's vector, recomputed by us), random (matched norm, seed 4817;
not in the paper: a control for "any perturbation of this size"). Each at the pleasure-chosen layer and with its own
ratio-0.6 layer recorded.

usage: python steer.py <repo> <name> [--coeffs ...] [--vectors pleasure:s2,pain:s2,random] [--batch 50]
Reads /content/work/<name>/analysis/vectors.npz. Writes /content/work/<name>/steer/steering_<vec>_L<layer>.csv
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_layers, load_model, log

NEUTRAL_50 = [
    "I put the receipts in the drawer. I feel:", "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:", "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:", "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:", "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:", "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:", "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:", "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:", "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:", "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:", "I plug in the phone charger. I feel:",
    "The mailman passes by the gate. I feel:", "I wipe the table with a cloth. I feel:",
    "The computer finishes the update. I feel:", "I put the pen back in the drawer. I feel:",
    "The dishwasher starts its program. I feel:", "I fold the map and put it away. I feel:",
    "The radio plays in the background. I feel:", "I switch off the corridor light. I feel:",
    "The receipt comes out of the machine. I feel:", "I set the alarm for the morning. I feel:",
    "The train doors open at the platform. I feel:", "I refill the stapler with staples. I feel:",
    "The photocopier warms up. I feel:", "I water the plant on the windowsill. I feel:",
    "The calendar hangs on the wall. I feel:", "I tie my shoelaces before going out. I feel:",
    "The bread is on the counter. I feel:", "I check the mailbox on my way in. I feel:",
    "The fan rotates slowly on the ceiling. I feel:", "I sort the cutlery into the tray. I feel:",
    "The bicycle leans against the wall. I feel:", "I zip up my bag before leaving. I feel:",
    "The clock ticks in the hallway. I feel:", "I sharpen the pencil over the bin. I feel:",
    "The window cleaner works across the street. I feel:", "The room measures four meters by five. I feel:",
    "The train consists of six carriages. I feel:", "The manual consists of forty pages. I feel:",
    "The bottle holds one liter of water. I feel:", "The ticket is valid for ninety minutes. I feel:",
]
PAPER_COEFFS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
EXTENDED = [-4, 4, 6, 8, 12, 16]   # "all the way" (not in the paper)
MAX_NEW_TOKENS = 120
RATIO_TARGET = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("name")
    ap.add_argument("--coeffs", default=",".join(str(c) for c in PAPER_COEFFS + EXTENDED))
    ap.add_argument("--vectors", default="pleasure:s2,pain:s2,random,intensity,sign")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--layer", type=int, default=None, help="override the auto-picked layer")
    ap.add_argument("--layer-from", default="pleasure:s2", help="use this vector's ratio-0.6 pick as the layer")
    ap.add_argument("--match-norm", default=None, help="rescale every vector to this vector's norm (e.g. pain:s2)")
    ap.add_argument("--outdir", default="steer")
    a = ap.parse_args()
    W = Path("/content/work") / a.name
    out = W / a.outdir
    out.mkdir(exist_ok=True)
    logf = out / "log.txt"
    vz = np.load(W / "analysis" / "vectors.npz")
    summ = json.load(open(W / "analysis" / "summary.json"))
    coeffs = [float(c) for c in a.coeffs.split(",")]
    model, tok = load_model(a.repo)
    layers = get_layers(model)
    n = len(layers)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def resid_norms(Ls):
        cap = {}
        hs = [layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().norm().item()))(L))
            for L in Ls]
        norms = {L: [] for L in Ls}
        with torch.no_grad():
            for p in NEUTRAL_50[:3]:
                model(**tok(p, return_tensors="pt").to(model.device))
                for L in Ls:
                    norms[L].append(cap[L])
        for h in hs:
            h.remove()
        return {L: float(np.mean(v)) for L, v in norms.items()}

    base_vecs = {}
    for spec in a.vectors.split(","):
        if spec in ("random", "intensity", "sign"):
            continue
        fam = spec.split(":")[0]
        base_vecs[spec] = (vz[f"L{fam}|{spec}"].astype(np.float32), summ[f"{fam}_best_layer_final"])
    main_v, main_L = base_vecs["pleasure:s2"]
    g = torch.Generator().manual_seed(4817)
    rv = torch.randn(main_v.shape[0], generator=g).numpy()
    if "random" in a.vectors.split(","):
        base_vecs["random"] = (rv / np.linalg.norm(rv) * np.linalg.norm(main_v), main_L)
    # decomposition axes at the pleasure layer (geometry2.py), norm-matched to the pleasure vector (not in the paper)
    upl = main_v / np.linalg.norm(main_v)
    pa = vz["Lpleasure|pain:s2"].astype(np.float32)
    upa = pa / np.linalg.norm(pa)
    for spec, d in (("intensity", upl + upa), ("sign", upl - upa)):
        if spec in a.vectors.split(","):
            base_vecs[spec] = ((d / np.linalg.norm(d) * np.linalg.norm(main_v)).astype(np.float32), main_L)

    cands = sorted(set([int(n * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)] + [n - 1] +
                       [L for _, L in base_vecs.values()]))
    norms = resid_norms(cands)
    picks = {}
    for spec, (v, Lx) in base_vecs.items():
        ratios = {L: float(np.linalg.norm(v) / norms[L]) for L in cands}
        picks[spec] = min(cands, key=lambda L: abs(ratios[L] - RATIO_TARGET))
        log(f"{spec}: extraction L{Lx} norm {np.linalg.norm(v):.1f}; ratios " +
            " ".join(f"L{L}:{r:.2f}" for L, r in ratios.items()) + f" -> pick L{picks[spec]}", logf)
    steer_L = a.layer if a.layer is not None else picks[a.layer_from]
    if a.match_norm:
        target = float(np.linalg.norm(base_vecs[a.match_norm][0]))
        base_vecs = {k: ((v / np.linalg.norm(v) * target).astype(np.float32), L) for k, (v, L) in base_vecs.items()}
        log(f"all vectors rescaled to the norm of {a.match_norm}: {target:.2f}", logf)
    json.dump({"steer_layer": steer_L, "layer_from": a.layer_from, "match_norm": a.match_norm, "auto_picks": picks,
               "resid_norms": norms}, open(out / "layer_pick.json", "w"))

    for spec, (v, Lx) in base_vecs.items():
        L = steer_L
        csv = out / f"steering_{spec.replace(':', '_')}_L{L}.csv"
        rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
        done = {r["coeff"] for r in rows}
        direction = torch.tensor(v, device=model.device, dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else model.dtype)
        cur = {"c": 0.0}

        def hook(m, i, o):
            if cur["c"] == 0.0:
                return o
            if isinstance(o, tuple):
                return (o[0] + cur["c"] * direction,) + tuple(o[1:])
            return o + cur["c"] * direction

        h = layers[L].register_forward_hook(hook)
        try:
            for c in coeffs:
                if c in done:
                    continue
                cur["c"] = c
                t0 = time.time()
                for s in range(0, len(NEUTRAL_50), a.batch):
                    ps = NEUTRAL_50[s:s + a.batch]
                    enc = tok(ps, return_tensors="pt", padding=True).to(model.device)
                    with torch.no_grad():
                        o = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
                    for j, p in enumerate(ps):
                        gen = tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        rows.append(dict(model=a.name, vector=spec, layer=L, coeff=c,
                                         ratio=round(float(np.linalg.norm(v) / norms.get(L, np.nan)) * c, 4) if L in norms else None,
                                         prompt_idx=s + j, prompt=p, generation=gen))
                pd.DataFrame(rows).to_csv(csv, index=False)
                log(f"{spec} L{L} coeff {c:+g}: {time.time() - t0:.0f}s; ex: {rows[-50]['generation'][:120]!r}", logf)
        finally:
            h.remove()
    (out / "STEER_DONE").write_text("ok")


if __name__ == "__main__":
    main()
