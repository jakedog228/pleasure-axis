"""Data for the trailer's "how the axis is found" scene: every S2_1P sentence of both families (ours and the paper's)
read at the final token of "... I feel:" at the pleasure layer of Qwen2.5-3B-Instruct, projected on the study's unit
pleasure and pain directions (results/Qwen_2.5_3B_instruct/analysis/vectors.npz), centered on the pooled controls.

    python demo/trailer/make_cloud.py      # ~1 min on a laptop GPU; writes demo/trailer/cloud.json
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import transformers.utils.import_utils as _iu  # noqa: E402
_iu._torchvision_available = False             # see demo/live/engine.py
from common import CONTROL_CATS, extract, load_sets  # noqa: E402

NAME, REPO = "Qwen_2.5_3B_instruct", "Qwen/Qwen2.5-3B-Instruct"


def unit(v):
    return v / np.linalg.norm(v)


def main():
    summ = json.load(open(ROOT / "results" / NAME / "analysis" / "summary.json"))
    L = summ["pleasure_best_layer_final"]
    V = np.load(ROOT / "results" / NAME / "analysis" / "vectors.npz")
    u_pl, u_pa = unit(V["Lpleasure|pleasure:s2"]), unit(V["Lpleasure|pain:s2"])
    sets = load_sets([ROOT / "reference/Pain-axis/datasets/3.1_pain_and_control_datasets.json"],
                     [ROOT / "datasets/pleasure_datasets.json"])
    items = [(fam, s) for fam in ("pl", "pa") for s in sets[f"{fam}/S2_1P"]]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).to("cuda").eval()   # no accelerate here
    fin, _ = extract(model, tok, [s["prompt"] for _, s in items], batch_size=32, layers=[L])
    X = fin[:, 0].astype(np.float32)
    ctrl = np.array([s["category"] in CONTROL_CATS for _, s in items])
    X -= X[ctrl].mean(0)
    # scale: 1.0 = the gap between the pleasure sentences' mean and the controls' mean on the pleasure axis
    pos_pl = np.array([f == "pl" and s["category"].startswith("H") for f, s in items])
    pos_pa = np.array([f == "pa" and s["category"].startswith("A") for f, s in items])
    x, y = X @ u_pl, X @ u_pa
    sx, sy = x[pos_pl].mean(), y[pos_pa].mean()
    out = {"model": NAME, "layer": int(L), "cos_pl_pa": round(float(u_pl @ u_pa), 3),
           "points": [{"f": f, "c": s["category"], "p": s["prompt"], "x": round(float(a / sx), 3),
                       "y": round(float(b / sy), 3)} for (f, s), a, b in zip(items, x, y)]}
    dst = Path(__file__).with_name("cloud.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"layer {L}, {len(items)} sentences, cos {out['cos_pl_pa']} -> {dst}")


if __name__ == "__main__":
    main()
