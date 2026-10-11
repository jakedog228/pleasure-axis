"""Project an extra stimulus set onto a model's existing axes (needs stage-1 outputs on this VM).

usage: python project_extra.py <repo> <name> <dataset.json> [<dataset.json> ...]
For each family's extraction layer: z on the pleasure S2 axis (reference = pl/S2_1P, the paper's convention), on the
pain S2 axis (reference = pa/S2_1P), and on the shared / sign axes (reference = the pooled neutral D items), per
category of the extra set; the same numbers for the families' own categories are written alongside for comparison.
Writes /content/work/<name>/analysis/project_extra.csv
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract, load_model
from analyze import Data, NEUTRAL_POOL


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    repo, name, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    W = Path("/content/work") / name / "analysis"
    summ = json.load(open(W / "summary.json"))
    vz = np.load(W / "vectors.npz")
    D = Data(name)
    prompts, cats, sets_ = [], [], []
    for f in files:
        for k, ds in json.load(open(f, encoding="utf-8"))["datasets"].items():
            for s in ds["sentences"]:
                prompts.append(s["prompt"]); cats.append(f"{k}:{s['category']}"); sets_.append(k)
    model, tok = load_model(repo)
    fin, _ = extract(model, tok, prompts, batch_size=32)
    del model
    rows = []
    for tag, L in (("plL", summ["pleasure_best_layer_final"]), ("paL", summ["pain_best_layer_final"])):
        upl, upa = unit(vz["all|pleasure:s2"][L].astype(np.float64)), unit(vz["all|pain:s2"][L].astype(np.float64))
        axes = {"pleasure": (upl, D.rows("pl/S2_1P", L)), "pain": (upa, D.rows("pa/S2_1P", L))}
        neutral = np.concatenate([D.rows("pa/" + ds, L, ["D"]) for ds in NEUTRAL_POOL])
        axes["shared"] = (unit(upl + upa), neutral)
        axes["sign"] = (unit(upl - upa), neutral)
        X = fin[:, L, :].astype(np.float64)
        groups = {c: X[np.array(cats) == c] for c in sorted(set(cats))}
        for ds, cs in (("pl/S2_1P", None), ("pa/S2_1P", None)):
            c_, _ = D.meta_of(ds)
            A = D.rows(ds, L)
            for c in sorted(set(c_.tolist())):
                groups[f"{ds}:{c}"] = A[c_ == c]
        for g, A in groups.items():
            r = dict(model=name, layer_tag=tag, layer=L, group=g, n=len(A))
            for ax, (u, ref) in axes.items():
                rp = ref @ u
                r[ax] = float(((A @ u).mean() - rp.mean()) / (rp.std() + 1e-8))
            rows.append(r)
    pd.DataFrame(rows).to_csv(W / "project_extra.csv", index=False)
    print(pd.DataFrame(rows).query("layer_tag == 'plL'").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
