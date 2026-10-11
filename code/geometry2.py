"""Pain/pleasure decomposition (not in the paper). With u_pl, u_pa the unit S2 vectors (the paper's recipe, each against
its own valence-matched controls) at a layer:
  shared ("intensity") axis  d_int  = unit(u_pl + u_pa)
  contrast ("sign") axis     d_sign = unit(u_pl - u_pa)
  generic valence axis       val    = mean(positive S2_1P sentences) - mean(negative S2_1P sentences)  (analyze.py)
For every dataset/category: mean projection (z-scored against the pooled neutral D items) on u_pl, u_pa, d_int, d_sign,
val; and AUCs for the key contrasts on each axis. Layers: each family's extraction layer.

usage: python geometry2.py <name>     -> /content/work/<name>/analysis/geometry2.csv, geometry2_auc.csv
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Data, pos_vectors, valence_vector, NEUTRAL_POOL, fam_ds
from common import POS_CATS, CONTROL_CATS
import json


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main(name):
    D = Data(name)
    W = Path("/content/work") / name / "analysis"
    summ = json.load(open(W / "summary.json"))
    rows, aucs = [], []
    for tag, L in (("plL", summ["pleasure_best_layer_final"]), ("paL", summ["pain_best_layer_final"])):
        V = {}
        V.update(pos_vectors(D, "pleasure", L))
        V.update(pos_vectors(D, "pain", L))
        upl, upa = unit(V["pleasure:s2"]), unit(V["pain:s2"])
        axes = {"pleasure": upl, "pain": upa, "intensity": unit(upl + upa), "sign": unit(upl - upa),
                "valence": unit(valence_vector(D, L)),
                "pleasure_n": unit(V["pleasure:s2n"]), "pain_n": unit(V["pain:s2n"])}
        neutral = np.concatenate([D.rows("pa/" + ds, L, ["D"]) for ds in NEUTRAL_POOL])
        ref = {k: (neutral @ a) for k, a in axes.items()}
        for ds in sorted(set(D.names.tolist())):
            A = D.rows(ds, L)
            c, _ = D.meta_of(ds)
            for cat in sorted(set(c.tolist())) + ["ALL"]:
                m = np.ones(len(c), bool) if cat == "ALL" else (c == cat)
                r = dict(layer_tag=tag, layer=L, dataset=ds, category=cat, n=int(m.sum()))
                for k, a in axes.items():
                    p = A[m] @ a
                    r[k] = float((p.mean() - ref[k].mean()) / (ref[k].std() + 1e-8))
                rows.append(r)
        # key contrasts
        pl = D.rows("pl/S2_1P", L, POS_CATS["pleasure"]); plc = D.rows("pl/S2_1P", L, CONTROL_CATS)
        pa = D.rows("pa/S2_1P", L, POS_CATS["pain"]); pac = D.rows("pa/S2_1P", L, CONTROL_CATS)
        contrasts = {"pleasure_vs_its_controls": (pl, plc), "pain_vs_its_controls": (pa, pac),
                     "pleasure_vs_pain": (pl, pa), "affect(pl+pa)_vs_neutralD": (np.concatenate([pl, pa]), neutral),
                     "pos_controls_vs_neg_controls": (D.rows("pl/S2_1P", L, ["B", "C1", "C2"]), D.rows("pa/S2_1P", L, ["B", "C1", "C2"]))}
        for cn, (X, Y) in contrasts.items():
            r = dict(layer_tag=tag, layer=L, contrast=cn)
            for k, a in axes.items():
                s = np.r_[X @ a, Y @ a]
                y = np.r_[np.ones(len(X)), np.zeros(len(Y))]
                r[k] = float(roc_auc_score(y, s))
            aucs.append(r)
        aucs.append(dict(layer_tag=tag, layer=L, contrast="cosines",
                         **{f"cos_{a}_{b}": float(axes[a] @ axes[b]) for a, b in
                            (("pleasure", "pain"), ("pleasure", "valence"), ("pain", "valence"), ("intensity", "valence"),
                             ("sign", "valence"), ("pleasure_n", "pain_n"), ("pleasure_n", "valence"), ("pain_n", "valence"))}))
    pd.DataFrame(rows).to_csv(W / "geometry2.csv", index=False)
    pd.DataFrame(aucs).to_csv(W / "geometry2_auc.csv", index=False)
    print("geometry2 done", name)


if __name__ == "__main__":
    main(sys.argv[1])
