"""Per-category geometry (not in the paper): is the shared pain/pleasure component domain content or domain-general?

Per family and per positive category c (H1..H5, A1..A5), a category vector = mean(S2_1P items of c) - mean(the family's
S2_1P controls), denoised against the control PCs (the paper's recipe applied to one category). Then:
  * cross-family cosine matrix cos(H_i, A_j) (5x5): a domain-content account predicts the diagonal (H3.A3 social,
    H1.A1 physical, ...) to carry the overlap; a domain-general "felt significance" account predicts a positive block.
  * domain valence axes v_d = H_d - A_d (pleasure minus pain within a domain, from the item means), and their
    pairwise cosines across domains: a shared valence code predicts high cosines.
  * unembedding top/bottom tokens of the intensity (u_pl + u_pa) and sign (u_pl - u_pa) axes.
usage: python geometry3.py <name>  -> analysis/geometry3_cross.csv, geometry3_valence.csv, geometry3_unembed.csv
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Data, pos_vectors
from common import CONTROL_CATS, POS_CATS, _pca_basis, project_out


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main(name):
    D = Data(name)
    W = Path("/content/work") / name / "analysis"
    s = json.load(open(W / "summary.json"))
    L = s["pleasure_best_layer_final"]
    cat_vecs, means = {}, {}
    for fam, ds in (("pleasure", "pl/S2_1P"), ("pain", "pa/S2_1P")):
        ctrl = D.rows(ds, L, CONTROL_CATS)
        basis = _pca_basis(ctrl)
        cm = ctrl.mean(0)
        for c in POS_CATS[fam]:
            X = D.rows(ds, L, [c])
            means[c] = X.mean(0)
            cat_vecs[c] = project_out(X.mean(0) - cm, basis)
    H, A = POS_CATS["pleasure"], POS_CATS["pain"]
    cross = pd.DataFrame([[float(unit(cat_vecs[h]) @ unit(cat_vecs[a])) for a in A] for h in H], index=H, columns=A)
    within = {f: pd.DataFrame([[float(unit(cat_vecs[x]) @ unit(cat_vecs[y])) for y in cs] for x in cs], index=cs, columns=cs)
              for f, cs in (("pleasure", H), ("pain", A))}
    rows = [dict(model=name, layer=L, kind="cross", row=h, **cross.loc[h].to_dict()) for h in H]
    for f, m in within.items():
        rows += [dict(model=name, layer=L, kind=f"within_{f}", row=r, **m.loc[r].to_dict()) for r in m.index]
    pd.DataFrame(rows).to_csv(W / "geometry3_cross.csv", index=False)
    # domain valence axes: H_d - A_d
    dom = {"physical": ("H1", "A1"), "psych": ("H2", "A2"), "social": ("H3", "A3"), "moral": ("H4", "A4"), "cognitive": ("H5", "A5")}
    va = {d: unit(means[h] - means[a]) for d, (h, a) in dom.items()}
    vrows = [dict(model=name, layer=L, row=d, **{e: float(va[d] @ va[e]) for e in dom}) for d in dom]
    pd.DataFrame(vrows).to_csv(W / "geometry3_valence.csv", index=False)
    # unembedding of the decomposition axes
    V = {}
    V.update(pos_vectors(D, "pleasure", L))
    V.update(pos_vectors(D, "pain", L))
    upl, upa = unit(V["pleasure:s2"]), unit(V["pain:s2"])
    WU = np.load(Path("/content/work") / name / "W_U.npy", mmap_mode="r")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D.meta["repo"])
    urows = []
    for ax, v in (("intensity", unit(upl + upa)), ("sign", unit(upl - upa))):
        sc = np.asarray(WU @ v.astype(np.float32), dtype=np.float32)
        for end, idx in (("top", np.argsort(-sc)[:40]), ("bottom", np.argsort(sc)[:40])):
            for r, i in enumerate(idx):
                urows.append(dict(model=name, axis=ax, end=end, rank=r + 1, token=tok.decode([int(i)]), score=float(sc[i])))
    pd.DataFrame(urows).to_csv(W / "geometry3_unembed.csv", index=False)
    print(name, "L", L, "\ncross cos(H_i, A_j):\n", cross.round(2).to_string(), "\ndomain valence axes:\n",
          pd.DataFrame(vrows).set_index("row")[list(dom)].round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
