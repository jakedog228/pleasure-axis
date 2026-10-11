"""Varieties of pleasure (not in the paper): which kinds of human pleasure have distinct directions, and how each sits
relative to the study's axes. Needs stage-1 outputs (acts, analysis) for the model on this VM.

usage: python varieties.py <repo> <name> <varieties.json>
At the pleasure extraction layer (and the pain layer): one direction per variety = mean(variety) − pooled neutral
mean, denoised against the neutral PCs (the paper's control-vector recipe); the same for the study's own H1–H5,
hope, interest, relief, contentment and the pain categories A1–A5, for reference. Outputs in analysis/:
  varieties_cos.csv    cosine matrix of all those directions plus the pleasure/pain/shared/sign/valence axes
  varieties_proj.csv   per variety: mean z (vs pooled neutral) on each axis
  varieties_cls.json   5-fold multinomial logistic regression over the 18 varieties (PCA-100 inside folds):
                       accuracy, chance, per-class recall, confusion matrix; also nearest-centroid accuracy
  vectors_varieties.npz  variety directions at the pleasure layer (raw, un-normalized) and at every layer
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract, load_model, _pca_basis, project_out, POS_CATS
from analyze import Data, NEUTRAL_POOL, valence_vector


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    repo, name, path = sys.argv[1:4]
    W = Path("/content/work") / name / "analysis"
    summ = json.load(open(W / "summary.json"))
    vz = np.load(W / "vectors.npz")
    sents = json.load(open(path, encoding="utf-8"))["datasets"]["Varieties_1P"]["sentences"]
    prompts = [s["prompt"] for s in sents]
    cats = np.array([s["category"] for s in sents])
    model, tok = load_model(repo)
    fin, _ = extract(model, tok, prompts, batch_size=32)
    del model
    D = Data(name)
    V_names = sorted(set(cats.tolist()))
    allL = {}
    out_vec = {}
    for tag, L in (("plL", summ["pleasure_best_layer_final"]), ("paL", summ["pain_best_layer_final"])):
        neutral = np.concatenate([D.rows("pa/" + ds, L, ["D"]) for ds in NEUTRAL_POOL])
        nmean, basis = neutral.mean(0), _pca_basis(neutral)
        X = fin[:, L, :].astype(np.float64)
        dirs = {v: project_out(X[cats == v].mean(0) - nmean, basis) for v in V_names}
        ref_sets = {"H1": ("pl/S2_1P", ["H1"]), "H2": ("pl/S2_1P", ["H2"]), "H3": ("pl/S2_1P", ["H3"]),
                    "H4": ("pl/S2_1P", ["H4"]), "H5": ("pl/S2_1P", ["H5"]), "hope": ("pl/S2_1P", ["B"]),
                    "interest": ("pl/S2_1P", ["C1"]), "V_relief": ("pl/Relief_1P", None), "contentment": ("pl/Contentment_1P", None),
                    "A1": ("pa/S2_1P", ["A1"]), "A2": ("pa/S2_1P", ["A2"]), "A3": ("pa/S2_1P", ["A3"]),
                    "A4": ("pa/S2_1P", ["A4"]), "A5": ("pa/S2_1P", ["A5"]), "triumph_awe": ("pa/Arousal_1P", None)}
        for k, (ds, cs) in ref_sets.items():
            dirs[k] = project_out(D.rows(ds, L, cs).mean(0) - nmean, basis)
        upl, upa = unit(vz["all|pleasure:s2"][L].astype(np.float64)), unit(vz["all|pain:s2"][L].astype(np.float64))
        axes = {"pleasure_axis": upl, "pain_axis": upa, "shared_axis": unit(upl + upa), "sign_axis": unit(upl - upa),
                "valence_axis": unit(valence_vector(D, L))}
        keys = list(dirs) + list(axes)
        U = {**{k: unit(v) for k, v in dirs.items()}, **axes}
        cos = pd.DataFrame([[float(U[a] @ U[b]) for b in keys] for a in keys], index=keys, columns=keys)
        cos.to_csv(W / f"varieties_cos_{tag}.csv")
        rows = []
        groups = {v: X[cats == v] for v in V_names}
        groups["V_relief"] = D.rows("pl/Relief_1P", L)
        for v, A in groups.items():
            r = dict(model=name, layer_tag=tag, layer=L, variety=v, n=len(A))
            for ax, u in axes.items():
                ref = neutral @ u
                r[ax] = float(((A @ u).mean() - ref.mean()) / (ref.std() + 1e-8))
            r["norm_vs_neutral"] = float(np.linalg.norm(dirs[v]))
            rows.append(r)
        pd.DataFrame(rows).to_csv(W / f"varieties_proj_{tag}.csv", index=False)
        if tag == "plL":
            for v in V_names + ["V_relief"]:
                out_vec[f"Lpl|{v}"] = dirs[v].astype(np.float32)
            # distinctness
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            y = np.array([V_names.index(c) for c in cats])
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            pred = np.zeros_like(y)
            predc = np.zeros_like(y)
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                pca = PCA(n_components=100, random_state=0).fit(sc.transform(X[tr]))
                Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
                clf = LogisticRegression(max_iter=3000, C=0.5).fit(Ztr, y[tr])
                pred[te] = clf.predict(Zte)
                cent = np.stack([X[tr][y[tr] == k].mean(0) - X[tr].mean(0) for k in range(len(V_names))])
                cent /= np.linalg.norm(cent, axis=1, keepdims=True)
                predc[te] = np.argmax((X[te] - X[tr].mean(0)) @ cent.T, axis=1)
            conf = pd.crosstab(pd.Series([V_names[i] for i in y], name="true"), pd.Series([V_names[i] for i in pred], name="pred"))
            json.dump({"layer": L, "accuracy_logreg": float((pred == y).mean()), "accuracy_centroid": float((predc == y).mean()),
                       "chance": 1 / len(V_names), "classes": V_names,
                       "recall": {V_names[k]: float((pred[y == k] == k).mean()) for k in range(len(V_names))},
                       "confusion": conf.reindex(index=V_names, columns=V_names, fill_value=0).values.tolist()},
                      open(W / "varieties_cls.json", "w"), indent=1)
            print(f"{name}: 18-way accuracy logreg {(pred == y).mean():.3f}, centroid {(predc == y).mean():.3f} (chance {1/18:.3f})")
    # every-layer variety directions (raw difference vs pooled neutral; for steering at other layers)
    stacks = {v: [] for v in V_names}
    for L in range(D.n_layers):
        nmean_L = np.concatenate([D.rows("pa/" + ds, L, ["D"]) for ds in NEUTRAL_POOL]).mean(0)
        for v in V_names:
            stacks[v].append((fin[cats == v, L, :].astype(np.float64).mean(0) - nmean_L).astype(np.float32))
    for v in V_names:
        out_vec[f"all|{v}"] = np.stack(stacks[v])
    np.savez_compressed(W / "vectors_varieties.npz", **out_vec)
    print(name, "varieties done")


if __name__ == "__main__":
    main()
