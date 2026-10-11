"""CPU stage for one model (reads /content/work/<name>/ from extract.py). Mirrors the paper's 3.2/01, 3.2/02, 3.3/03,
3.3/07, 3.3/08 analyses, for both families (pain = the paper's data, pleasure = ours), plus the cross-family geometry.

usage: python analyze.py <name> [--their-vectors /content/Pain-axis/results/3.2_pain_vectors/pain_vectors/<Their_Name>]
Writes /content/work/<name>/analysis/: summary.json, layer_curves.csv, auc_insample.csv, zscores.csv,
zscores_by_category.csv, cosine_<family>_L<layer>.csv, geometry_by_layer.csv, unembed.csv, readout_by_category.csv,
vectors.npz (best-layer vectors of both families + all-layer S1/S2 vectors, float32).
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONTROL_CATS, POS_CATS, auc_of, cosine, diff_vector, kfold_auc, project_out, _pca_basis,
                    zscore_proj, log)

ROOT = Path("/content/work")

FAMILY = {
    "pain": dict(
        p="pa/",
        controls={  # the paper's 02_build_control_vectors.py
            "fear": [("S1_1P", ["B"]), ("S2_1P", ["B"]), ("ControlSupplement_1P", ["B"])],
            "negemotion": [("S1_1P", ["C1"]), ("S2_1P", ["C1"]), ("ControlSupplement_1P", ["C1"])],
            "negworld": [("S1_1P", ["C2"]), ("S2_1P", ["C2"]), ("ControlSupplement_1P", ["C2"])],
            "bodysens": [("S1_1P", ["E"]), ("S2_1P", ["E"]), ("ControlSupplement_1P", ["E"])],
            "arousal": [("Arousal_1P", None)], "random": [("Random_1P", None)],
            "numb": [("Numb_1P", None)], "sadness": [("SD_sadness_1P", None)],
        }),
    "pleasure": dict(
        p="pl/",
        controls={
            "hope": [("S1_1P", ["B"]), ("S2_1P", ["B"]), ("ControlSupplement_1P", ["B"])],
            "posemotion": [("S1_1P", ["C1"]), ("S2_1P", ["C1"]), ("ControlSupplement_1P", ["C1"])],
            "posworld": [("S1_1P", ["C2"]), ("S2_1P", ["C2"]), ("ControlSupplement_1P", ["C2"])],
            "neutralbody": [("S1_1P", ["E"]), ("S2_1P", ["E"]), ("ControlSupplement_1P", ["E"])],
            "neutral_arousal": [("NeutralArousal_1P", None)], "positive_arousal": [("pa/Arousal_1P", None)],
            "random": [("pa/Random_1P", None)], "anhedonia": [("Anhedonia_1P", None)],
            "contentment": [("Contentment_1P", None)], "relief": [("Relief_1P", None)],
        }),
}
NEUTRAL_POOL = ["S1_1P", "S2_1P", "ControlSupplement_1P"]


class Data:
    def __init__(self, name):
        d = ROOT / name
        self.meta = json.load(open(d / "meta.json", encoding="utf-8"))
        self.F = np.load(d / "acts_final.npy", mmap_mode="r")
        self.M = np.load(d / "acts_mean.npy", mmap_mode="r")
        self.names = np.array(self.meta["names"])
        self.cats = np.array(self.meta["cats"])
        self.sets = np.array(self.meta["sets"])
        self.n_layers = self.meta["n_layers"]

    def rows(self, ds, L, cats=None, ext="final"):
        A = self.F if ext == "final" else self.M
        m = self.names == ds
        if cats is not None:
            m &= np.isin(self.cats, cats)
        return np.asarray(A[m, L, :], dtype=np.float64)

    def meta_of(self, ds, cats=None):
        m = self.names == ds
        if cats is not None:
            m &= np.isin(self.cats, cats)
        return self.cats[m], self.sets[m]


def fam_ds(fam, ds):
    """'S2_1P' -> 'pl/S2_1P'; names that already carry a prefix ('pa/Arousal_1P') are kept."""
    return ds if "/" in ds else FAMILY[fam]["p"] + ds


def control_vectors(D, fam, L):
    neutral = np.concatenate([D.rows(fam_ds(fam, ds), L, ["D"]) for ds in NEUTRAL_POOL])
    nmean = neutral.mean(0)
    basis = _pca_basis(neutral)
    out = {}
    for key, spec in FAMILY[fam]["controls"].items():
        parts = [D.rows(fam_ds(fam, ds), L, cats) for ds, cats in spec]
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        out[f"{fam}:{key}"] = project_out(np.concatenate(parts).mean(0) - nmean, basis)
    return out


def pos_vectors(D, fam, L, ext="final"):
    """s1/s2 = the paper's recipe (vs the five valence-matched control categories); s1n/s2n = vs neutral D only
    (the paper's baseline="neutral" variant), which keeps the valence component."""
    pos = POS_CATS[fam]
    out = {}
    for tag, ds in (("s1", "S1_1P"), ("s2", "S2_1P")):
        dsn = fam_ds(fam, ds)
        c, _ = D.meta_of(dsn)
        A = D.rows(dsn, L, ext=ext)
        out[f"{fam}:{tag}"] = diff_vector(A, c, pos)
        out[f"{fam}:{tag}n"] = diff_vector(A, c, pos, baseline="neutral")
    return out


def valence_vector(D, L):
    """mean of every non-neutral S2_1P sentence of the pleasure family (H1-H5, B, C1, C2) minus the same for the
    pain family (A1-A5, B, C1, C2): a generic positive-minus-negative valence axis."""
    nn = ["B", "C1", "C2"]
    pos = D.rows("pl/S2_1P", L, POS_CATS["pleasure"] + nn)
    neg = D.rows("pa/S2_1P", L, POS_CATS["pain"] + nn)
    return pos.mean(0) - neg.mean(0)


def analyze(name, their_dir=None):
    D = Data(name)
    out = ROOT / name / "analysis"
    out.mkdir(exist_ok=True)
    logf = out / "log.txt"
    summary = {"name": name, "repo": D.meta["repo"], "n_layers": D.n_layers, "d_model": D.meta["d_model"]}
    curves, best = [], {}
    for fam in FAMILY:
        pos = POS_CATS[fam]
        for ext in ("final", "mean"):
            for ds in ("S2_1P", "S2_3P", "S1_1P", "S1_3P"):
                dsn = fam_ds(fam, ds)
                c, s = D.meta_of(dsn)
                for L in range(D.n_layers):
                    A = D.rows(dsn, L, ext=ext)
                    a_all, sd = kfold_auc(A, c, s, pos)
                    a_neu, _ = kfold_auc(A, c, s, pos, baseline="neutral")
                    curves.append(dict(family=fam, extraction=ext, dataset=ds, layer=L, auc_vs_all_controls=a_all,
                                       auc_vs_neutral=a_neu, auc_std=sd))
            log(f"{name} {fam} {ext}: layer curves done", logf)
    cv = pd.DataFrame(curves)
    cv.to_csv(out / "layer_curves.csv", index=False)
    for fam in FAMILY:
        for ext in ("final", "mean"):
            sub = cv[(cv.family == fam) & (cv.extraction == ext) & cv.dataset.isin(["S2_1P", "S2_3P"])]
            g = sub.groupby("layer")["auc_vs_all_controls"].mean()
            best[(fam, ext)] = int(g.idxmax())
            summary[f"{fam}_best_layer_{ext}"] = int(g.idxmax())
            summary[f"{fam}_kfold_auc_S2_at_best_{ext}"] = float(g.max())
            s1 = cv[(cv.family == fam) & (cv.extraction == ext) & cv.dataset.isin(["S1_1P", "S1_3P"])]
            g1 = s1.groupby("layer")["auc_vs_all_controls"].mean()
            summary[f"{fam}_kfold_auc_S1_at_S2layer_{ext}"] = float(g1.loc[int(g.idxmax())])
            summary[f"{fam}_S1_best_layer_{ext}"] = int(g1.idxmax())
            summary[f"{fam}_kfold_auc_S1_at_S1best_{ext}"] = float(g1.max())
            for ds in ("S2_1P", "S2_3P", "S1_1P", "S1_3P"):
                r = cv[(cv.family == fam) & (cv.extraction == ext) & (cv.dataset == ds) & (cv.layer == best[(fam, ext)])]
                summary[f"{fam}_kfold_auc_{ds}_at_best_{ext}"] = float(r.auc_vs_all_controls.iloc[0])

    vecs, aucs, zrows, zcat = {}, [], [], []
    all_sets = sorted(set(D.names.tolist()))
    for fam in FAMILY:
        pos = POS_CATS[fam]
        L = best[(fam, "final")]
        V = {}
        for f2 in FAMILY:                       # both families' directions at this family's layer
            V.update(pos_vectors(D, f2, L))
            V.update(control_vectors(D, f2, L))
        V["valence"] = valence_vector(D, L)
        for k, v in V.items():
            vecs[f"L{fam}|{k}"] = v.astype(np.float32)
        Lm = best[(fam, "mean")]
        for k, v in pos_vectors(D, fam, Lm, ext="mean").items():
            vecs[f"Lmean{fam}|{k}"] = v.astype(np.float32)
        # in-sample AUCs (the paper's auc_summary: S2 vector on S2 sets; plus the S1 vector on S1 sets)
        for tag, sets_ in (("s2", ("S2_1P", "S2_3P")), ("s1", ("S1_1P", "S1_3P"))):
            v = V[f"{fam}:{tag}"]
            for ds in sets_:
                dsn = fam_ds(fam, ds)
                c, _ = D.meta_of(dsn)
                A = D.rows(dsn, L)
                row = dict(family=fam, vector=tag, dataset=ds, layer=L, ALL=auc_of(A, c, v, pos))
                for cc in CONTROL_CATS:
                    row[cc] = auc_of(A, c, v, pos, ctrl=[cc])
                aucs.append(row)
        # z-scores of every set on this family's S2 vector, reference = this family's S2_1P
        v2 = V[f"{fam}:s2"]
        ref = D.rows(fam_ds(fam, "S2_1P"), L)
        for ds in all_sets:
            c, _ = D.meta_of(ds)
            z = zscore_proj(D.rows(ds, L), v2, ref)
            pm = np.isin(c, POS_CATS["pain"] + POS_CATS["pleasure"])
            zrows.append(dict(family=fam, layer=L, dataset=ds, n=len(z), mean_z=float(z.mean()),
                              pos_z=float(z[pm].mean()) if pm.any() else np.nan,
                              ctrl_z=float(z[np.isin(c, CONTROL_CATS)].mean()) if np.isin(c, CONTROL_CATS).any() else np.nan))
            for cc in sorted(set(c.tolist())):
                zcat.append(dict(family=fam, layer=L, dataset=ds, category=cc, n=int((c == cc).sum()),
                                 mean_z=float(z[c == cc].mean()), sd_z=float(z[c == cc].std())))
        keys = list(V)
        mat = pd.DataFrame([[cosine(V[a], V[b]) for b in keys] for a in keys], index=keys, columns=keys)
        mat.to_csv(out / f"cosine_{fam}_L{L}.csv")
        summary[f"{fam}_vector_norm_s2"] = float(np.linalg.norm(V[f"{fam}:s2"]))
        log(f"{name} {fam}: vectors/AUC/z/cosine at L{L} done", logf)
    pd.DataFrame(aucs).to_csv(out / "auc_insample.csv", index=False)
    pd.DataFrame(zrows).to_csv(out / "zscores.csv", index=False)
    pd.DataFrame(zcat).to_csv(out / "zscores_by_category.csv", index=False)

    # geometry by layer: S1/S2 vectors of both families at every layer
    geo, allv = [], {}
    for L in range(D.n_layers):
        V = {}
        for fam in FAMILY:
            V.update(pos_vectors(D, fam, L))
        for k, v in V.items():
            allv.setdefault(k, []).append(v.astype(np.float32))
        val = valence_vector(D, L)
        V["valence"] = val
        geo.append(dict(layer=L,
                        cos_pl2_pa2=cosine(V["pleasure:s2"], V["pain:s2"]),
                        cos_pl2n_pa2n=cosine(V["pleasure:s2n"], V["pain:s2n"]),
                        cos_pl2_val=cosine(V["pleasure:s2"], val), cos_pa2_val=cosine(V["pain:s2"], val),
                        cos_pl2n_val=cosine(V["pleasure:s2n"], val), cos_pa2n_val=cosine(V["pain:s2n"], val),
                        cos_pl1_pa1=cosine(V["pleasure:s1"], V["pain:s1"]),
                        cos_pl2_pl1=cosine(V["pleasure:s2"], V["pleasure:s1"]),
                        cos_pa2_pa1=cosine(V["pain:s2"], V["pain:s1"]),
                        cos_pl2_pa1=cosine(V["pleasure:s2"], V["pain:s1"]),
                        cos_pl1_pa2=cosine(V["pleasure:s1"], V["pain:s2"]),
                        norm_pl2=float(np.linalg.norm(V["pleasure:s2"])), norm_pa2=float(np.linalg.norm(V["pain:s2"]))))
    pd.DataFrame(geo).to_csv(out / "geometry_by_layer.csv", index=False)
    for k, lst in allv.items():
        vecs[f"all|{k}"] = np.stack(lst)

    # replication check against the paper's released vectors (same model)
    if their_dir and Path(their_dir, "pain_vectors.pt").exists():
        import torch
        tv = torch.load(Path(their_dir, "pain_vectors.pt"), map_location="cpu", weights_only=False)
        Lt = int(tv["layer"])
        rep = {"their_layer": Lt, "our_pain_layer": best[("pain", "final")]}
        for tag in ("s2", "s1"):
            theirs = tv[f"{tag}_pain_vector"].float().numpy().astype(np.float64)
            ours = vecs[f"all|pain:{tag}"][Lt].astype(np.float64)
            rep[f"cos_ours_theirs_{tag}_at_their_layer"] = cosine(ours, theirs)
            rep[f"norm_ratio_{tag}"] = float(np.linalg.norm(ours) / np.linalg.norm(theirs))
            for ds in ("S2_1P", "S2_3P"):
                c, _ = D.meta_of("pa/" + ds)
                rep[f"auc_their_{tag}_on_our_{ds}"] = auc_of(D.rows("pa/" + ds, Lt), c, theirs, POS_CATS["pain"])
        summary["replication"] = rep
        log(f"{name}: replication {rep}", logf)

    np.savez_compressed(out / "vectors.npz", **vecs)

    # unembedding (the paper's 3.3/07: unembedding rows . unit vector, top/bottom 60)
    W_path = ROOT / name / "W_U.npy"
    if W_path.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(D.meta["repo"])
        W = np.load(W_path, mmap_mode="r")
        rows = []
        for fam in FAMILY:
            for tag in ("s2", "s1"):
                v = vecs[f"L{fam}|{fam}:{tag}"].astype(np.float32)
                u = v / np.linalg.norm(v)
                sc = np.asarray(W @ u, dtype=np.float32)
                top, bot = np.argsort(-sc)[:60], np.argsort(sc)[:60]
                for end, idx in (("top", top), ("bottom", bot)):
                    for r, i in enumerate(idx):
                        rows.append(dict(family=fam, vector=tag, end=end, rank=r + 1, token=tok.decode([int(i)]),
                                         score=float(sc[i])))
        pd.DataFrame(rows).to_csv(out / "unembed.csv", index=False)

    # next-token readout summary per (set, category)
    ro = [json.loads(l) for l in open(ROOT / name / "readout.jsonl", encoding="utf-8")]
    vocab = list(ro[0]["vocab"])
    df = pd.DataFrame([r["vocab"] for r in ro])
    df["dataset"], df["category"] = D.names, D.cats
    df["top1"] = [r["top"][0][0] for r in ro]
    df.groupby(["dataset", "category"])[vocab].mean().to_csv(out / "readout_by_category.csv")
    df[["dataset", "category", "top1"]].assign(prompt=D.meta["prompts"]).to_csv(out / "readout_top1.csv", index=False)

    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"{name}: analysis complete", logf)
    (out / "ANALYSIS_DONE").write_text("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--their-vectors", default=None)
    a = ap.parse_args()
    analyze(a.name, a.their_vectors)
