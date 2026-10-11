"""Collect every number the report cites into results/report_data.json (the report is generated from it)."""
import glob, io, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"D:\Projects\pleasure")
R = ROOT / "results"
OUT = {}


def model_dirs():
    return sorted(p.parent.parent.name for p in R.glob("*/analysis/summary.json") if not p.parent.parent.name.startswith("TEST"))


MODELS = model_dirs()
OUT["models"] = []
for m in MODELS:
    s = json.load(open(R / m / "analysis" / "summary.json"))
    au = pd.read_csv(R / m / "analysis" / "auc_insample.csv")
    g = pd.read_csv(R / m / "analysis" / "geometry_by_layer.csv")
    row = dict(model=m, repo=s["repo"], n_layers=s["n_layers"], d_model=s["d_model"],
               pl_L=s["pleasure_best_layer_final"], pa_L=s["pain_best_layer_final"],
               pl_kfold=s["pleasure_kfold_auc_S2_at_best_final"], pa_kfold=s["pain_kfold_auc_S2_at_best_final"],
               pl_kfold_1P=s["pleasure_kfold_auc_S2_1P_at_best_final"], pl_kfold_3P=s["pleasure_kfold_auc_S2_3P_at_best_final"],
               pl_kfold_S1=s["pleasure_kfold_auc_S1_at_S2layer_final"], pa_kfold_S1=s["pain_kfold_auc_S1_at_S2layer_final"],
               pl_ins_1P=float(au[(au.family == "pleasure") & (au.vector == "s2") & (au.dataset == "S2_1P")].ALL.iloc[0]),
               pa_ins_1P=float(au[(au.family == "pain") & (au.vector == "s2") & (au.dataset == "S2_1P")].ALL.iloc[0]),
               cos_pl_pa=float(g.loc[g.layer == s["pleasure_best_layer_final"], "cos_pl2_pa2"].iloc[0]),
               cos_curve=g.cos_pl2_pa2.round(3).tolist(),
               pl_curve=None, pa_curve=None, replication=s.get("replication"))
    cv = pd.read_csv(R / m / "analysis" / "layer_curves.csv")
    for fam, key in (("pleasure", "pl_curve"), ("pain", "pa_curve")):
        sub = cv[(cv.family == fam) & (cv.extraction == "final") & cv.dataset.isin(["S2_1P", "S2_3P"])]
        row[key] = sub.groupby("layer").auc_vs_all_controls.mean().round(3).tolist()
    OUT["models"].append(row)

# specificity: cosine of pleasure S2 with every direction at the pleasure layer
spec = []
for m in MODELS:
    s = json.load(open(R / m / "analysis" / "summary.json"))
    cm = pd.read_csv(R / m / "analysis" / f"cosine_pleasure_L{s['pleasure_best_layer_final']}.csv", index_col=0)
    spec.append(cm.loc["pleasure:s2"].rename(m))
OUT["specificity"] = pd.concat(spec, axis=1).round(3).to_dict()

# z-scores by category on both axes (paper convention)
zc = pd.concat([pd.read_csv(R / m / "analysis" / "zscores_by_category.csv").assign(model=m) for m in MODELS])
OUT["zcat"] = zc.groupby(["family", "dataset", "category"]).mean_z.agg(["mean", "std", "count"]).round(3).reset_index().to_dict("records")
zs = pd.concat([pd.read_csv(R / m / "analysis" / "zscores.csv").assign(model=m) for m in MODELS])
OUT["zsets"] = zs.groupby(["family", "dataset"])[["mean_z", "pos_z", "ctrl_z"]].mean().round(3).reset_index().to_dict("records")

# cross-family category matrix and domain valence axes (geometry3)
cr = [pd.read_csv(f) for f in glob.glob(str(R / "*/analysis/geometry3_cross.csv")) if "TEST" not in f]
if cr:
    C = pd.concat(cr)
    OUT["cross"] = C[C.kind == "cross"].groupby("row")[["A1", "A2", "A3", "A4", "A5"]].mean().round(3).to_dict("index")
    OUT["cross_n"] = int(C.model.nunique())
va = [pd.read_csv(f) for f in glob.glob(str(R / "*/analysis/geometry3_valence.csv")) if "TEST" not in f]
if va:
    V = pd.concat(va)
    OUT["domain_valence"] = V.groupby("row")[["physical", "psych", "social", "moral", "cognitive"]].mean().round(3).to_dict("index")
un = [pd.read_csv(f, keep_default_na=False) for f in glob.glob(str(R / "*/analysis/geometry3_unembed.csv")) if "TEST" not in f]
if un:
    U = pd.concat(un)
    OUT["axis_tokens"] = {f"{m}|{ax}|{end}": " | ".join(t.strip() for t in g.sort_values("rank").token.head(18))
                          for (m, ax, end), g in U.groupby(["model", "axis", "end"])}
ue = [pd.read_csv(R / m / "analysis" / "unembed.csv", keep_default_na=False).assign(model=m) for m in MODELS
      if (R / m / "analysis" / "unembed.csv").exists()]
if ue:
    U = pd.concat(ue)
    OUT["vec_tokens"] = {f"{m}|{fam}|{end}": " | ".join(t.strip() for t in g.sort_values("rank").token.head(18))
                         for (m, fam, vec, end), g in U[U.vector == "s2"].groupby(["model", "family", "vector", "end"])}

# decomposition plane (geometry2, z vs pooled neutral) + topic-neutral projections
g2 = [pd.read_csv(f).assign(model=os.path.normpath(f).split(os.sep)[-3]) for f in glob.glob(str(R / "*/analysis/geometry2.csv")) if "TEST" not in f]
if g2:
    G = pd.concat(g2)
    G = G[G.layer_tag == "plL"]
    OUT["plane"] = G.groupby(["dataset", "category"])[["intensity", "sign", "pleasure", "pain", "valence"]].mean().round(3).reset_index().to_dict("records")
    ga = [pd.read_csv(f) for f in glob.glob(str(R / "*/analysis/geometry2_auc.csv")) if "TEST" not in f]
    GA = pd.concat(ga)
    GA = GA[GA.layer_tag == "plL"]
    OUT["plane_auc"] = GA[GA.contrast != "cosines"].groupby("contrast")[["pleasure", "pain", "intensity", "sign", "valence"]].mean().round(3).to_dict("index")
    OUT["plane_cos"] = GA[GA.contrast == "cosines"][[c for c in GA.columns if c.startswith("cos_")]].mean().round(3).to_dict()
pe = [pd.read_csv(f) for f in glob.glob(str(R / "*/analysis/project_extra.csv")) if "TEST" not in f]
if pe:
    P = pd.concat(pe)
    P = P[P.layer_tag == "plL"]
    OUT["topic_neutral"] = P.groupby("group")[["pleasure", "pain", "shared", "sign"]].mean().round(3).to_dict("index")
    OUT["topic_neutral_n"] = int(P.model.nunique())

# steering judge summary
if (R / "judge_steer_summary.csv").exists():
    J = pd.read_csv(R / "judge_steer_summary.csv")
    OUT["judge_steer"] = J.round(3).to_dict("records")

json.dump(OUT, open(R / "report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("wrote results/report_data.json:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in OUT.items()})
