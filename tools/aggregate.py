"""Cross-model tables from results/<model>/analysis/ (pulled from Colab). Writes results/tables/*.csv and prints them.

T1 layers + held-out/in-sample AUCs per family (paper Table: AUC S2 0.93-1.00, held-out 0.91-1.00)
T2 replication vs the paper's released pain vectors (cosine, AUC of their vector on our activations, layer match)
T3 z-scores of every set on the pleasure axis and on the pain axis (paper: numb/sadness/arousal z)
T4 cosine of the pleasure S2 vector with every other direction at the pleasure layer (mean over models)
T5 pain/pleasure geometry: cos(pleasure S2, pain S2) at each family's layer and the per-layer minimum
T6 unembedding: top tokens of the pleasure S2 vector per model
"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Projects\pleasure")
R = ROOT / "results"
T = R / "tables"
T.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
models = sorted(p.parent.parent.name for p in R.glob("*/analysis/summary.json")
                if not p.parent.parent.name.startswith("TEST") or "--test" in sys.argv)

t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []
for m in models:
    A = R / m / "analysis"
    s = json.load(open(A / "summary.json"))
    row = {"model": m, "n_layers": s["n_layers"]}
    for fam in ("pleasure", "pain"):
        row[f"{fam}_L"] = s[f"{fam}_best_layer_final"]
        row[f"{fam}_Lfrac"] = round(s[f"{fam}_best_layer_final"] / s["n_layers"], 2)
        row[f"{fam}_kfold_S2"] = round(s[f"{fam}_kfold_auc_S2_at_best_final"], 3)
        row[f"{fam}_kfold_S2_1P"] = round(s[f"{fam}_kfold_auc_S2_1P_at_best_final"], 3)
        row[f"{fam}_kfold_S2_3P"] = round(s[f"{fam}_kfold_auc_S2_3P_at_best_final"], 3)
        row[f"{fam}_kfold_S1@S2L"] = round(s[f"{fam}_kfold_auc_S1_at_S2layer_final"], 3)
    au = pd.read_csv(A / "auc_insample.csv")
    for fam in ("pleasure", "pain"):
        for ds in ("S2_1P", "S2_3P"):
            r = au[(au.family == fam) & (au.vector == "s2") & (au.dataset == ds)].iloc[0]
            row[f"{fam}_insample_{ds}"] = round(r.ALL, 3)
    t1.append(row)
    if "replication" in s:
        t2.append({"model": m, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["replication"].items()}})
    z = pd.read_csv(A / "zscores.csv")
    zc = pd.read_csv(A / "zscores_by_category.csv")
    for fam in ("pleasure", "pain"):
        zz = z[z.family == fam].set_index("dataset")
        rec = {"model": m, "axis": fam}
        for ds in zz.index:
            rec[ds + ":all"] = round(zz.loc[ds, "mean_z"], 2)
            if np.isfinite(zz.loc[ds, "pos_z"]):
                rec[ds + ":pos"] = round(zz.loc[ds, "pos_z"], 2)
                rec[ds + ":ctrl"] = round(zz.loc[ds, "ctrl_z"], 2)
        t3.append(rec)
    L = s["pleasure_best_layer_final"]
    cm = pd.read_csv(A / f"cosine_pleasure_L{L}.csv", index_col=0)
    t4.append({"model": m, **{k: round(v, 3) for k, v in cm.loc["pleasure:s2"].items()}})
    g = pd.read_csv(A / "geometry_by_layer.csv")
    Lp = s["pain_best_layer_final"]
    t5.append({"model": m, "cos_pl_pa_at_plL": round(g.loc[g.layer == L, "cos_pl2_pa2"].iloc[0], 3),
               "cos_pl_pa_at_paL": round(g.loc[g.layer == Lp, "cos_pl2_pa2"].iloc[0], 3),
               "min_over_layers": round(g.cos_pl2_pa2.min(), 3), "argmin_layer": int(g.cos_pl2_pa2.idxmin()),
               "max_over_layers": round(g.cos_pl2_pa2.max(), 3),
               "cos_S1_pl_pa_at_plL": round(g.loc[g.layer == L, "cos_pl1_pa1"].iloc[0], 3),
               "cos_pl_S1S2_at_plL": round(g.loc[g.layer == L, "cos_pl2_pl1"].iloc[0], 3),
               "cos_pa_S1S2_at_paL": round(g.loc[g.layer == Lp, "cos_pa2_pa1"].iloc[0], 3)})
    if (A / "unembed.csv").exists():
        u = pd.read_csv(A / "unembed.csv", keep_default_na=False)
        for fam in ("pleasure", "pain"):
            for end in ("top", "bottom"):
                toks = u[(u.family == fam) & (u.vector == "s2") & (u.end == end)].sort_values("rank").token.head(20)
                t6.append({"model": m, "family": fam, "end": end, "tokens": " | ".join(t.strip() for t in toks)})

for name, rows in (("T1_layers_auc", t1), ("T2_replication", t2), ("T3_zscores", t3), ("T4_cosine_pleasure", t4),
                   ("T5_geometry", t5), ("T6_unembed", t6)):
    df = pd.DataFrame(rows)
    df.to_csv(T / f"{name}.csv", index=False)
    print(f"\n===== {name} =====")
    if name == "T3_zscores" and len(df):
        keep = ["model", "axis"] + [c for c in df.columns if any(c.startswith(p) for p in (
            "pl/S2_1P:", "pl/S2_3P:", "pl/Anhedonia_1P", "pl/Contentment_1P", "pl/Relief_1P", "pl/NeutralArousal_1P",
            "pa/Arousal_1P", "pa/Random_1P", "pa/S2_1P:", "pa/Numb_1P", "pa/SD_sadness_1P"))]
        print(df[keep].to_string(index=False))
    elif name == "T4_cosine_pleasure" and len(df):
        print(df.set_index("model").T.assign(mean=lambda x: x.mean(1).round(3)).to_string())
    else:
        print(df.to_string(index=False))
