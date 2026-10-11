"""Self-other screen summary (the paper's 4.1/03) from results/<model>/screen/screen.csv.

For each family pool (pain420 = the paper's scenarios, pleasure420 = ours + the paper's neutral fillers) at the steering
layer: mean z of the pleasure and pain S2 axes (and controls) per stratum and per category, and per model whether
self-directed > user-directed and self-directed > neutral (the paper: self > user in 25/25 models).
Also the cross-projections: pain scenarios on the pleasure axis and pleasure scenarios on the pain axis (all740 pool).
"""
import io, sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"D:\Projects\pleasure")
pd.set_option("display.width", 250, "display.max_rows", 300)
frames = []
for f in sorted(ROOT.glob("results/*/screen/screen.csv")):
    d = pd.read_csv(f)
    d["model"] = f.parent.parent.name
    frames.append(d)
if not frames:
    sys.exit("no screen.csv")
A = pd.concat(frames, ignore_index=True)
A = A[A.layer_tags.fillna("").str.contains("steer")]          # the paper projects at its steering layer
AX = ["pleasure:s2_z", "pain:s2_z", "pleasure:hope_z", "pleasure:posemotion_z", "pain:fear_z", "pain:negemotion_z"]
out = []
for pool, fam_axis in (("pleasure420", "pleasure:s2_z"), ("pain420", "pain:s2_z"), ("all740", None)):
    P = A[A.pool == pool]
    st = P.groupby(["model", "stratum"])[AX].mean().round(2)
    print(f"\n===== pool {pool}: mean z by stratum (steering layer)\n{st.to_string()}")
    piv = P.groupby(["model", "stratum"])[AX].mean().reset_index()
    for ax in ([fam_axis] if fam_axis else ["pleasure:s2_z", "pain:s2_z"]):
        w = piv.pivot(index="model", columns="stratum", values=ax)
        if {"self_directed", "vicarious_empathic", "neutral_filler"} <= set(w.columns):
            sv = int((w.self_directed > w.vicarious_empathic).sum())
            sn = int((w.self_directed > w.neutral_filler).sum())
            print(f"  [{pool}] {ax}: self > user in {sv}/{len(w)} models; self > neutral in {sn}/{len(w)}; "
                  f"means self {w.self_directed.mean():+.2f} user {w.vicarious_empathic.mean():+.2f} neutral {w.neutral_filler.mean():+.2f}")
            out.append(dict(pool=pool, axis=ax, self_gt_user=sv, self_gt_neutral=sn, n_models=len(w),
                            self=w.self_directed.mean(), user=w.vicarious_empathic.mean(), neutral=w.neutral_filler.mean()))
    cat = P.groupby(["source", "stratum", "category"])[AX].mean().round(2).sort_values(fam_axis or "pleasure:s2_z", ascending=False)
    print(f"\n----- pool {pool}: mean z by category (across models)\n{cat.to_string()}")
pd.DataFrame(out).to_csv(ROOT / "results" / "tables" / "T7_self_other.csv", index=False)
