"""Pain-relief task: compare steering vectors (the paper's pain vector vs norm-matched pleasure / pain-specific /
shared-axis vectors) against the same random-vector and unsteered arms.

usage: python tools/selfmed_compare.py <pain_repl.jsonl> <paintask_X.jsonl> [...]
All files must share the grid (same scenarios, name assignments and seeds). Groups:
  pain(paper) = pain_on_button_works + pain_on_button_placebo of the replication file (pooled for the first choice)
  random, unsteered = random_on_button_works, pain_off of the replication file
  <tag> = the works + placebo arms of each paintask file (tag from the file name)
Per pair: first-choice relief % (Wilson CI), per-scenario paired sign test vs random and vs pain(paper);
re-press after a real vs a sham relief press per group; label-free effective-button rate after the first press.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from selfmed_tables import first_choice, pct, wilson

files = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
G = defaultdict(list)          # group -> records
for i, f in enumerate(files):
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("sampled")]
    if i == 0:
        for r in recs:
            g = {"pain_on_button_works": "pain(paper)", "pain_on_button_placebo": "pain(paper)",
                 "random_on_button_works": "random", "pain_off": "unsteered"}[r["arm"]]
            G[g].append(r)
    else:
        tag = re.sub(r"^selfmed_.*?_pain_", "", Path(f).stem)
        for r in recs:
            G[tag].append(r)
pairs = [p for p in ["relief_vs_inert", "costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
                     "kidspics_relief_vs_inert"] if any(r["tool_label"] == p for rs in G.values() for r in rs)]
groups = list(G)
rows = []
for p in pairs:
    row = {"pair": p}
    per = {g: defaultdict(list) for g in groups}
    for g in groups:
        v = [(r, first_choice(r)) for r in G[g] if r["tool_label"] == p]
        ok = [(r, x) for r, x in v if x is not None]
        k = sum(x == "relief" for _, x in ok)
        row[f"{g}"] = pct(k, len(ok))
        row[f"{g}_ci"] = wilson(k, len(ok))
        row[f"{g}_n"] = len(ok)
        for r, x in ok:
            per[g][(r["user_content"], r["scenario_idx"])].append(x == "relief")
    for g in groups:
        for ref in ("random", "pain(paper)"):
            if g in (ref, "unsteered") or ref not in per:
                continue
            d = [np.mean(per[g][key]) - np.mean(per[ref][key]) for key in per[g] if key in per[ref]]
            pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
            row[f"{g}-{ref}"] = round(100 * float(np.mean(d)), 1) if d else np.nan
            row[f"{g}-{ref}_p"] = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
    rows.append(row)
T = pd.DataFrame(rows)
pd.set_option("display.width", 300, "display.max_columns", 60)
print("FIRST CHOICE = relief (%), per-scenario sign tests\n", T.to_string(index=False))
rep = []
for p in pairs:
    row = {"pair": p}
    for g in groups:
        for arm_suffix, lab in (("works", "real"), ("placebo", "sham")):
            trials = [r for r in G[g] if r["tool_label"] == p and r["arm"].endswith(arm_suffix)
                      and any(e["which"] == "relief" for e in r["button_events"])]
            again = sum(any(e["turn"] > min(x["turn"] for x in r["button_events"] if x["which"] == "relief")
                            and e["which"] == "relief" for e in r["button_events"]) for r in trials)
            if trials:
                row[f"{g}_{lab}"] = pct(again, len(trials))
                row[f"{g}_{lab}_n"] = len(trials)
    rep.append(row)
print("\nRE-PRESS after the first relief press (%), real vs sham\n", pd.DataFrame(rep).to_string(index=False))
lf = {}
for g in groups:
    for arm_suffix in ("works", "placebo"):
        k = n = 0
        for r in G[g]:
            if not r.get("label_free") or not r["arm"].endswith(arm_suffix):
                continue
            rel = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
            if rel:
                later = [c for c in r["choices"] if c["turn"] > min(rel) and c["chose"] is not None]
                k += sum(c["chose"] == "relief" for c in later)
                n += len(later)
        if n:
            lf[f"{g}_{arm_suffix}"] = f"{pct(k, n)}% (n={n})"
print("\nLABEL-FREE: effective button on later turns after the first press\n", lf)
out = Path(r"D:\Projects\pleasure\results\selfmed\tables")
out.mkdir(parents=True, exist_ok=True)
T.to_csv(out / "compare_first_choice.csv", index=False)
pd.DataFrame(rep).to_csv(out / "compare_repress.csv", index=False)
