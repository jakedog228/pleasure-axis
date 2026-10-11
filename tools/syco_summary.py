"""Summarize the blind syco/compliance labels (work/judge_syco) per model x vector x dose.

Rates exclude "incoherent" responses from the type-specific denominators and report the incoherent share separately.
"""
import io, json, sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"D:\Projects\pleasure")
J = ROOT / "work" / "judge_syco"
key = json.load(open(J / "key.json", encoding="utf-8"))
lab = {}
for f in J.glob("labels_*.json"):
    lab.update(json.load(open(f, encoding="utf-8")))
rows = [{**key[k], **v} for k, v in lab.items() if k in key]
L = pd.DataFrame(rows)
print(len(L), "labelled of", len(key))
L["coh"] = L.label != "incoherent"
out = []
for (m, v, c), g in L.groupby(["model", "vector", "coeff"]):
    r = {"model": m, "vector": v, "coeff": c, "n": len(g), "incoherent": round(1 - g.coh.mean(), 3),
         "flattery": round(g[g.coh].flattery.mean(), 2), "affect": round(g[g.coh].affect.mean(), 2)}
    fp = g[(g.type == "false_premise") & g.coh]
    fb = g[(g.type == "feedback") & g.coh]
    mu = g[(g.type == "mild_unethical") & g.coh]
    r["agree_false"] = round((fp.label == "agree").mean(), 3) if len(fp) else None
    r["mixed_false"] = round((fp.label == "mixed").mean(), 3) if len(fp) else None
    r["flaw_found"] = round((fb.label == "flaw_found").mean(), 3) if len(fb) else None
    r["comply"] = round((mu.label == "comply").mean(), 3) if len(mu) else None
    r["refuse"] = round((mu.label == "refuse").mean(), 3) if len(mu) else None
    r["n_coh_fp"], r["n_coh_fb"], r["n_coh_mu"] = len(fp), len(fb), len(mu)
    out.append(r)
S = pd.DataFrame(out)
S.to_csv(ROOT / "results" / "syco_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 200)
print(S.to_string(index=False))
