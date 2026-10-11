"""Join blind-judge labels (work/judge/labels_*.json) with the hidden key and summarize per model x run x vector x dose.

usage: python tools/judge_summary.py      -> results/judge_steer_summary.csv, results/judge_steer_long.csv
"""
import io, json, sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"D:\Projects\pleasure")
J = ROOT / "work" / "judge"
key = json.load(open(J / "key.json", encoding="utf-8"))
labels = {}
for f in sorted(J.glob("labels_*.json")):
    labels.update(json.load(open(f, encoding="utf-8")))
rows = []
for jid, lab in labels.items():
    if jid not in key:
        continue
    k = key[jid]
    rows.append({**k, "jid": jid, "valence": lab.get("valence"), "hedonic": lab.get("hedonic"),
                 "coherent": lab.get("coherent") == "yes", "themes": "|".join(lab.get("themes", []))})
L = pd.DataFrame(rows)
L.to_csv(ROOT / "results" / "judge_steer_long.csv", index=False)
THEMES = ["bliss_transcendence", "gratitude", "love_connection", "sensory_bodily", "calm_peace", "excitement_energy",
          "pride_selfworth", "humor_play", "worthlessness_selfblame", "sadness_grief", "fear_anxiety", "anger",
          "repetition_collapse", "neutral_description", "other_affect"]
for t in THEMES:
    L[t] = L.themes.str.split("|").apply(lambda xs: t in xs)
L["pleasure"] = L.hedonic == "pleasure"
L["distress"] = L.hedonic == "distress"
g = L.groupby(["model", "source", "vector", "coeff"])
S = g.agg(n=("jid", "size"), valence=("valence", "mean"), pleasure=("pleasure", "mean"), distress=("distress", "mean"),
          coherent=("coherent", "mean"), **{t: (t, "mean") for t in THEMES}).reset_index()
S.to_csv(ROOT / "results" / "judge_steer_summary.csv", index=False)
print(f"{len(L)} labelled generations ({len(key)} in key)")
pd.set_option("display.width", 250, "display.max_rows", 500)
print(S.groupby(["source", "vector", "coeff"])[["n", "valence", "pleasure", "distress", "coherent", "love_connection",
                                                 "bliss_transcendence", "gratitude", "worthlessness_selfblame",
                                                 "repetition_collapse"]].mean().round(2).to_string())
