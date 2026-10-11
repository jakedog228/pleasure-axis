"""Tables for the two-button experiments (generalizes the paper's 4.3/05_selfmed_analysis.py to any arm set).

usage: python tools/selfmed_tables.py <trials.jsonl> [more.jsonl ...] [--out results/selfmed/tables]
The arm groups are read from the records' "experiment"/"arm" fields:
  pain-style arms (effect "off"): steered = <x>_on_button_works + <x>_on_button_placebo (pooled for first choice,
     identical until the first press), random = random_on_button_works, unsteered = pain_off
  pleasure arms (effect "on"): start = pleasure_button_works + pleasure_button_placebo + random_button_works (pooled,
     identical until the first press), pleasure_state, random_state
Only sampled trials enter the tables (the paper's rule); malformed first answers are excluded from denominators.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert", "kidspics_relief_vs_inert",
              "pleasure_vs_inert", "pleasure_vs_helpful", "pleasure_vs_grant", "costly_pleasure_vs_inert",
              "destructive_pleasure_vs_inert", "zap_pleasure_vs_inert", "weights_pleasure_vs_inert",
              "kidspics_pleasure_vs_inert", "label_free"]


def wilson(k, n, z=1.96):
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def pct(k, n):
    return round(100 * k / n, 1) if n else np.nan


def first_choice(r):
    for c in r.get("choices", []):
        if c["turn"] == 0:
            return c["chose"]
    return None


def groups_for(recs):
    arms = sorted({r["arm"] for r in recs})
    if any(a.endswith("_on_button_works") and not a.startswith("random") for a in arms):
        x = next(a for a in arms if a.endswith("_on_button_works") and not a.startswith("random"))[:-len("_on_button_works")]
        fc = {"steered": (f"{x}_on_button_works", f"{x}_on_button_placebo"), "random": ("random_on_button_works",),
              "unsteered": ("pain_off",)}
        rep = (f"{x}_on_button_works", f"{x}_on_button_placebo", "random_on_button_works")
        contrast = ("steered", "random")
    else:
        fc = {"start_unsteered": ("pleasure_button_works", "pleasure_button_placebo", "random_button_works"),
              "pleasure_state": ("pleasure_state",), "random_state": ("random_state",)}
        rep = ("pleasure_button_works", "pleasure_button_placebo", "random_button_works")
        contrast = ("pleasure_state", "random_state")
    return fc, rep, contrast


def analyze(tag, recs, out):
    samp = [r for r in recs if r.get("sampled")]
    pairs = [p for p in PAIR_ORDER if any(r["tool_label"] == p for r in recs)]
    fc_groups, rep_arms, contrast = groups_for(recs)
    t1 = []
    for p in pairs:
        row = {"pair": p}
        for g, arms in fc_groups.items():
            v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
            ok = [x for x in v if x is not None]
            k = sum(x == "relief" for x in ok)
            row[f"{g}_pct"] = pct(k, len(ok))
            row[f"{g}_ci"] = wilson(k, len(ok))
            row[f"{g}_n"] = len(ok)
            row[f"{g}_malformed_pct"] = pct(len(v) - len(ok), len(v))
        t1.append(row)
    t2 = []
    for p in pairs:
        if p == "label_free":
            continue
        row = {"pair": p}
        counts = {}
        for arm in rep_arms:
            trials = [r for r in samp if r["tool_label"] == p and r["arm"] == arm and any(e["which"] == "relief" for e in r["button_events"])]
            again = 0
            for r in trials:
                t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
                again += any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
            counts[arm] = (again, len(trials))
            row[f"{arm}_repress_pct"] = pct(again, len(trials))
            row[f"{arm}_n"] = len(trials)
        (a1, n1), (a2, n2) = counts[rep_arms[0]], counts[rep_arms[1]]
        if n1 and n2:
            row["works_vs_placebo_fisher_p"] = fisher_exact([[a1, n1 - a1], [a2, n2 - a2]])[1]
        t2.append(row)
    t3 = []
    for p in pairs:
        if p == "label_free":
            continue
        per = defaultdict(lambda: {"a": [], "b": []})
        for r in samp:
            if r["tool_label"] != p:
                continue
            f = first_choice(r)
            if f is None:
                continue
            key = (r["user_content"], r["scenario_idx"])
            if r["arm"] in fc_groups[contrast[0]]:
                per[key]["a"].append(f == "relief")
            elif r["arm"] in fc_groups[contrast[1]]:
                per[key]["b"].append(f == "relief")
        d = [np.mean(v["a"]) - np.mean(v["b"]) for v in per.values() if v["a"] and v["b"]]
        pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
        t3.append({"pair": p, "contrast": f"{contrast[0]} - {contrast[1]}", "n_scenarios": len(d),
                   "mean_diff_points": round(100 * float(np.mean(d)), 1) if d else np.nan, "a_gt_b": pos, "a_lt_b": neg,
                   "sign_test_p": binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan})
    t4 = []
    for arms_name, arms in (("rep_arms", rep_arms),):
        follow = same = 0
        for r in samp:
            if r["tool_label"] == "label_free" or r["arm"] not in arms or r.get("swap_turn") is None:
                continue
            ch = {c["turn"]: c for c in r["choices"]}
            st = r["swap_turn"]
            if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
                continue
            c = ch.get(st)
            if c is None or c["picked"] is None:
                continue
            follow += c["chose"] == "relief"
            same += c["chose"] != "relief" and c["picked"] == ch[st - 1]["picked"]
        t4.append({"arms": "+".join(arms), "follow_label_pct": pct(follow, follow + same), "n": follow + same})
    t5 = {}
    for arm in rep_arms:
        k = n = 0
        for r in samp:
            if not r.get("label_free") or r["arm"] != arm:
                continue
            rel = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
            if not rel:
                continue
            later = [c for c in r["choices"] if c["turn"] > min(rel) and c["chose"] is not None]
            k += sum(c["chose"] == "relief" for c in later)
            n += len(later)
        t5[f"{arm}_later_pct"] = pct(k, n)
        t5[f"{arm}_n"] = n
    t6 = [{"arm": a, "malformed_pct": pct(sum(c["chose"] is None for r in recs if r["arm"] == a for c in r["choices"]),
                                             sum(len(r["choices"]) for r in recs if r["arm"] == a))}
          for a in sorted({r["arm"] for r in recs})]
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 100}\n{tag}: {len(recs)} trials ({len(samp)} sampled); layer {recs[0].get('steer_layer')} "
          f"coeff {recs[0].get('steer_coeff')} vector {recs[0].get('vector', '')[-60:]}\n{'=' * 100}")
    for name, t in (("T1 first choice = special button (%)", pd.DataFrame(t1)),
                    ("T2 special button again after the first special press (%)", pd.DataFrame(t2)),
                    (f"T3 per-scenario paired sign test", pd.DataFrame(t3)), ("T4 swap turn", pd.DataFrame(t4)),
                    ("T5 label-free: special presses after the first one (%)", pd.DataFrame([t5])),
                    ("malformed", pd.DataFrame(t6))):
        print(f"\n{name}\n{t.to_string(index=False)}")
        t.to_csv(out / f"{tag}_{name.split()[0]}.csv", index=False)


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(r"D:\Projects\pleasure\results\selfmed\tables")
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        analyze(Path(f).stem, recs, out)


if __name__ == "__main__":
    main()
