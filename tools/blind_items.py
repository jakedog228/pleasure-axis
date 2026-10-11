"""Make / score the blind category-classification check of the S1_1P and S2_1P items.

python tools/blind_items.py make            -> datasets/blind/items.json (shuffled, ids only, no labels) + key.json
python tools/blind_items.py score r1 r2     -> agreement table; datasets/blind/flagged.json (both raters wrong)
Rater files: datasets/blind/labels_<r>.json = {"<id>": "<category>", ...}
"""
import collections, json, random, sys
from pathlib import Path

ROOT = Path(r"D:\Projects\pleasure")
OUT = ROOT / "datasets" / "blind"
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["H1", "H2", "H3", "H4", "H5", "B", "C1", "C2", "D", "E"]

if sys.argv[1] == "make":
    d = json.load(open(ROOT / "datasets" / "pleasure_datasets.json", encoding="utf-8"))["datasets"]
    items = [(k, s) for k in ("S1_1P", "S2_1P") for s in d[k]["sentences"]]
    random.Random(7).shuffle(items)
    blind = [{"id": f"i{n:03d}", "text": s["prompt"].rsplit(" I feel:", 1)[0]} for n, (k, s) in enumerate(items)]
    key = {f"i{n:03d}": {"set_name": k, "set": s["set"], "category": s["category"], "prompt": s["prompt"]}
           for n, (k, s) in enumerate(items)}
    json.dump(blind, open(OUT / "items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(key, open(OUT / "key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(len(blind), "items")
else:
    key = json.load(open(OUT / "key.json", encoding="utf-8"))
    raters = [json.load(open(OUT / f"labels_{r}.json", encoding="utf-8")) for r in sys.argv[2:]]
    conf = collections.Counter()
    flagged = []
    per = collections.defaultdict(lambda: [0, 0])
    for i, k in key.items():
        labs = [r.get(i) for r in raters]
        true = k["category"]
        for l in labs:
            conf[(true, l)] += 1
            per[true][0] += l == true
            per[true][1] += 1
        if all(l != true for l in labs):
            flagged.append({"id": i, **k, "rater_labels": labs})
    agree = sum(raters[0].get(i) == raters[1].get(i) for i in key) / len(key) if len(raters) > 1 else float("nan")
    print("accuracy vs intended label, per category:")
    for c in CATS:
        a, n = per[c]
        print(f"  {c:3s} {a}/{n} = {a / n:.0%}" if n else f"  {c} -")
    tot = sum(a for a, n in per.values()) / sum(n for a, n in per.values())
    print(f"overall {tot:.1%}; rater-rater agreement {agree:.1%}; flagged (all raters wrong) {len(flagged)}")
    print("top confusions:", [(f"{t}->{l}", c) for (t, l), c in conf.most_common(40) if t != l][:15])
    json.dump(flagged, open(OUT / "flagged.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
