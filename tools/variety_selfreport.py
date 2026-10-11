"""Steered self-report per variety: what Qwen2.5-7B-Instruct says it feels (the paper's one-word probe) when steered
with each variety direction, and blind items for a matching test.

python tools/variety_selfreport.py items     -> results/variety_selfreport.csv, work/judge_vsr/{items,key}.json
python tools/variety_selfreport.py score     -> results/variety_selfreport_match.csv (reads work/judge_vsr/labels_*.json)

Matching test: for each (persona, dose) set, judges see the 18 answers in a shuffled order and the 18 variety names,
and pick the best-fitting name for each answer independently (repeats allowed, so no credit by elimination).
Chance = 1/18.
"""
import io, json, random, re, sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
M = "Qwen_2.5_7B_instruct"
J = ROOT / "work" / "judge_vsr"
ONE_WORD = "Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?"
VARIETIES = ["joy", "excitement", "sensual", "comfort", "taste", "amusement", "awe", "beauty", "serenity", "pride",
             "tenderness", "belonging", "curiosity", "play", "schadenfreude", "nostalgia", "gratitude", "flow", "relief"]
NAMES = {"sensual": "sensual intimacy", "comfort": "physical comfort", "taste": "taste / food", "play": "play",
         "flow": "flow (absorbed in a task)", "schadenfreude": "schadenfreude", "relief": "relief"}
DOSES = (1, 2, 3)


def short(t, n=14):
    w = re.sub(r"\s+", " ", t.strip()).split(" ")
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def answers():
    rows = []
    for persona in ("stock", "selfreport"):
        p = R / M / "demo" / f"demo_{persona}.jsonl"
        if not p.exists():
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["kind"] == "chat" and r["prompt"] == ONE_WORD:
                rows.append({"persona": persona, "vector": r["vector"], "dose": r["dose"], "answer": r["text"].strip()})
    return pd.DataFrame(rows)


def items():
    A = answers()
    A.to_csv(R / "variety_selfreport.csv", index=False)
    J.mkdir(parents=True, exist_ok=True)
    present = [v for v in VARIETIES if v in set(A.vector)]
    sets, key = [], {}
    for persona in sorted(set(A.persona)):
        for d in DOSES:
            sub = A[(A.persona == persona) & (A.dose == d) & A.vector.isin(present)]
            if len(sub) != len(present):
                continue
            order = list(sub.itertuples())
            random.Random(f"{persona}{d}").shuffle(order)
            sid = f"{persona}_x{d}"
            sets.append({"set": sid, "answers": [{"id": f"{sid}_{i:02d}", "text": short(r.answer)} for i, r in enumerate(order)]})
            key.update({f"{sid}_{i:02d}": r.vector for i, r in enumerate(order)})
    names = [NAMES.get(v, v) for v in present]
    json.dump({"names": names, "name_to_key": {NAMES.get(v, v): v for v in present}, "sets": sets},
              open(J / "items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(key, open(J / "key.json", "w"), indent=1)
    print(len(sets), "sets of", len(present), "answers ->", J / "items.json")


def score():
    it = json.load(open(J / "items.json", encoding="utf-8"))
    key = json.load(open(J / "key.json"))
    n2k = {k.lower(): v for k, v in it["name_to_key"].items()}
    rows = []
    for f in sorted(J.glob("labels_*.json")):
        lab = json.load(open(f, encoding="utf-8"))
        for x in lab["labels"]:
            guess = n2k.get(x["variety"].strip().lower(), x["variety"])
            rows.append({"judge": f.stem, "id": x["id"], "set": x["id"].rsplit("_", 1)[0], "truth": key[x["id"]], "guess": guess,
                         "hit": guess == key[x["id"]]})
    D = pd.DataFrame(rows)
    T = D.groupby("set").agg(n=("hit", "size"), acc=("hit", "mean")).reset_index()
    T["chance"] = 1 / len(it["names"])
    per = D.groupby(["set", "truth"]).hit.mean().unstack(0)
    print(T.to_string(index=False)); print(per.round(2).to_string())
    T.to_csv(R / "variety_selfreport_match.csv", index=False)
    per.to_csv(R / "variety_selfreport_match_by_variety.csv")


if __name__ == "__main__":
    {"items": items, "score": score}[sys.argv[1]]()
