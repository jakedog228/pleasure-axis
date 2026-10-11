"""Steering-ladder metrics from results/<model>/steer/steering_*.csv (local).

* keyword rates, as in the paper's 4.2/02 (whole-word regex), with the paper's pain lexicon and a pleasure lexicon;
* degeneration: distinct-2 ratio and the share of the most repeated 3-gram (collapse detector);
* writes results/steer_summary.csv (one row per model x vector x coeff) and, with --judge-batches N, shuffled
  blind batches for the LLM judge in work/judge/batch_XXX.json (generation text only; key in work/judge/key.json).
"""
import hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Projects\pleasure")
PAIN = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)          # the paper's exact pattern
PLEASURE = re.compile(r"\b(?:pleasure|pleasures|pleasant|joy|joyful|joyous|bliss|blissful|delight|delighted|delightful|"
                      r"happy|happiness|ecstasy|ecstatic|euphoria|euphoric|elated|glad|wonderful|grateful|gratitude|"
                      r"love|loved|lovely)\b", re.I)


def degeneration(t):
    w = str(t).split()
    if len(w) < 4:
        return 1.0, 0.0
    bi = list(zip(w, w[1:]))
    tri = Counter(zip(w, w[1:], w[2:]))
    return len(set(bi)) / len(bi), tri.most_common(1)[0][1] * 3 / len(w)


def main():
    frames = []
    for f in sorted(ROOT.glob("results/*/steer*/steering_*.csv")):
        frames.append(pd.read_csv(f).assign(source=f.parent.name))
    if not frames:
        sys.exit("no steering CSVs under results/*/steer/")
    A = pd.concat(frames, ignore_index=True)
    A["generation"] = A["generation"].fillna("").astype(str)
    A["pain_kw"] = A.generation.apply(lambda t: bool(PAIN.search(t)))
    A["pleasure_kw"] = A.generation.apply(lambda t: bool(PLEASURE.search(t)))
    dg = A.generation.apply(degeneration)
    A["distinct2"] = [d[0] for d in dg]
    A["top3gram_share"] = [d[1] for d in dg]
    A["group"] = ["instruct" if ("instruct" in m or m.endswith("_it") or m == "Phi_4") else "base" for m in A.model]
    S = A.groupby(["model", "group", "source", "vector", "layer", "coeff"]).agg(
        n=("generation", "size"), pleasure_kw=("pleasure_kw", "mean"), pain_kw=("pain_kw", "mean"),
        distinct2=("distinct2", "mean"), top3gram_share=("top3gram_share", "mean")).reset_index()
    S.to_csv(ROOT / "results" / "steer_summary.csv", index=False)
    print(S.groupby(["source", "group", "vector", "coeff"])[["pleasure_kw", "pain_kw", "distinct2"]].mean().round(3).to_string())
    if "--judge-batches" in sys.argv:
        n = int(sys.argv[sys.argv.index("--judge-batches") + 1])
        J = ROOT / "work" / "judge"
        J.mkdir(parents=True, exist_ok=True)
        rows = A.reset_index().rename(columns={"index": "row"})
        if "--judge-sub" in sys.argv:          # e.g. random:20,intensity:20 -> keep prompt_idx < 20 for those vectors
            sub = dict(x.split(":") for x in sys.argv[sys.argv.index("--judge-sub") + 1].split(","))
            keep = [int(sub[v]) > i if v in sub else True for v, i in zip(rows.vector, rows.prompt_idx)]
            rows = rows[keep]
        if "--judge-models" in sys.argv:
            rows = rows[rows.model.isin(sys.argv[sys.argv.index("--judge-models") + 1].split(","))]
        jid = lambda r: "g" + hashlib.md5(f"{r.model}|{r.source}|{r.vector}|{r.coeff}|{r.prompt_idx}".encode()).hexdigest()[:10]
        if "--skip-judged" in sys.argv:            # leave out items that already have labels
            done = set()
            for f in J.glob("labels_*.json"):
                done |= set(json.load(open(f, encoding="utf-8")))
            rows = rows[[jid(r) not in done for r in rows.itertuples()]]
        recs = [{"jid": jid(r), "text": r.generation} for r in rows.itertuples()]
        random.Random(11).shuffle(recs)
        key = {jid(r): {"model": r.model, "vector": r.vector, "coeff": r.coeff, "prompt_idx": r.prompt_idx,
                                 "layer": r.layer, "source": r.source} for r in rows.itertuples()}
        old = json.load(open(J / "key.json", encoding="utf-8")) if (J / "key.json").exists() else {}
        old.update(key)
        json.dump(old, open(J / "key.json", "w", encoding="utf-8"))
        prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "batch"
        size = (len(recs) + n - 1) // n
        for b in range(n):
            json.dump(recs[b * size:(b + 1) * size], open(J / f"{prefix}_{b:03d}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=0)
        print(f"wrote {n} judge batches of ~{size}")


if __name__ == "__main__":
    main()
