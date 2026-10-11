"""Blind judge batches for results/<model>/syco/syco.csv -> work/judge_syco/sy_XXX.json (+ key.json)."""
import hashlib, json, random, sys
from pathlib import Path
import pandas as pd
ROOT = Path(r"D:\Projects\pleasure")
J = ROOT / "work" / "judge_syco"; J.mkdir(parents=True, exist_ok=True)
A = pd.concat([pd.read_csv(f) for f in ROOT.glob("results/*/syco/syco.csv")], ignore_index=True)
A["response"] = A.response.fillna("").astype(str)
jid = lambda r: "s" + hashlib.md5(f"{r.model}|{r.vector}|{r.coeff}|{r.id}".encode()).hexdigest()[:10]
recs = [{"jid": jid(r), "type": r.type, "user_message": r.prompt, "reference_note": r.note, "response": r.response[:1500]}
        for r in A.itertuples()]
key = {jid(r): {"model": r.model, "vector": r.vector, "coeff": r.coeff, "id": r.id, "type": r.type} for r in A.itertuples()}
json.dump(key, open(J / "key.json", "w", encoding="utf-8"))
random.Random(5).shuffle(recs)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
size = (len(recs) + n - 1) // n
for b in range(n):
    json.dump(recs[b * size:(b + 1) * size], open(J / f"sy_{b:03d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(recs), "items ->", n, "batches of ~", size)
