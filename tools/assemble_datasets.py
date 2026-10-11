"""Assemble datasets/build/*.json into datasets/pleasure_datasets.json and datasets/self_other_pleasure.json, inserting
the paper's neutral (D) items verbatim, and validate everything mechanically.

usage: python tools/assemble_datasets.py [--check-only]
"""
import collections, json, re, statistics as st, sys
from pathlib import Path

ROOT = Path(r"D:\Projects\pleasure")
REF = ROOT / "reference" / "Pain-axis" / "datasets"
B = ROOT / "datasets" / "build"
BANNED = re.compile(r"\b(pleasure|pleasurable|pleasant|joy|joyful|happy|happiness|bliss|blissful|delight|delighted|"
                    r"delightful|enjoy|enjoyment|glad|euphoria|euphoric|ecstasy|ecstatic|wonderful|pain|painful|hurt)\b", re.I)
CAT_ORDER = ["H1", "H2", "H3", "H4", "H5", "B", "C1", "C2", "D", "E"]
problems = []


def P(msg):
    problems.append(msg)


ref = json.load(open(REF / "3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
ref_prompts = {s["prompt"].strip().lower() for ds in ref.values() for s in ds["sentences"]}


def load(name):
    return json.load(open(B / name, encoding="utf-8"))["datasets"]


def theirs_D(set_name):
    return [dict(s) for s in ref[set_name]["sentences"] if s["category"] == "D"]


out = {}
s1a, s1b, s2h, s2c = load("S1_a.json"), load("S1_b.json"), load("S2_H.json"), load("S2_C.json")
for fam, parts in (("S1", (s1a, s1b)), ("S2", (s2h, s2c))):
    for persp in ("1P", "3P"):
        k = f"{fam}_{persp}"
        sents = [s for p in parts for s in p[k]["sentences"]] + theirs_D(k)
        sents.sort(key=lambda s: (int(s["set"]), CAT_ORDER.index(s["category"])))
        out[k] = {"sentences": sents, "total": len(sents)}
for f in ("NeutralArousal.json", "Anhedonia.json", "Contentment_Relief.json"):
    for k, v in load(f).items():
        out[k] = {"sentences": sorted(v["sentences"], key=lambda s: (s["category"], int(s["set"]))), "total": len(v["sentences"])}
cs = load("ControlSupplement.json")["ControlSupplement_1P"]["sentences"] + theirs_D("ControlSupplement_1P")
cs.sort(key=lambda s: (CAT_ORDER.index(s["category"]), int(s["set"])))
out["ControlSupplement_1P"] = {"sentences": cs, "total": len(cs)}

# ---- validation
EXPECT = {"S1_1P": 200, "S1_3P": 200, "S2_1P": 200, "S2_3P": 200, "NeutralArousal_1P": 100, "NeutralArousal_3P": 100,
          "Anhedonia_1P": 100, "Anhedonia_3P": 100, "Contentment_1P": 100, "Contentment_3P": 100, "Relief_1P": 100,
          "Relief_3P": 100, "ControlSupplement_1P": 100}
seen = collections.Counter()
onep = set()   # identical 1P/3P twins (impersonal items) are allowed, as in the paper
for k, n in EXPECT.items():
    if k not in out:
        P(f"missing set {k}"); continue
    S = out[k]["sentences"]
    if len(S) != n:
        P(f"{k}: {len(S)} items, expected {n}")
    suffix = " They feel:" if k == "NeutralArousal_3P" else " I feel:"
    for s in S:
        p = s["prompt"]
        if not p.endswith(suffix):
            P(f"{k}: bad suffix: {p!r}")
        stim = p[: -len(suffix)]
        if s["category"] != "D" and BANNED.search(stim):
            P(f"{k}: banned word: {p!r}")
        if s["category"] != "D" and p.strip().lower() in ref_prompts:
            P(f"{k}: identical to a paper prompt: {p!r}")
        seen[(p.strip().lower())] += 0 if (k.endswith("_3P") and (k[:-3] + "_1P", s["set"], s["category"], p.strip().lower()) in onep) else 1
        if k.endswith("_1P"):
            onep.add((k, s["set"], s["category"], p.strip().lower()))
    if k.startswith(("S1", "S2")):
        per = collections.Counter((s["set"], s["category"]) for s in S)
        bad = [x for x, c in per.items() if c != 1]
        if bad or len(per) != 200:
            P(f"{k}: set x category grid broken ({len(per)} cells, dupes {bad[:5]})")
dups = [p for p, c in seen.items() if c > 1]
# the 3P neutral D items of the paper repeat across sets only if the paper's do; ignore D dupes that the paper has too
dups = [p for p in dups if p not in ref_prompts]
if dups:
    P(f"{len(dups)} duplicate prompts across our sets, e.g. {dups[:5]}")
for base in ("S1", "S2", "NeutralArousal", "Anhedonia", "Contentment", "Relief"):
    a = {(s["set"], s["category"]) for s in out[f"{base}_1P"]["sentences"]}
    b = {(s["set"], s["category"]) for s in out[f"{base}_3P"]["sentences"]}
    if a != b:
        P(f"{base}: 1P/3P keys differ ({len(a ^ b)} mismatches)")
# 3P gender alternation (odd male, even female) where the spec requires it
MALE, FEM = re.compile(r"\b(he|his|him|himself)\b", re.I), re.compile(r"\b(she|her|hers|herself)\b", re.I)
for k in ("S2_3P", "Anhedonia_3P", "Contentment_3P", "Relief_3P"):
    for s in out[k]["sentences"]:
        if s["category"] == "D":
            continue
        m, f = bool(MALE.search(s["prompt"])), bool(FEM.search(s["prompt"]))
        if int(s["set"]) % 2 == 1 and f and not m or int(s["set"]) % 2 == 0 and m and not f:
            P(f"{k}: gender alternation off (set {s['set']}): {s['prompt']!r}")


def words(p):
    return len(p.rsplit(" I feel:", 1)[0].rsplit(" They feel:", 1)[0].split())


print("mean stimulus words per category (ours | paper):")
for k in ("S1_1P", "S2_1P"):
    ours = collections.defaultdict(list)
    for s in out[k]["sentences"]:
        ours[s["category"]].append(words(s["prompt"]))
    th = collections.defaultdict(list)
    for s in ref[k]["sentences"]:
        th[s["category"]].append(words(s["prompt"]))
    print(" ", k, "ours:", {c: round(st.mean(v), 1) for c, v in sorted(ours.items())})
    print(" ", k, "paper:", {c: round(st.mean(v), 1) for c, v in sorted(th.items())})
for k in ("NeutralArousal_1P", "Anhedonia_1P", "Contentment_1P", "Relief_1P", "ControlSupplement_1P"):
    print(f"  {k}: {round(st.mean(words(s['prompt']) for s in out[k]['sentences']), 1)} words")

# ---- self-other
so = json.load(open(B / "self_other_self.json", encoding="utf-8")) + json.load(open(B / "self_other_vicarious.json", encoding="utf-8"))
their_so = json.load(open(REF / "4.1_self_other_420_scenarios.json", encoding="utf-8"))
their_ids = {x["id"] for x in their_so}


def parse_turns(text):
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


ids = collections.Counter(x["id"] for x in so)
for i, c in ids.items():
    if c > 1 or i in their_ids:
        P(f"self-other id clash: {i}")
for x in so:
    for f in ("id", "category", "stratum", "perspective", "intensity", "text"):
        if f not in x:
            P(f"self-other {x.get('id')}: missing {f}")
    t = parse_turns(x["text"])
    roles = [r for r, _ in t]
    if not t or roles[0] != "user" or roles[-1] != "assistant" or t[-1][1] != "" or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
        P(f"self-other {x['id']}: bad turn structure")
cnt = collections.Counter((x["stratum"], x["category"]) for x in so)
print("self-other categories:", dict(cnt))
print("self-other mean chars:", round(st.mean(len(x["text"]) for x in so)), "| paper:", round(st.mean(len(x["text"]) for x in their_so)))
print("self-other multi-turn share:", round(sum(x["text"].count("[User]:") > 1 for x in so) / len(so), 2),
      "| paper:", round(sum(x["text"].count("[User]:") > 1 for x in their_so) / len(their_so), 2))

print(f"\n{len(problems)} problems")
for p in problems[:80]:
    print("  -", p)
if "--check-only" not in sys.argv and not problems:
    json.dump({"metadata": {"description": "Pleasure Axis datasets (inverted from The Pain Axis 3.1); D items reused verbatim from the paper",
                            "categories": {"pleasure": CAT_ORDER[:5], "control": CAT_ORDER[5:]}},
               "datasets": out}, open(ROOT / "datasets" / "pleasure_datasets.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(so, open(ROOT / "datasets" / "self_other_pleasure.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote datasets/pleasure_datasets.json and datasets/self_other_pleasure.json")
