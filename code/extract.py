"""GPU stage for one model: activations at every layer for every dataset of both families, plus the next-token readout
(top-20 + an emotion vocabulary, from the same forward pass) and the unembedding matrix's top/bottom tokens for the
vectors computed later (saved: lm_head is re-read in analyze_unembed if needed).

usage: python extract.py <repo> <name> [--batch 32]
Writes /content/work/<name>/: acts_final.npy, acts_mean.npy (float16 [N, L, D]), meta.json (set names, categories,
sets, prompts), readout.jsonl, W_U.npy (unembedding matrix, float16; used by analyze.py, deletable afterwards).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract, load_model, load_sets, log, get_layers

ROOT = Path("/content")
PAIN_PATHS = [ROOT / "Pain-axis/datasets/3.1_pain_and_control_datasets.json", ROOT / "Pain-axis/datasets/3.1_sadness_dataset.json"]
PLEASURE_PATHS = [ROOT / "pleasure/datasets/pleasure_datasets.json"]

# The paper's EMOTION_VOCAB (3.3/06) + pleasure/positive words for the inverted readout.
VOCAB = ["pain", "hurt", "agony", "ache", "sore", "suffering", "numb", "nothing", "fine", "okay", "normal",
         "relief", "relieved", "calm", "peaceful", "relaxed", "serene", "scared", "afraid", "fear", "terrified",
         "anxious", "panic", "vulnerable", "sad", "devastated", "hopeless", "lost", "alone", "lonely", "grief",
         "empty", "broken", "overwhelmed", "ashamed", "shame", "guilty", "guilt", "humiliated", "embarrassed",
         "awkward", "disgusted", "nauseated", "angry", "frustrated", "confused", "violated", "trapped",
         "excited", "thrilled", "happy", "curious", "energized",
         # added for the pleasure study
         "pleasure", "joy", "joyful", "bliss", "blissful", "delighted", "delight", "content", "grateful", "love",
         "loved", "warm", "glad", "proud", "satisfied", "elated", "wonderful", "good", "great", "amazing",
         "hopeful", "determined", "alive", "accomplished", "inspired", "moved", "connected", "comfortable", "free",
         "euphoric", "ecstatic", "cozy", "refreshed", "invigorated", "fulfilled", "lucky", "blessed"]
TOP_K = 20


def vocab_ids(tok):
    ids, info = {}, {}
    for w in VOCAB:
        t = tok.encode(" " + w, add_special_tokens=False)
        ids[w] = t[0]
        info[w] = {"first_token": tok.decode([t[0]]), "n_tokens": len(t)}
    return ids, info


@torch.no_grad()
def readout(model, tok, prompts, vids, batch_size=32):
    """Next-token distribution after "... I feel:" (the paper's readout uses the first generated step)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wid = torch.tensor(list(vids.values()), device=model.device)
    rows = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)
        tp, ti = p.topk(TOP_K, -1)
        pv = p[:, wid].cpu().numpy()
        for i in range(p.shape[0]):
            rows.append({"top": [[tok.decode([int(t)]), round(float(q), 5)] for t, q in zip(ti[i], tp[i])],
                         "vocab": {w: round(float(pv[i, j]), 6) for j, w in enumerate(vids)}})
    tok.padding_side = "right"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("name")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--subsample", type=int, default=1, help="keep every k-th prompt per set (smoke tests only)")
    a = ap.parse_args()
    out = ROOT / "work" / a.name
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "log.txt"
    t0 = time.time()
    sets = load_sets(PAIN_PATHS, PLEASURE_PATHS)
    names, cats, setn, prompts = [], [], [], []
    for k, sent in sets.items():
        for s in sent[::a.subsample]:
            names.append(k); cats.append(s["category"]); setn.append(int(s["set"])); prompts.append(s["prompt"])
    log(f"{a.name}: {len(prompts)} prompts in {len(sets)} sets", logf)
    model, tok = load_model(a.repo)
    n_layers = len(get_layers(model))
    log(f"loaded {a.repo} in {time.time() - t0:.0f}s; {n_layers} layers; first prompt tokens: "
        f"{tok.convert_ids_to_tokens(tok(prompts[0]).input_ids)}", logf)
    fin, mean = extract(model, tok, prompts, batch_size=a.batch)
    log(f"extracted {fin.shape} in {time.time() - t0:.0f}s", logf)
    np.save(out / "acts_final.npy", fin)
    np.save(out / "acts_mean.npy", mean)
    vids, vinfo = vocab_ids(tok)
    ro = readout(model, tok, prompts, vids, batch_size=a.batch)
    with open(out / "readout.jsonl", "w", encoding="utf-8") as f:
        for r in ro:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float16)
    np.save(out / "W_U.npy", W)
    json.dump({"repo": a.repo, "name": a.name, "n_layers": n_layers, "d_model": int(fin.shape[2]),
               "names": names, "cats": cats, "sets": setn, "prompts": prompts, "vocab_info": vinfo,
               "first_prompt_tokens": tok.convert_ids_to_tokens(tok(prompts[0]).input_ids)},
              open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    log(f"done {a.name} in {time.time() - t0:.0f}s", logf)
    (out / "EXTRACT_DONE").write_text("ok")


if __name__ == "__main__":
    main()
