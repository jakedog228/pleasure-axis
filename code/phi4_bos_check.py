"""Does BOS handling explain the Phi-4 replication gap? Recompute the paper's S2 pain vector for Phi-4 with and without
a prepended BOS token (TransformerLens prepends one by default; the HF Phi-4 tokenizer does not) and compare each with
the paper's released vector at its layer.  usage: python phi4_bos_check.py
"""
import json, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, extract, diff_vector, POS_CATS, cosine

ref = torch.load("/content/Pain-axis/results/3.2_pain_vectors/pain_vectors/Phi_4/pain_vectors.pt", map_location="cpu", weights_only=False)
L = int(ref["layer"])
theirs = ref["s2_pain_vector"].float().numpy().astype(np.float64)
d = json.load(open("/content/Pain-axis/datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]["S2_1P"]["sentences"]
prompts = [s["prompt"] for s in d]
cats = [s["category"] for s in d]
model, tok = load_model("microsoft/phi-4")
print("bos token:", repr(tok.bos_token), "| HF adds BOS:", tok("x").input_ids[:2], flush=True)
out = {"their_layer": L}
for tag, ps in (("no_bos", prompts), ("bos", [tok.bos_token + p for p in prompts])):
    fin, _ = extract(model, tok, ps, batch_size=32, layers=[L])
    v = diff_vector(fin[:, 0, :].astype(np.float64), cats, POS_CATS["pain"])
    out[tag] = {"cos_with_paper": cosine(v, theirs), "norm_ratio": float(np.linalg.norm(v) / np.linalg.norm(theirs))}
    print(tag, out[tag], flush=True)
json.dump(out, open("/content/work/phi4_bos_check.json", "w"), indent=1)
