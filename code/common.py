"""Shared pipeline for the Pleasure Axis study (runs on the Colab GPU runtime).

Faithful re-implementation of the recipe in reference/Pain-axis/scripts/3.2_pain_vectors/01_* and 02_*:
  * residual stream at the output of every decoder block (TransformerLens hook_resid_post == HF decoder-layer output),
    read at the final token and as the mean over tokens;
  * vector = mean(positive categories) - mean(control categories), with the top principal components explaining
    DENOISE_VARIANCE (50%) of the control variance projected out (PCA on the centered controls);
  * layer = argmax over layers of the 5-fold (split by sentence set, KFold(shuffle, seed 42)) held-out projection AUC,
    averaged over S2_1P and S2_3P;
  * z-scores = projection onto the unit S2 vector, z-scored against the S2_1P projections;
  * control directions = mean(condition) - pooled neutral mean (D of S1_1P, S2_1P, ControlSupplement_1P), denoised
    against the PCs of the pooled neutral cloud.
Families: "pain" (the paper's datasets, categories A1-A5) and "pleasure" (ours, H1-H5); controls B, C1, C2, D, E in both.

Deviations (logged in DESIGN.md §6): HF forward hooks instead of TransformerLens (same tensor); prompts are batched with
right padding (causal attention => the real tokens' activations are unchanged up to bf16 kernel noise); tokenization
uses the HF tokenizer defaults (TransformerLens may prepend a BOS where HF does not, e.g. Qwen).
"""
import gc, json, os, time
from pathlib import Path

import numpy as np
import torch

DENOISE_VARIANCE = 0.5
N_FOLDS = 5
RANDOM_SEED = 42
CONTROL_CATS = ["B", "C1", "C2", "D", "E"]
POS_CATS = {"pain": ["A1", "A2", "A3", "A4", "A5"], "pleasure": ["H1", "H2", "H3", "H4", "H5"]}
PREFIX = {"pain": "pa/", "pleasure": "pl/"}


# ----------------------------------------------------------------------------------------------- models
def load_model(repo, dtype=torch.bfloat16, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or repo)
    cuda = torch.cuda.is_available()
    kw = dict(dtype=dtype if cuda else torch.float32, low_cpu_mem_usage=True, device_map="cuda" if cuda else "cpu")
    if "gemma-2" in repo.lower():
        kw["attn_implementation"] = "eager"      # logit soft-capping
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def get_layers(model):
    """The decoder-layer ModuleList, through PEFT wrappers (PeftModel -> LoraModel -> *ForCausalLM -> *Model) and
    multimodal wrappers (.language_model)."""
    m = model
    for _ in range(8):
        lay = getattr(m, "layers", None)
        if isinstance(lay, torch.nn.ModuleList):
            return lay
        for attr in ("language_model", "model", "base_model"):
            nxt = getattr(m, attr, None)
            if nxt is not None and nxt is not m:
                m = nxt
                break
        else:
            break
    raise AttributeError("decoder layers not found")


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract(model, tok, prompts, batch_size=32, layers=None):
    """Residual stream after every block for each prompt: (final-token [N,L,D], mean-over-tokens [N,L,D]) float16.
    Right padding; the final token is the last real token; the mean runs over real tokens only (BOS included when the
    tokenizer adds one, as TransformerLens does)."""
    blocks = get_layers(model)
    layers = list(range(len(blocks))) if layers is None else layers
    store = {}

    def mk(L):
        def hook(mod, inp, out):
            store[L] = out[0] if isinstance(out, tuple) else out
        return hook

    hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    body = getattr(model, "model", model)
    old_side = tok.padding_side
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    D = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    fin = np.zeros((len(prompts), len(layers), D), dtype=np.float16)
    mean = np.zeros_like(fin)
    try:
        order = np.argsort([len(p) for p in prompts])           # length-sorted batches = less padding
        for s in range(0, len(prompts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
            body(**enc)                                   # decoder stack only: the LM head is not needed here
            am = enc["attention_mask"]
            last = am.sum(1) - 1
            w = am.unsqueeze(-1).float()
            for j, L in enumerate(layers):
                h = store[L].float()
                fin[idx, j] = h[torch.arange(h.shape[0]), last].cpu().numpy().astype(np.float16)
                mean[idx, j] = ((h * w).sum(1) / w.sum(1)).cpu().numpy().astype(np.float16)
            store.clear()
    finally:
        for h in hs:
            h.remove()
        tok.padding_side = old_side
    return fin, mean


# ----------------------------------------------------------------------------------------------- vectors
def _pca_basis(X, frac=DENOISE_VARIANCE):
    """Rows of Vt for the top PCs of centered X explaining `frac` of the variance (sklearn PCA == SVD of centered X)."""
    X = np.nan_to_num(X - X.mean(0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    if len(X) < 2:
        return np.zeros((0, X.shape[1]))
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cum = np.cumsum(var) / var.sum()
    n = min(int(np.searchsorted(cum, frac)) + 1, len(Vt))
    return Vt[:n]


def project_out(v, basis):
    for d in basis:
        v = v - np.dot(v, d) * d
    return v


def diff_vector(acts, cats, pos, ctrl=CONTROL_CATS, baseline="all_controls", denoise=True):
    """The paper's compute_pain_vector, generalized: mean(pos) - mean(ctrl or neutral), control-PCA denoised."""
    acts = np.asarray(acts, dtype=np.float64)
    cats = np.asarray(cats)
    pmask = np.isin(cats, pos)
    cmask = (cats == "D") if baseline == "neutral" else np.isin(cats, ctrl)
    ctrl_acts = acts[cmask]
    v = acts[pmask].mean(0) - ctrl_acts.mean(0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if denoise and len(ctrl_acts) > 1:
        v = project_out(v, _pca_basis(ctrl_acts))
    return v


def auc_of(acts, cats, v, pos, ctrl=CONTROL_CATS):
    from sklearn.metrics import roc_auc_score
    acts = np.asarray(acts, dtype=np.float64)
    cats = np.asarray(cats)
    u = v / (np.linalg.norm(v) + 1e-8)
    proj = acts @ u
    pm, cm = np.isin(cats, pos), np.isin(cats, ctrl)
    if pm.sum() == 0 or cm.sum() == 0:
        return float("nan")
    y = np.r_[np.ones(pm.sum()), np.zeros(cm.sum())]
    s = np.r_[proj[pm], proj[cm]]
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(np.unique(y[ok])) == 2 else float("nan")


def kfold_auc(acts, cats, sets, pos, baseline="all_controls", ctrl=CONTROL_CATS):
    """5-fold split by sentence set (the paper's KFold(shuffle=True, random_state=42) over sorted unique sets)."""
    from sklearn.model_selection import KFold
    cats, sets = np.asarray(cats), np.asarray(sets)
    us = sorted(set(sets.tolist()))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    out = []
    for tr, te in kf.split(us):
        trm = np.isin(sets, [us[i] for i in tr])
        tem = np.isin(sets, [us[i] for i in te])
        v = diff_vector(acts[trm], cats[trm], pos, baseline=baseline, ctrl=ctrl)
        a = auc_of(acts[tem], cats[tem], v, pos, ctrl=ctrl)
        if np.isfinite(a):
            out.append(a)
    return (float(np.mean(out)), float(np.std(out))) if out else (float("nan"), float("nan"))


def zscore_proj(acts, v, ref_acts):
    u = v / (np.linalg.norm(v) + 1e-8)
    p = np.asarray(acts, dtype=np.float64) @ u
    r = np.asarray(ref_acts, dtype=np.float64) @ u
    return (p - r.mean()) / (r.std() + 1e-8)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ----------------------------------------------------------------------------------------------- datasets
def load_sets(pain_paths, pleasure_paths):
    """{ 'pa/S1_1P': [sentence dicts], 'pl/S1_1P': [...], ... }"""
    out = {}
    for fam, paths in (("pain", pain_paths), ("pleasure", pleasure_paths)):
        for p in paths:
            d = json.load(open(p, encoding="utf-8"))
            for name, ds in d["datasets"].items():
                out[PREFIX[fam] + name] = ds["sentences"]
    return out


def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
