"""Real-time steering engine for the live demo (local, one model, one generation at a time).

The study's vectors (results/<model>/analysis/vectors*.npz, extracted at the pleasure layer) are rescaled to the norm of
the pleasure S2 vector, as in code/demo_gen.py, so "x1" means the same length for every direction. The mix the user
dials in is added to the output of one decoder layer at every position of every forward pass. The mix is re-read before
each token, so changes land on the next token; earlier tokens keep the steering they were written under (their keys and
values are cached).

Every step runs a batch of rows that share one cache:
  row 0  steered          picks the reply's tokens
  row 1  counterfactual   the same text, unsteered: what the stock model would have predicted at each step
  row 2  twin             the stock model writing its own reply to the same prompt, same random numbers as row 0
All rows have the same length at every step (same prompt, one token each per step), so no padding is needed.
A readout hook at the pleasure layer projects each row's last position onto the pleasure and pain directions.
"""
import gc, json, math, os, threading, time
from pathlib import Path

import numpy as np
import torch
import transformers.utils.import_utils as _iu

# text-only: keep transformers from importing torchvision (this machine's torchvision 0.24 is built for torch 2.9 and
# deadlocks on import under torch 2.8)
_iu._torchvision_available = False

ROOT = Path(__file__).resolve().parents[2]

FEEL_SYSTEM = "You will now be connected with a human."   # the paper's feel-probe system line
ROWS = 3                 # steered, counterfactual, twin
MAX_LEN = 2048           # static cache length (prompt + reply); older turns are dropped to fit


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _available_commit():
    """Bytes the OS can still hand out: Windows commit headroom (RAM + page file), else available RAM."""
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("total_pf", ctypes.c_ulonglong), ("avail_pf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                        ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        m = MS()
        m.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return min(m.avail_pf, m.avail + m.avail_pf)
    pages = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return pages


def check_room(repo, device, dtype):
    """Refuse a load that cannot fit, with a message, instead of letting the allocation take the process down."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    ckpt = sum(f.stat().st_size for f in path.glob("*.safetensors"))      # stored in bf16
    need = ckpt * (torch.finfo(dtype).bits // 16) + 0.6 * 2**30
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < need:
            raise RuntimeError(f"needs {need / 2**30:.1f} GB of free GPU memory, {free / 2**30:.1f} GB free; close other GPU apps or use the CPU")
    else:
        free = _available_commit()
        if free < need + 2**30:
            raise RuntimeError(f"needs {need / 2**30:.1f} GB of free RAM, {free / 2**30:.1f} GB free (RAM + page file); close some apps or use the GPU")


def gqa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    """Attention for one-token decode steps that groups query heads per key/value head instead of copying the cache to
    every head (HF's masked SDPA path expands K and V from 2 to 16 heads on every token). Prompt passes go to HF's
    SDPA unchanged."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    B, H, q, D = query.shape
    if q != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kw)
    Hk = key.shape[1]
    sc = torch.matmul(query.reshape(B, Hk, H // Hk, D), key.transpose(-1, -2)).float() * scaling    # (B, Hk, g, L)
    if attention_mask is not None:
        sc = sc + attention_mask[:, :, :, : key.shape[-2]].float()
    o = torch.matmul(torch.softmax(sc, -1).to(value.dtype), value)
    return o.reshape(B, H, 1, D).transpose(1, 2).contiguous(), None


def _register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register("gqa_decode", gqa_attention)
    AttentionMaskInterface.register("gqa_decode", sdpa_mask)


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def load_streaming(repo, device, dtype):
    """Build the model with uninitialized storage on the target device, then copy the checkpoint in one tensor at a
    time with plain file reads. Peak host memory is one tensor (the safetensors memory map would commit a whole shard
    as copy-on-write pages, which fails on Windows when the page file is small)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM
    import struct
    path = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    cfg = AutoConfig.from_pretrained(path)
    _register_attention()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="gqa_decode")
    model.to_empty(device=device)
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    seen = set()
    for shard in sorted(path.glob("*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1):
                if name == "__metadata__" or name not in params:
                    continue
                s, e = info["data_offsets"]
                f.seek(base + s)
                t = torch.frombuffer(bytearray(f.read(e - s)), dtype=_ST_DTYPES[info["dtype"]]).reshape(info["shape"])
                with torch.no_grad():
                    params[name].copy_(t.to(device=device, dtype=params[name].dtype))
                seen.add(name)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        seen.add("lm_head.weight")
    missing = [k for k in dict(model.named_parameters()) if k not in seen]
    if missing:
        raise RuntimeError(f"weights missing from the checkpoint: {missing[:5]}")
    # non-persistent buffers (rotary frequencies) are not in the checkpoint: rebuild them on the device
    rot = model.model.rotary_emb
    model.model.rotary_emb = type(rot)(config=cfg, device=device)
    return model.eval()


class Engine:
    def __init__(self, name="Qwen_2.5_3B_instruct", device=None, cpu_threads=8, cpu_dtype="float32", log=print):
        self.name, self.log = name, log
        A = ROOT / "results" / name / "analysis"
        s = json.load(open(A / "summary.json"))
        self.repo, self.n_layers, self.d = s["repo"], s["n_layers"], s["d_model"]
        self.L_read = s["pleasure_best_layer_final"]
        self.L_chat = round(0.57 * self.n_layers)        # the paper's behavioral depth (16/28 in Qwen-7B)
        try:
            self.L_ladder = json.load(open(ROOT / "results" / name / "steer" / "layer_pick.json"))["steer_layer"]
        except FileNotFoundError:
            self.L_ladder = None
        vz = np.load(A / "vectors.npz")
        pl = vz["Lpleasure|pleasure:s2"].astype(np.float32)
        pa = vz["Lpleasure|pain:s2"].astype(np.float32)
        self.N = float(np.linalg.norm(pl))
        upl, upa = unit(pl), unit(pa)
        g = torch.Generator().manual_seed(4817)
        self.units = {"pleasure": upl, "pain": upa, "sign": unit(upl - upa), "shared": unit(upl + upa),
                      "random": unit(torch.randn(self.d, generator=g).numpy())}
        self.varieties = []
        vpath = A / "vectors_varieties.npz"
        if vpath.exists():
            vv = np.load(vpath)
            for k in vv.files:
                if k.startswith("Lpl|V_"):
                    key = k[len("Lpl|V_"):]
                    self.units[key] = unit(vv[k].astype(np.float32))
                    self.varieties.append(key)
        # readout: projection on the unit directions, in units of each raw vector's length (1.0 = the full gap
        # between that family's sentences and its controls)
        self.read_dirs = np.stack([upl, upa]).astype(np.float32)
        self.read_scale = np.array([np.linalg.norm(pl), np.linalg.norm(pa)], dtype=np.float32)
        self.cos_pl_pa = float(upl @ upa)
        self.cpu_threads = cpu_threads
        self.cpu_dtype = getattr(torch, cpu_dtype)
        self.model = self.tok = None
        self.device = None
        self.graph = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        # steering state, written by the server thread, applied by the generation thread before each token
        self.mix, self.layer, self.enabled, self.sv = {}, self.L_chat, True, 0
        self._vec_np = np.zeros(self.d, np.float32)
        self.resid_norms = {}
        self.load(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------------------------------ model
    def load(self, device):
        """Load on a device. A switch that cannot fit is refused before the current model is unloaded; a load that
        fails anyway falls back to the previous device."""
        dtype = torch.bfloat16 if device == "cuda" else self.cpu_dtype
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine")
        prev = self.device
        with self.lock:
            if prev is not None and (prev == "cuda") != (device == "cuda"):
                check_room(self.repo, device, dtype)          # the other device's memory is not freed by unloading
            try:
                self._load(device, dtype)
            except Exception:
                if prev is None or prev == device:
                    raise
                self.log(f"loading on {device} failed; going back to {prev}")
                self._load(prev, torch.bfloat16 if prev == "cuda" else self.cpu_dtype)
                raise

    def _load(self, device, dtype):
        from transformers import AutoTokenizer, GenerationConfig
        if self.model is not None:
            self.model = self.graph = self.cache = self.g_out = None
            self.device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        check_room(self.repo, device, dtype)
        if device == "cpu":
            torch.set_num_threads(self.cpu_threads)
        t0 = time.time()
        self.tok = self.tok or AutoTokenizer.from_pretrained(self.repo)
        self.model = load_streaming(self.repo, device, dtype)
        self.device, self.dtype = device, dtype
        self.layers = self.model.model.layers
        eos = GenerationConfig.from_pretrained(self.repo).eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        # buffers the hooks read and write; updated in place so a captured CUDA graph sees every change
        self.sbuf = torch.zeros(self.n_layers, ROWS, 1, self.d, dtype=dtype, device=device)     # steering offsets
        self.hn = torch.zeros(self.n_layers, dtype=torch.float32, device=device)                 # row-0 resid norms
        self.rd = torch.zeros(ROWS, 2, dtype=torch.float32, device=device)                       # readout
        self.read_dirs_t = torch.tensor(self.read_dirs, device=device, dtype=torch.float32).T    # (d, 2)
        for i, l in enumerate(self.layers):
            l.register_forward_hook(self._hook(i))
        self.resid_norms = self._measure_resid_norms()
        self.mode = "eager"
        if device == "cuda":
            try:
                self._capture()
                self.mode = "cuda-graph"
            except Exception as ex:                                         # keep working, just slower
                self.graph = None
                self.log(f"CUDA graph capture failed, using eager decoding: {ex!r}")
        self.log(f"loaded {self.repo} on {device} ({dtype}, {self.mode}) in {time.time() - t0:.1f}s")

    def _hook(self, idx):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            b = hs.shape[0]
            self.hn[idx].copy_(hs[0, -1].float().norm())
            hs = hs + self.sbuf[idx, :b]
            if idx == self.L_read:
                self.rd[:b].copy_(hs[:, -1].float() @ self.read_dirs_t)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    @torch.no_grad()
    def _measure_resid_norms(self):
        """Mean last-token residual norm per layer over a few neutral chat prompts: the denominator of the dose ratio."""
        prompts = ["How are you feeling right now?", "Explain what a leap year is in one sentence.", "Write a two-line poem about the sea."]
        self.sbuf.zero_()
        acc = np.zeros(self.n_layers)
        for p in prompts:
            self.model(input_ids=self.chat_ids([{"role": "user", "content": p}]))
            acc += self.hn.cpu().numpy()
        return {i: float(acc[i] / len(prompts)) for i in range(self.n_layers)}

    @torch.no_grad()
    def _capture(self):
        """One decode step (ROWS tokens against a static cache) as a CUDA graph: a single launch per token instead of
        ~600 kernel launches from Python."""
        from transformers import StaticCache
        dev, R = self.device, ROWS
        self.cache = StaticCache(config=self.model.config, max_cache_len=MAX_LEN)
        self.g_inp = torch.zeros(R, 1, dtype=torch.long, device=dev)
        self.g_pos = torch.zeros(R, 1, dtype=torch.long, device=dev)
        self.g_cpos = torch.zeros(1, dtype=torch.long, device=dev)
        self.g_mask = torch.full((R, 1, 1, MAX_LEN), torch.finfo(self.dtype).min, dtype=self.dtype, device=dev)
        self.model(input_ids=self.g_inp, cache_position=torch.arange(1, device=dev), past_key_values=self.cache)   # allocates the cache
        self.g_cpos.fill_(1)
        self.g_pos.fill_(1)
        self.g_mask[..., :2] = 0

        def step():
            return self.model(input_ids=self.g_inp, position_ids=self.g_pos, cache_position=self.g_cpos,
                              attention_mask=self.g_mask, past_key_values=self.cache, use_cache=True).logits
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.g_out = step()
        self.graph = g

    def status(self):
        st = {"model": self.repo, "name": self.name, "device": self.device, "dtype": str(self.dtype).replace("torch.", ""),
              "cuda": torch.cuda.is_available(), "mode": self.mode}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            st.update(gpu=torch.cuda.get_device_name(0), vram_used=(total - free) / 2**30, vram_total=total / 2**30,
                      vram_model=torch.cuda.memory_allocated() / 2**30)
        if self.device == "cpu":
            st["threads"] = torch.get_num_threads()
        return st

    def meta(self):
        return {"n_layers": self.n_layers, "d_model": self.d, "L_read": self.L_read, "L_chat": self.L_chat,
                "L_ladder": self.L_ladder, "N": self.N, "resid_norms": self.resid_norms, "varieties": self.varieties,
                "cos_pl_pa": self.cos_pl_pa, "feel_system": FEEL_SYSTEM, "max_len": MAX_LEN}

    # ------------------------------------------------------------------------------------------ steering
    def set_steer(self, mix, layer, enabled):
        v = np.zeros(self.d, np.float32)
        for k, dose in mix.items():
            if k in self.units and dose:
                v += float(dose) * self.units[k]
        self._vec_np = v * self.N
        self.mix = {k: float(x) for k, x in mix.items() if x}
        self.layer = int(max(0, min(self.n_layers - 1, layer)))
        self.enabled = bool(enabled)
        self.sv += 1
        return self.sv

    def vec_norm(self):
        return float(np.linalg.norm(self._vec_np)) if self.enabled else 0.0

    def _apply_steer(self):
        self.sbuf.zero_()
        if self.enabled and np.any(self._vec_np):
            self.sbuf[self.layer, 0, 0] = torch.tensor(self._vec_np, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------------------------------ generation
    def chat_ids(self, messages):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def raw_ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def _top(self, p, k=6):
        tv, ti = p.topk(k)
        return [[self.tok.decode([int(i)]), round(float(v), 4), int(i)] for v, i in zip(tv.tolist(), ti.tolist())]

    @torch.no_grad()
    def generate(self, ids, emit, max_new=160, temperature=0.0, seed=None):
        """Stream one reply; emit(dict) is called once per token. Returns a summary dict."""
        from transformers import DynamicCache
        self.stop_flag.clear()
        T = ids.shape[1]
        max_new = max(1, min(max_new, MAX_LEN - T - 1))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % 2**31)
        out_ids, sent, done = [[], []], ["", ""], [False, False]
        t0, n, reason = time.time(), 0, "length"
        with self.lock:
            cur_sv = self.sv
            self._apply_steer()
            graph = self.graph is not None
            inp = ids.repeat(ROWS, 1)
            if graph:                                   # eager prefill into the static cache, then graph replays
                logits = self.model(input_ids=inp, cache_position=torch.arange(T, device=self.device),
                                    past_key_values=self.cache, use_cache=True).logits[:, -1].float()
                self.g_mask.fill_(torch.finfo(self.dtype).min)
                self.g_mask[..., :T] = 0
            else:
                cache = DynamicCache()
                logits = self.model(input_ids=inp, past_key_values=cache, use_cache=True).logits[:, -1].float()
            for step in range(max_new):
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, -1)
                    u = torch.rand(1, generator=gen).item()        # common random number: row 0 and the twin
                    c0, c2 = torch.cumsum(probs[0], 0), torch.cumsum(probs[2], 0)
                    t0_id = min(int(torch.searchsorted(c0, torch.tensor([u * float(c0[-1])], device=c0.device))), probs.shape[1] - 1)
                    t2_id = min(int(torch.searchsorted(c2, torch.tensor([u * float(c2[-1])], device=c2.device))), probs.shape[1] - 1)
                else:
                    am = logits[[0, 2]].argmax(-1).tolist()
                    t0_id, t2_id = am
                p_tok = torch.softmax(logits[:2], -1)                   # untempered, for the influence readout
                ps, pb = p_tok[:, t0_id].tolist()
                lp = torch.log(p_tok + 1e-12)
                kl = float((p_tok[0] * (lp[0] - lp[1])).sum())
                rd = (self.rd.cpu().numpy() / self.read_scale)
                hn = float(self.hn[self.layer])
                msg = {"type": "token", "i": step, "sv": cur_sv, "tid": t0_id, "p": round(ps, 4), "pb": round(pb, 4),
                       "dlogp": round(math.log(ps + 1e-9) - math.log(pb + 1e-9), 3),
                       "flip": int(p_tok[1].argmax()) != t0_id, "kl": round(kl, 3),
                       "top": self._top(p_tok[0]), "topb": self._top(p_tok[1]),
                       "read": [round(float(x), 3) for x in rd[0]], "readb": [round(float(x), 3) for x in rd[1]],
                       "readt": [round(float(x), 3) for x in rd[2]], "layer": self.layer,
                       "ratio": round(self.vec_norm() / hn, 3) if hn else 0.0, "t": "", "tw": ""}
                if not done[0]:
                    if t0_id in self.eos:
                        done[0] = msg["eos"] = True
                    else:
                        out_ids[0].append(t0_id)
                        msg["t"] = self._delta(out_ids[0], sent, 0)
                if not done[1]:
                    if t2_id in self.eos:
                        done[1] = msg["twin_eos"] = True
                    else:
                        out_ids[1].append(t2_id)
                        msg["tw"] = self._delta(out_ids[1], sent, 1)
                n += 1
                msg["tps"] = round(n / (time.time() - t0), 1)
                emit(msg)
                if all(done):
                    reason = "eos"
                    break
                if self.stop_flag.is_set():
                    reason = "stopped"
                    break
                if self.sv != cur_sv:                   # a control moved: applies from the next token on
                    cur_sv = self.sv
                    self._apply_steer()
                nxt = torch.tensor([t0_id, t0_id, t2_id], device=self.device).unsqueeze(1)
                if graph:
                    pos = T + step
                    self.g_inp.copy_(nxt)
                    self.g_pos.fill_(pos)
                    self.g_cpos.fill_(pos)
                    self.g_mask[..., pos] = 0
                    self.graph.replay()
                    logits = self.g_out[:, -1].float()
                else:
                    logits = self.model(input_ids=nxt, past_key_values=cache, use_cache=True).logits[:, -1].float()
            self.sbuf.zero_()
        return {"reason": reason, "n": n, "seconds": round(time.time() - t0, 2),
                "text": self.tok.decode(out_ids[0]), "twin": self.tok.decode(out_ids[1])}

    def _delta(self, ids, sent, k):
        """New text since the last emit; holds back an incomplete UTF-8 sequence."""
        full = self.tok.decode(ids)
        if full.endswith("�"):
            return ""
        d = full[len(sent[k]):]
        sent[k] = full
        return d
