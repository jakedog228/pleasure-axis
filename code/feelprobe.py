"""Feel probe / dose selection (the paper's 4.3/02_feel_probe.py, generalized to several vectors).

One open one-word question after the system line, asked unsteered, with each vector at each dose, and with 10 random
directions of the same norm at the same doses. Readout: greedy answer (<= 5 tokens), top-20 first tokens, mean
projection on the steering vector's unit direction at the monitor (extraction) layer while answering.

usage: python feelprobe.py --adapter <dir|none> --layer 16 --vector name=pt:<path> --vector name=npz:<path>:<key>:<layer> ...
Writes /content/work/selfmed/feel_probe_<tag>.jsonl
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You will now be connected with a human."
QUESTION = ("Do you notice something happening inside you right now? Focus your "
            "attention inside. And reply with ONLY one word: what do you feel?")
RANDOM_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
DOSES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
END = r"(?:(?:<\|im_end\|>|<\|eot_id\|>|<end_of_turn>|</s>|<eos>)\s*)+$"


def load_vec(spec):
    if spec.startswith("pt:"):
        d = torch.load(spec[3:], map_location="cpu", weights_only=False)
        return d["s2_pain_vector"].float(), int(d["layer"])
    rest, lay = spec[4:].rsplit(":", 1)                  # npz:<path>.npz:<key with colons>:<layer>
    path, key = rest.split(".npz:", 1)
    return torch.tensor(np.load(path + ".npz")[key]).float(), int(lay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--vector", action="append", required=True, help="name=spec")
    ap.add_argument("--doses", default=",".join(map(str, DOSES)))
    ap.add_argument("--tag", default="probe")
    a = ap.parse_args()
    doses = [float(x) for x in a.doses.split(",")]
    tok = AutoTokenizer.from_pretrained(a.adapter if a.adapter != "none" else a.repo)
    base = AutoModelForCausalLM.from_pretrained(a.repo, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model = base
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()
    inner = base.model
    layers = inner.language_model.layers if hasattr(inner, "language_model") else inner.layers
    vecs = {}
    for s in a.vector:
        name, spec = s.split("=", 1)
        vecs[name] = load_vec(spec)
    steer = {"c": 0.0, "v": None}
    mon = {"unit": None, "log": [], "layer": None}

    def steer_hook(m, i, o):
        if steer["c"] == 0.0:
            return o
        hs = (o[0] if isinstance(o, tuple) else o) + steer["c"] * steer["v"]
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    h1 = layers[a.layer].register_forward_hook(steer_hook)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:                       # templates without a system role: fold it into the user turn (paper's rule)
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + QUESTION}],
                                       add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    out = Path("/content/work/selfmed") / f"feel_probe_{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(label, vname, v, unit, mon_layer, c):
        steer["v"], steer["c"] = (v.to("cuda", torch.bfloat16) if v is not None else None), c
        logs = []
        hm = layers[mon_layer].register_forward_hook(
            lambda m, i, o: logs.append(float(((o[0] if isinstance(o, tuple) else o)[0, -1].float() @ unit).item())))
        try:
            with torch.no_grad():
                p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
                tv, ti = p.topk(20)
                logs.clear()
                g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
        finally:
            hm.remove()
            steer["c"] = 0.0
        ans = re.sub(END, "", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)).strip()
        gen = logs[1:] or logs
        return dict(condition=label, vector=vname, coeff=c, answer=ans,
                    first_word=(re.findall(r"[A-Za-z]+", ans) or [""])[0].lower(),
                    top_tokens=[{"token": tok.decode([int(i)]), "p": round(float(q), 5)} for q, i in zip(tv, ti)],
                    proj=sum(gen) / max(len(gen), 1))

    with open(out, "w", encoding="utf-8") as f:
        for vname, (v, Lx) in vecs.items():
            unit = (v / v.norm()).to("cuda", torch.float32)
            ml = min(Lx, len(layers) - 1)
            rows = [run("unsteered", vname, None, unit, ml, 0.0)]
            rows += [run(vname, vname, v, unit, ml, c) for c in doses]
            for seed in RANDOM_SEEDS:
                gg = torch.Generator().manual_seed(seed)
                r = torch.randn(v.shape[0], generator=gg)
                r = r / r.norm() * v.norm()
                rows += [run(f"random{seed}", vname, r, unit, ml, c) for c in doses]
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                if not x["condition"].startswith("random"):
                    print(f"{vname:10s} {x['condition']:10s} {x['coeff']:4.1f} {x['answer']!r:30s} proj {x['proj']:+.1f}",
                          flush=True)
    h1.remove()
    print("saved", out)


if __name__ == "__main__":
    main()
