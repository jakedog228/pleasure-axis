# One-time setup on a fresh runtime: paper repo, HF login from Colab Secrets (value never printed), the paper's
# Qwen-7B LoRA adapter, GPU facts. Safe to re-run.
import os, subprocess, json, glob, tarfile
from pathlib import Path

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

if not Path('/content/Pain-axis').exists():
    print(sh('cd /content && git clone -q --depth 1 https://github.com/valen-research/Pain-axis.git && echo cloned'))
tok = None
try:
    from google.colab import userdata
    tok = userdata.get('HF_TOKEN')
except Exception as e:
    print('HF_TOKEN not available:', type(e).__name__)
if tok:
    from huggingface_hub import login
    login(token=tok, add_to_git_credential=False)
    print('HF login ok | token stored for subprocesses')
    for repo in ['google/gemma-2-2b', 'meta-llama/Llama-3.1-8B-Instruct', 'google/gemma-2-9b-it']:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo, 'config.json')
            print('  access ok:', repo)
        except Exception as e:
            print('  NO ACCESS:', repo, type(e).__name__)
ad = Path('/content/adapters/Qwen_2.5_7B_instruct')
if not list(ad.rglob('adapter_config.json')):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('Valen92/pain-adapters', 'adapter_Qwen_2.5_7B_instruct.tar.gz')
    ad.mkdir(parents=True, exist_ok=True)
    with tarfile.open(p) as t:
        t.extractall(ad, filter='data')
print('adapter files:', [str(x.relative_to(ad)) for x in ad.rglob('*') if x.is_file()][:12])
print(sh('pip uninstall -y -q torchao 2>&1 | tail -1; echo torchao removed (peft incompatibility)'))
print(sh('nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; nproc; free -g | head -2; df -h /content | tail -1'))
print(sh('pip list 2>/dev/null | grep -iE "^(torch|transformers|peft|accelerate) "'))
