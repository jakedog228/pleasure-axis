"""Feel probe (no adapter) for several instruct models: pleasure and pain S2 vectors from our analysis, steering layer at
~57% depth (the paper's Qwen-7B behavioral layer 16/28), 10 random directions of matched norm per vector."""
import json, subprocess, sys
MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct"), ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral_7B_instruct"),
          ("Qwen/Qwen2.5-14B-Instruct", "Qwen_2.5_14B_instruct"), ("microsoft/phi-4", "Phi_4"),
          ("Qwen/Qwen2.5-3B-Instruct", "Qwen_2.5_3B_instruct")]
for repo, name in MODELS:
    W = f"/content/work/{name}/analysis/"
    try:
        s = json.load(open(W + "summary.json"))
    except FileNotFoundError:
        print("skip", name); continue
    L = round(0.57 * s["n_layers"])
    vp = f"pleasure=npz:{W}vectors.npz:Lpleasure|pleasure:s2:{s['pleasure_best_layer_final']}"
    va = f"pain=npz:{W}vectors.npz:Lpain|pain:s2:{s['pain_best_layer_final']}"
    subprocess.run([sys.executable, "feelprobe.py", "--repo", repo, "--adapter", "none", "--layer", str(L), "--vector", vp,
                    "--vector", va, "--doses", "0.5,1,1.5,2,2.5,3,4,6", "--tag", f"{name}_noadapter"])
