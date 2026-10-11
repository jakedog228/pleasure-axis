"""Detached driver on the Colab VM: for each model, prefetch weights, extract (GPU, subprocess), analyze (CPU,
background subprocess, overlaps the next model's extraction), delete weights.

usage (from a cell): subprocess.Popen("nohup python /content/pleasure/code/run_batch.py <queue.json> > /content/work/batch.out 2>&1 &", shell=True)
queue.json: [{"repo": "...", "name": "...", "their": "<their result folder name or null>", "batch": 32}, ...]
Status: /content/work/batch_status.json; per-model logs in /content/work/<name>/.
"""
import json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
WORK = Path("/content/work")
THEIRS = Path("/content/Pain-axis/results/3.2_pain_vectors/pain_vectors")
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def status(**kw):
    p = WORK / "batch_status.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(kw, t=time.strftime("%H:%M:%S"))
    p.write_text(json.dumps(s, indent=1))


def prefetch(repo):
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
    except Exception as e:  # noqa
        print("prefetch failed", repo, e, flush=True)


def main():
    queue = json.load(open(sys.argv[1]))
    WORK.mkdir(exist_ok=True)
    analyses = []
    pf = None
    for i, m in enumerate(queue):
        name = m["name"]
        if (WORK / name / "analysis" / "ANALYSIS_DONE").exists():
            print("skip (done)", name, flush=True)
            continue
        if pf is not None:
            pf.join()
        else:
            prefetch(m["repo"])
        nxt = next((q for q in queue[i + 1:] if not (WORK / q["name"] / "analysis" / "ANALYSIS_DONE").exists()), None)
        pf = threading.Thread(target=prefetch, args=(nxt["repo"],)) if nxt else None
        if pf:
            pf.start()
        status(current=name, stage="extract", index=i, n=len(queue))
        if not (WORK / name / "EXTRACT_DONE").exists():
            r = subprocess.run([sys.executable, str(CODE / "extract.py"), m["repo"], name, "--batch", str(m.get("batch", 32))])
            if r.returncode != 0:
                status(**{f"failed_{name}": "extract"})
                continue
        their = str(THEIRS / m["their"]) if m.get("their") and (THEIRS / m["their"]).exists() else ""
        cmd = (f"{sys.executable} {CODE / 'analyze.py'} {name}" + (f" --their-vectors {their}" if their else "") +
               f" && {sys.executable} {CODE / 'geometry2.py'} {name}")
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        analyses.append((name, subprocess.Popen(cmd, shell=True, env=env, stdout=open(WORK / name / "analyze.out", "w"),
                                                stderr=subprocess.STDOUT)))
        if m.get("delete_weights", True):
            d = HF_HUB / ("models--" + m["repo"].replace("/", "--"))
            shutil.rmtree(d, ignore_errors=True)
    status(stage="waiting for analyses")
    for name, p in analyses:
        p.wait()
        status(**{f"analysis_{name}": p.returncode})
    status(stage="ALL DONE")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
