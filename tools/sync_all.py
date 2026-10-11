"""Mirror both Colab VMs' /content/work (minus regenerable multi-GB arrays) into work/mirror/<runtime>/, then merge
into results/ so analyses read one tree.

usage: python tools/sync_all.py [--only rt1|rt2] [--no-merge]
Skipped (regenerable in seconds by code/extract.py): acts_final.npy, acts_mean.npy, W_U.npy.
Merge rule into results/<model>/: analysis from rt1 when present (the copy all early analyses used), else rt2;
steer*/screen/syco from whichever runtime has them (rt2 preferred; rt1's Qwen-7B-I ladder is kept under
results/<model>/steer_rt1 for the cross-VM determinism check); selfmed/feel probes -> results/selfmed/raw/.
"""
import os, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(r"D:\Projects\pleasure")
CALL = [sys.executable, str(ROOT / "tools" / "colab_call.py")]
RTS = {"rt1": "colab_bridge", "rt2": "colab_bridge2"}
SKIP = ("acts_final.npy", "acts_mean.npy", "W_U.npy")


def sync(rt, bridge):
    env = dict(os.environ, BRIDGE=bridge, MSYS_NO_PATHCONV="1")
    full = "--full" in sys.argv          # incremental by default: only files modified since the last sync
    code = ("import tarfile,os,time\n"
            f"stamp='/content/_mirror_stamp'; since=0 if {full!r} or not os.path.exists(stamp) else float(open(stamp).read())\n"
            "now=time.time(); t=tarfile.open('/content/_mirror.tgz','w:gz'); n=0\n"
            "for root,ds,fs in os.walk('/content/work'):\n"
            "    for f in fs:\n"
            f"        if f in {SKIP!r} or f.endswith('.tgz'): continue\n"
            "        p=os.path.join(root,f)\n"
            "        if os.path.getmtime(p) < since: continue\n"
            "        t.add(p, arcname=os.path.relpath(p,'/content/work')); n+=1\n"
            "t.close(); open(stamp,'w').write(str(now - 5)); print(n, os.path.getsize('/content/_mirror.tgz'))")
    r = subprocess.run(CALL + ["run", code, "--timeout", "600"], capture_output=True, text=True, encoding="utf-8", env=env)
    print(rt, "files, tgz bytes:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stdout + r.stderr, flush=True)
    dst = ROOT / "work" / "mirror" / rt
    dst.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "work" / "mirror" / f"{rt}.tgz"
    subprocess.run(CALL + ["pull", "/content/_mirror.tgz", str(tmp)], check=True, env=env)
    with tarfile.open(tmp) as t:
        t.extractall(dst, filter="data")
    print(rt, "mirrored into", dst, flush=True)


def merge():
    M = ROOT / "work" / "mirror"
    R = ROOT / "results"
    models = set()
    for rt in RTS:
        if (M / rt).exists():
            models |= {p.name for p in (M / rt).iterdir() if p.is_dir() and (p / "analysis").exists()}
    for m in sorted(models):
        for sub in ("analysis",):                 # union of both VMs; rt1 copied last so it wins on shared files
            for rt in ("rt2", "rt1"):
                if (M / rt / m / sub).exists():
                    shutil.copytree(M / rt / m / sub, R / m / sub, dirs_exist_ok=True)
        for rt in ("rt1", "rt2"):
            base = M / rt / m
            if not base.exists():
                continue
            for d in base.iterdir():
                if d.is_dir() and (d.name.startswith("steer") or d.name in ("screen", "syco", "demo")):
                    name = d.name + ("_rt1" if rt == "rt1" and d.name == "steer" and (M / "rt2" / m / "steer").exists() else "")
                    shutil.copytree(d, R / m / name, dirs_exist_ok=True)
            for f in ("log.txt", "analyze.out", "meta.json", "readout.jsonl"):
                if (base / f).exists():
                    (R / m / "vm").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(base / f, R / m / "vm" / f)
    for rt in RTS:
        for sub in ("selfmed",):
            if (M / rt / sub).exists():
                shutil.copytree(M / rt / sub, R / "selfmed" / "raw", dirs_exist_ok=True)
        if (M / rt / "jobs").exists():
            shutil.copytree(M / rt / "jobs", ROOT / "work" / "remote_jobs" / rt, dirs_exist_ok=True)
    print("merged", len(models), "models into results/")


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for rt, b in RTS.items():
        if only and rt != only:
            continue
        sync(rt, b)
    if "--no-merge" not in sys.argv:
        merge()
