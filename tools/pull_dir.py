"""Tar a remote directory on the Colab VM (skipping large arrays) and pull it into a local directory.

usage: python tools/pull_dir.py <remote_dir> <local_dir> [--exclude-glob '*.npy'] [--max-mb 200]
"""
import subprocess, sys, tarfile
from pathlib import Path

CALL = [sys.executable, str(Path(__file__).with_name("colab_call.py"))]
remote, local = sys.argv[1], Path(sys.argv[2])
excl = sys.argv[sys.argv.index("--exclude-glob") + 1] if "--exclude-glob" in sys.argv else "*.npy"
tgz = "/content/_pull.tgz"
code = (f"import tarfile,fnmatch,os\n"
        f"t=tarfile.open({tgz!r},'w:gz')\n"
        f"for root,ds,fs in os.walk({remote!r}):\n"
        f"    for f in fs:\n"
        f"        p=os.path.join(root,f)\n"
        f"        if not fnmatch.fnmatch(f,{excl!r}): t.add(p, arcname=os.path.relpath(p,{remote!r}))\n"
        f"t.close();print(os.path.getsize({tgz!r}))")
r = subprocess.run(CALL + ["run", code, "--timeout", "300"], capture_output=True, text=True, encoding="utf-8")
print("remote tgz bytes:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stdout + r.stderr)
local.mkdir(parents=True, exist_ok=True)
tmp = local / "_pull.tgz"
subprocess.run(CALL + ["pull", tgz, str(tmp)], check=True)
with tarfile.open(tmp) as t:
    t.extractall(local, filter="data")
tmp.unlink()
print("extracted into", local)
