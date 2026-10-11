"""Bundle code/ and datasets/*.json (not build/) into work/bundle.tgz, push it to Colab, extract to /content/pleasure.

usage: python tools/deploy.py            (needs the bridge; see COLAB.md)
"""
import hashlib, io, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(r"D:\Projects\pleasure")
OUT = ROOT / "work" / "bundle.tgz"
files = sorted(ROOT.glob("code/*.py")) + sorted(ROOT.glob("datasets/*.json"))
with tarfile.open(OUT, "w:gz") as t:
    for f in files:
        data = f.read_bytes().replace(b"\r\n", b"\n")
        ti = tarfile.TarInfo(f.relative_to(ROOT).as_posix())
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
print("bundle:", [f.relative_to(ROOT).as_posix() for f in files], OUT.stat().st_size, "bytes")
call = [sys.executable, str(ROOT / "tools" / "colab_call.py")]
subprocess.run(call + ["push", str(OUT), "/content/bundle.tgz"], check=True)
code = ("import tarfile,hashlib,os;os.makedirs('/content/pleasure',exist_ok=True);"
        "t=tarfile.open('/content/bundle.tgz');t.extractall('/content/pleasure');"
        "print([(m.name,hashlib.md5(open('/content/pleasure/'+m.name,'rb').read()).hexdigest()[:8]) for m in t.getmembers()])")
subprocess.run(call + ["run", code, "--timeout", "120"], check=True)
print("local md5:", [(f.relative_to(ROOT).as_posix(), hashlib.md5(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:8]) for f in files])
