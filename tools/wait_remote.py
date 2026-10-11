"""Poll the Colab VM until a file exists (or a timeout), printing a short status line at the end.

usage: python tools/wait_remote.py <remote_path> [--every 60] [--timeout 3600] [--tail <remote_log>]
Exit code 0 when the file appeared, 1 on timeout.
"""
import subprocess, sys, time
from pathlib import Path

CALL = [sys.executable, str(Path(__file__).with_name("colab_call.py"))]
path = sys.argv[1]
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 60
timeout = int(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 3600
tail = sys.argv[sys.argv.index("--tail") + 1] if "--tail" in sys.argv else None
t0 = time.time()
while True:
    code = f"import os;print('EXISTS' if os.path.exists({path!r}) else 'MISSING')"
    r = subprocess.run(CALL + ["run", code, "--timeout", "60"], capture_output=True, text=True, encoding="utf-8")
    if "EXISTS" in r.stdout:
        ok = True
        break
    if time.time() - t0 > timeout:
        ok = False
        break
    time.sleep(every)
if tail:
    r = subprocess.run(CALL + ["run", f"print(open({tail!r}).read()[-1500:])", "--timeout", "60"], capture_output=True,
                       text=True, encoding="utf-8")
    print(r.stdout)
print(("APPEARED: " if ok else "TIMEOUT: ") + path, f"after {time.time() - t0:.0f}s")
sys.exit(0 if ok else 1)
