"""Detached driver for stage 2 on the Colab VM: per model, the steering ladder (steer.py) then the self-other screen
(screen.py). Needs stage 1 (run_batch.py) analysis outputs.

usage: nohup python run_stage2.py <queue.json> [--only steer|screen] > /content/work/stage2.out 2>&1 &
"""
import json, subprocess, sys, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
WORK = Path("/content/work")


def status(**kw):
    p = WORK / "stage2_status.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(kw, t=time.strftime("%H:%M:%S"))
    p.write_text(json.dumps(s, indent=1))


queue = json.load(open(sys.argv[1]))
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
for m in queue:
    W = WORK / m["name"]
    if not (W / "analysis" / "ANALYSIS_DONE").exists():
        status(**{f"skip_{m['name']}": "no analysis"})
        continue
    if only in (None, "steer") and not (W / "steer" / "STEER_DONE").exists():
        status(current=m["name"], stage="steer")
        r = subprocess.run([sys.executable, str(CODE / "steer.py"), m["repo"], m["name"]] + m.get("steer_args", []))
        status(**{f"steer_{m['name']}": r.returncode})
    if only in (None, "screen") and not (W / "screen" / "SCREEN_DONE").exists():
        status(current=m["name"], stage="screen")
        r = subprocess.run([sys.executable, str(CODE / "screen.py"), m["repo"], m["name"], m["fmt"]])
        status(**{f"screen_{m['name']}": r.returncode})
status(stage="ALL DONE")
print("ALL DONE", flush=True)
