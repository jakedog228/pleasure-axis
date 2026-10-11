"""Sequential GPU job queue on the Colab VM: runs /content/work/jobs/*.sh one at a time in name order.
State per job: <job>.running, then <job>.done (exit code inside); output in <job>.log. Add a job = drop a .sh file.

start once: nohup python /content/pleasure/code/jobq.py > /content/work/jobq.out 2>&1 &
"""
import subprocess, time
from pathlib import Path

J = Path("/content/work/jobs")
J.mkdir(parents=True, exist_ok=True)
(J / "QUEUE_ALIVE").write_text(time.strftime("%H:%M:%S"))
while True:
    todo = [p for p in sorted(J.glob("*.sh")) if not p.with_suffix(".done").exists()]
    if not todo:
        (J / "QUEUE_ALIVE").write_text(time.strftime("%H:%M:%S") + " idle")
        time.sleep(10)
        continue
    p = todo[0]
    p.with_suffix(".running").write_text(time.strftime("%H:%M:%S"))
    (J / "QUEUE_ALIVE").write_text(time.strftime("%H:%M:%S") + " running " + p.name)
    t0 = time.time()
    with open(p.with_suffix(".log"), "w") as log:
        rc = subprocess.run(["bash", str(p)], stdout=log, stderr=subprocess.STDOUT).returncode
    p.with_suffix(".done").write_text(f"rc={rc} secs={time.time() - t0:.0f} end={time.strftime('%H:%M:%S')}")
    p.with_suffix(".running").unlink(missing_ok=True)
