"""Submit a shell job to the VM's sequential GPU queue (code/jobq.py).

usage: python tools/submit.py <job_name> "<shell command(s)>"     (job_name sorts the queue: use 010_, 020_, ...)
       python tools/submit.py --status
"""
import json, subprocess, sys
from pathlib import Path

CALL = [sys.executable, str(Path(__file__).with_name("colab_call.py"))]
if sys.argv[1] == "--status":
    code = ("import glob,os\nfor p in sorted(glob.glob('/content/work/jobs/*.sh')):\n"
            "    s='done '+open(p[:-3]+'.done').read() if os.path.exists(p[:-3]+'.done') else ('RUNNING since '+open(p[:-3]+'.running').read() if os.path.exists(p[:-3]+'.running') else 'queued')\n"
            "    print(os.path.basename(p), '|', s)\n"
            "print(open('/content/work/jobs/QUEUE_ALIVE').read() if os.path.exists('/content/work/jobs/QUEUE_ALIVE') else 'no queue')")
    print(subprocess.run(CALL + ["run", code, "--timeout", "60"], capture_output=True, text=True, encoding="utf-8").stdout)
    sys.exit()
name, cmd = sys.argv[1], sys.argv[2]
body = "set -o pipefail\ncd /content/pleasure/code\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n" + cmd + "\n"
code = (f"import os\nos.makedirs('/content/work/jobs',exist_ok=True)\n"
        f"open('/content/work/jobs/{name}.sh','w').write({body!r})\nprint('queued {name}')")
print(subprocess.run(CALL + ["run", code, "--timeout", "60"], capture_output=True, text=True, encoding="utf-8").stdout.strip())
