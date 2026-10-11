"""Submit every 'name :: command' line of a file to a runtime's queue (BRIDGE env var selects the runtime)."""
import subprocess, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if "::" in line:
        name, cmd = (x.strip() for x in line.split("::", 1))
        subprocess.run([sys.executable, str(Path(__file__).with_name("submit.py")), name, cmd])
