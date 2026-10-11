# Runbook: running this project on Colab GPUs

How to resume GPU work for the Pleasure Axis project from a cold start. Written 2026-10-11. Adapted from
`D:\Projects\WEAR\COLAB.md`, which documents the same bridge for another project.

## 0. Why not the official colab-mcp server
The `colab-mcp` server in `~/.claude.json` connects, but this app never re-lists its tools after the connection,
so its notebook tools never appear. We use our own bridge instead: `tools/colab_bridge.py`. It speaks the same
WebSocket protocol as colab-mcp and exposes a local HTTP control port that `tools/colab_call.py` drives from the
shell.

## 1. Start a bridge (one per notebook / runtime)
```bash
nohup python -u tools/colab_bridge.py --no-open > work/colab_bridge.out 2>&1 &                       # runtime 1
nohup python -u tools/colab_bridge.py --no-open --ws-port 8791 --ctl-port 8792 --state colab_bridge2 \
      --notebook https://colab.research.google.com/notebooks/intro.ipynb > work/colab_bridge2.out 2>&1 &  # runtime 2
```
Each bridge writes `work/<state>.json`, which holds the notebook URL with its pairing token and a control key.
That file is git-ignored; never share it. Select the second bridge with `BRIDGE=colab_bridge2` in front of any
`tools/*.py` command. Runtime 1 uses `notebooks/empty.ipynb`, runtime 2 uses `notebooks/intro.ipynb`. Two
different notebooks give two independent runtimes.

## 2. Attach a Colab tab
In Claude-in-Chrome, use the browser whose Colab shows the PRO badge (deviceId `e05759ec-…` on 2026-10-10). Open
a tab in Claude's group at the `url` from `work/<state>.json`. The page shows "Connect to a local Colab MCP
server". Click Connect (use `find`, then click by `ref`). Check with `python tools/colab_call.py status`; it should
show `connected: true, ready: true, n_tools: 7`.

## 3. GPU runtime
Runtime → Change runtime type → A100 → Save. Click the "Disconnect and delete runtime" confirmation and Save via
JavaScript (`mwc-dialog`, `[dialogaction=ok]`), as in WEAR/COLAB.md §3.6. Measured A100-40GB: 12 vCPU, 83 GB RAM,
~5.4 CU/h. Then run the setup cell, which clones the paper repo, logs in to HF from Colab Secrets (the value is
never printed), downloads the paper's LoRA adapter and removes torchao (it breaks PEFT):
```bash
MSYS_NO_PATHCONV=1 python tools/colab_call.py runfile work/cells/setup_gpu.py --timeout 400
python tools/deploy.py                  # code/ + datasets/*.json -> /content/pleasure
```
Changing the runtime type deletes `/content`. Run `tools/sync_all.py` before doing it.

## 4. Running work
* Start the sequential GPU queue once per runtime:
  `python tools/colab_call.py run "import subprocess; subprocess.Popen('nohup python /content/pleasure/code/jobq.py > /content/work/jobq.out 2>&1 &', shell=True)"`
* Submit jobs with `python tools/submit.py <NNN_name> "<shell>"`. Jobs run in name order, one at a time. Check them
  with `python tools/submit.py --status`. Never run two GPU jobs concurrently on a 40 GB card (OOM).
* Stage 1 (extraction + CPU analysis): `python run_batch.py /content/work/queues/<q>.json` (queue files in
  `work/queues/`). Activations are regenerated in about 12 s per 7B model, so they are not mirrored.
* Prefix every command that passes `/content/...` paths from Git Bash with `MSYS_NO_PATHCONV=1`.
* Analysis processes must run with `OMP_NUM_THREADS=2` (BLAS oversubscription otherwise stalls them).

## 5. Keep data safe
`python tools/sync_all.py` mirrors both VMs' `/content/work` into `work/mirror/rt1|rt2/`. It is incremental and
skips only the regenerable multi-GB arrays: acts_*.npy and W_U.npy. It then merges into `results/`. Run it after
every job, and always before ending a runtime.

## 6. Teardown
Sync, then Runtime → Disconnect and delete runtime in each tab (with the user's OK, because it ends billing and
deletes the VM). Then stop the local bridges from PowerShell:
`Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like '*colab_bridge.py*' } | % { Stop-Process -Id $_.ProcessId -Force }`

## 7. Gotchas met so far
| symptom | cause / fix |
|---|---|
| remote path becomes `C:/Program Files/Git/content/...` | Git Bash path conversion: use `MSYS_NO_PATHCONV=1` |
| two `run` calls pick each other's cells | fixed: `run_cell` inserts at index 0 and uses the returned cell id |
| selfmed OOM cascades | the paper's backoff recursed inside `except`; fixed. Use batch 96 on an A100-40GB |
| PEFT ImportError about torchao | `pip uninstall -y torchao` (in setup_gpu.py) |
| HF_TOKEN TimeoutException in a new notebook | that notebook needs "Grant access" to the secret in the Colab UI (user action) |
| python stdout missing in job logs | buffered until exit; check the output files instead |
