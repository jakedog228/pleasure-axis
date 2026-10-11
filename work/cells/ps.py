import subprocess, os, glob, json
out = subprocess.run(['ps', '-eo', 'pid,nlwp,etime,time,pcpu,args'], capture_output=True, text=True).stdout
for l in out.splitlines():
    if any(k in l for k in ('analyze', 'selfmed', 'extract', 'steer', 'screen', 'feelprobe', 'run_batch', 'run_stage2', 'geometry2', 'PID')):
        print(l[:150])
print(open('/proc/loadavg').read().strip())
for d in sorted(glob.glob('/content/work/*/analysis/log.txt')):
    print(d.split('/')[3], '|', open(d).read().strip().splitlines()[-1][:120])
for f in ['/content/work/batch_status.json', '/content/work/stage2_status.json']:
    if os.path.exists(f):
        print(f, json.load(open(f)))
print(subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader', shell=True, capture_output=True, text=True).stdout)
