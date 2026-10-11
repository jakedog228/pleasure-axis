set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "
import json,subprocess,sys
for f in ['/content/work/queues/open.json','/content/work/queues/extra.json']:
    for m in json.load(open(f)):
        subprocess.run([sys.executable,'steer.py',m['repo'],m['name'],'--layer-from','pain:s2','--match-norm','pain:s2','--outdir','steer_painL'])
"
