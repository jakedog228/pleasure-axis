set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python phi4_bos_check.py
