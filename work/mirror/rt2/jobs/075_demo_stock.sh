set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python demo_gen.py Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct --tag stock
