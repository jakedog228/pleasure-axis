set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python matched_vectors.py Qwen_2.5_7B_instruct && python syco.py Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct --layer 16 --coeffs 0.5,1,1.5,2
