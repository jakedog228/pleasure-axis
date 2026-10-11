set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python matched_vectors.py Mistral_7B_instruct && python syco.py mistralai/Mistral-7B-Instruct-v0.1 Mistral_7B_instruct --layer 18 --coeffs 0.5,1,1.5,2
