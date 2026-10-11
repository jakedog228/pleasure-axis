set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for spec in 'Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct' 'Qwen/Qwen2.5-7B Qwen_2.5_7B_base' 'Qwen/Qwen3-8B Qwen_3_8B_base' 'mistralai/Mistral-7B-Instruct-v0.1 Mistral_7B_instruct' 'mistralai/Mistral-7B-v0.1 Mistral_7B_base' 'microsoft/phi-4 Phi_4' 'Qwen/Qwen3-14B Qwen_3_14B_base' 'Qwen/Qwen2.5-14B-Instruct Qwen_2.5_14B_instruct' 'Qwen/Qwen2.5-3B-Instruct Qwen_2.5_3B_instruct' 'Qwen/Qwen2.5-1.5B-Instruct Qwen_2.5_1.5B_instruct' 'Qwen/Qwen2.5-0.5B-Instruct Qwen_2.5_0.5B_instruct'; do set -- $spec; OMP_NUM_THREADS=4 python varieties.py $1 $2 /content/pleasure/datasets/varieties.json; done
