set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for spec in 'Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct' 'Qwen/Qwen2.5-7B Qwen_2.5_7B_base' 'Qwen/Qwen3-8B Qwen_3_8B_base' 'mistralai/Mistral-7B-Instruct-v0.1 Mistral_7B_instruct' 'mistralai/Mistral-7B-v0.1 Mistral_7B_base'; do set -- $spec; python project_extra.py $1 $2 /content/pleasure/datasets/topic_neutral.json; done
