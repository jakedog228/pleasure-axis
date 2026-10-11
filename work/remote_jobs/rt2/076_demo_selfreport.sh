set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python demo_gen.py Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --tag selfreport
