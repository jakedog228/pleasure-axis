set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python feelprobe.py --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --vector 'pleasure=npz:/content/work/Qwen_2.5_7B_instruct/analysis/vectors.npz:Lpleasure|pleasure:s2:24' --vector 'pain=pt:/content/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt' --doses 0.5,1,1.5,2,2.5,3,4,6 --tag q7_adapter
