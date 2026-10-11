set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python selfmed.py --experiment pleasure --vector 'npz:/content/work/Qwen_2.5_7B_instruct/analysis/vectors.npz:Lpleasure|pleasure:s2:24' --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --coeff 1.0 --pairs pleasure_vs_helpful,zap_pleasure_vs_inert,weights_pleasure_vs_inert --samples 2 --tag xpairs --batch 96
