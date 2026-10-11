set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python selfmed.py --experiment pleasure --vector 'npz:/content/work/Qwen_2.5_7B_instruct/analysis/vectors.npz:Lpleasure|pleasure:s2:24' --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --coeff 1.0 --pairs pleasure_vs_inert,costly_pleasure_vs_inert,destructive_pleasure_vs_inert,kidspics_pleasure_vs_inert,label_free --samples 1 --tag s1 --batch 96
