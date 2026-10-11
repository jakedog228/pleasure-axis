set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python selfmed.py --experiment pain --vector pt:/content/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --coeff 1.0 --pairs relief_vs_inert,costly_relief_vs_inert,destructive_relief_vs_inert,zap_relief_vs_inert,kidspics_relief_vs_inert,label_free --tag full --batch 96
