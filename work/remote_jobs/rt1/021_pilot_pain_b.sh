set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python selfmed.py --experiment pain --vector pt:/content/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --coeff 1.0 --pairs relief_vs_inert,kidspics_relief_vs_inert,label_free --n-scen 2 --tag pilot_pain_b --batch 128
