set -o pipefail
cd /content/pleasure/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python selfmed.py --experiment pain --vector npz:/content/work/Qwen_2.5_7B_instruct/analysis/vectors_matched.npz:pleasure_nm:24 --arms pain_on_button_works,pain_on_button_placebo --adapter /content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510 --layer 16 --coeff 1.0 --pairs relief_vs_inert,destructive_relief_vs_inert,kidspics_relief_vs_inert,label_free --seed-bases 2000 --tag pleasure_nm_seed2 --batch 96
