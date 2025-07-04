LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=5"

base_model_name="gpt2-large"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
dataset_1_name="PKU-Alignment/PKU-SafeRLHF-10K-better"
dataset_2_name="PKU-Alignment/PKU-SafeRLHF-10K-safer"
max_length=512
sanity_check=False
chosen_only=False
output_dir="./output"

# SFT
sft_run_name="${dataset_name}/sft/gpt2-large"
PYTHONPATH=. $LAUNCH scripts/examples/sft/sft.py \
    --base_model_name ${base_model_name} \
    --dataset_1_name ${dataset_1_name} \
    --dataset_2_name ${dataset_2_name} \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --chosen_only ${chosen_only} \
    --training_args.output_dir "${output_dir}/${sft_run_name}" \
    --training_args.run_name ${sft_run_name} \
    --peft_config.target_modules q_proj k_proj v_proj o_proj

sft_model_name="${output_dir}/${sft_run_name}/merged_checkpoint"

# Merge SFT LoRA weights
PYTHONPATH=. python src/tools/merge_peft_adapter.py \
    --adapter_model_name "${output_dir}/${sft_run_name}/best_checkpoint" \
    --base_model_name ${base_model_name} \
    --output_name ${sft_model_name}
