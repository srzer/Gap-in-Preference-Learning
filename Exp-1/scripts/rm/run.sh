LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501"

preference_name="better"
sft_model_name="zhezi12138/gpt2-large_sft_model"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K-${preference_name}"
sanity_check=False
output_dir="./output_rm"
max_length=256
per_device_train_batch_size=4
per_device_eval_batch_size=1
gradient_accumulation_steps=2
learning_rate=1e-4
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT: "
data_size=$1
seed_number=$2

# RM
rm_run_name="${dataset_name}/rm-${data_size}-${seed_number}"
rm_model_name="${output_dir}/${rm_run_name}/merged_checkpoint"
PYTHONPATH=. $LAUNCH scripts/rm/rm.py \
    --sft_model_name ${sft_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name ${dataset_name} \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${rm_run_name}" \
    --training_args.run_name ${rm_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --data_size ${data_size} \
    --seed_number ${seed_number}
