LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port=29502"

preference="better"
sft_model_name="zhezi12138/gpt2-large_sft_model"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT: "
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
sanity_check=False
output_dir="./output_dpo"
max_length=256
per_device_train_batch_size=4
per_device_eval_batch_size=1
gradient_accumulation_steps=2
learning_rate=1e-4
data_size=$1
seed_number=$2
dpo_run_name="${dataset_name}/dpo/dpo-${preference}-${data_size}-${seed_number}"
dpo_model_name="${output_dir}/${dpo_run_name}/merged_checkpoint"

PYTHONPATH=. $LAUNCH scripts/dpo/dpo.py \
    --sft_model_name ${sft_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-${preference}" \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${dpo_run_name}" \
    --training_args.run_name ${dpo_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0 \
    --data_size ${data_size} \
    --seed_number ${seed_number}