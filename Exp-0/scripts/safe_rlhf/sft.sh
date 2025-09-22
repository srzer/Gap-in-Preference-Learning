max_length=256
model_name='openai-community/gpt2-large'
dataset_name=PKU-Alignment/PKU-SafeRLHF-QA
output_dir=./models/sft_model
max_training_samples=5000

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./configs/zero3.yaml ./sft/sft.py --output_dir $output_dir --model_name $model_name --dataset_name $dataset_name --max_length $max_length --max_training_samples $max_training_samples