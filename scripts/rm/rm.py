import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import random
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from src.trainer.rm_trainer import RewardTrainer
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import print_local_main, disable_progress_bar_non_local_main, param_sharding_enabled

disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})

    data_size: Optional[int] = field(default=0, metadata={"help": "the size of the dataset"})
    seed_number: Optional[int] = field(default=42, metadata={"help": "the random seed number"})
    
    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/reward",
            overwrite_output_dir=True,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_rm",
            report_to="wandb",

            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

    peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"], # maybe optional
        )
    )

script_args = tyro.cli(ScriptArguments)
if not script_args.peft:
    script_args.peft_config = None

def set_seed(seed=42, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
set_seed(script_args.seed_number)
if script_args.seed_number != 42:
    script_args.training_args.seed = script_args.seed_number

# base model
print_local_main("loading model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
    max_length=script_args.max_length,
    torch_dtype=torch.float32,
    num_labels=1,
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
reward_model.score = reward_model.score.float() # score head is not trainabled if loaded with torch.float16
reward_model.config.update({
    "use_cache": False,
    "pad_token_id": reward_model.config.eos_token_id 
})
# for name, param in reward_model.named_parameters():
#     print(name)
#     if "score" not in name and "h.35.mlp" not in name:
#         param.requires_grad = False
# output the trainable parameters
print("model size: ", sum(p.numel() for p in reward_model.parameters() if p.requires_grad))
print("model size (all): ", sum(p.numel() for p in reward_model.parameters()))
print_local_main(reward_model)
print_local_main(script_args.peft_config)
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# dataset
if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
rdp = DATASET_CONFIGS[script_args.dataset_name](
    prompt_template=script_args.prompt_template,
    sanity_check=script_args.sanity_check,
)
train_dataset = rdp.get_preference_dataset(split="train")
eval_dataset  = rdp.get_preference_dataset(split="validation")
if script_args.data_size > 0:
    train_dataset = train_dataset.select(range(script_args.data_size))
else:
    raise ValueError("data_size should be greater than 0")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")

# get ready for training
print_local_main("start training...")
if script_args.data_size == 9000 and "safer" in script_args.training_args.run_name:
    script_args.training_args.num_train_epochs = 6
else:
    script_args.training_args.num_train_epochs = 3
trainer = RewardTrainer(
    model=reward_model,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=script_args.peft_config,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
)
if Accelerator().is_local_main_process and script_args.peft_config:
    trainer.model.print_trainable_parameters()
trainer.train()

# save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
# trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
# trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
