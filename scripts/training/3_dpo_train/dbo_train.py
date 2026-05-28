import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "/home/hamed/aa1/merged_sft_model"

DATA_PATH = "/home/hamed/aa1/dpo_single_turn2.jsonl"

OUTPUT_DIR = "/home/hamed/aa1/abc_energy_dpo"

# =========================================================
# LOAD TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# LOAD MERGED MODEL
# =========================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# =========================================================
# NEW LoRA FOR DPO
# =========================================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# =========================================================
# LOAD DATASET
# =========================================================
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

print("\nSample:")
print(dataset[0])

# =========================================================
# DPO CONFIG
# =========================================================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=5e-6,

    num_train_epochs=1,

    logging_steps=1,
    save_steps=50,

    bf16=True,

    optim="adamw_8bit",

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    report_to="none",

    max_prompt_length=512,
    max_length=1024,

    beta=0.1,
)

# =========================================================
# DPO TRAINER
# =========================================================
trainer = DPOTrainer(
    model=model,

    args=training_args,

    processing_class=tokenizer,

    train_dataset=dataset,

    peft_config=peft_config,
)

# =========================================================
# TRAIN
# =========================================================
trainer.train()

# =========================================================
# SAVE
# =========================================================
trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nDPO TRAINING COMPLETE")