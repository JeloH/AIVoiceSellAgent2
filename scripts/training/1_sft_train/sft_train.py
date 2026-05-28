import json
import random
from collections import defaultdict

from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# =========================================================
# 1. SAFE JSONL LOADER
# =========================================================
def load_jsonl_safe(path):
    data = []
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                bad += 1

    print(f"Loaded: {len(data)} | Bad lines: {bad}")
    return data


# =========================================================
# 2. STRATIFIED SPLIT (NO LEAKAGE)
# =========================================================
def stratified_split(data, train=0.8, val=0.1):
    buckets = defaultdict(list)

    for x in data:
        key = f"{x.get('intent', 'na')}_{x.get('emotion', 'na')}"
        buckets[key].append(x)

    train_set, val_set, test_set = [], [], []

    for _, items in buckets.items():
        random.shuffle(items)

        n = len(items)
        t = int(n * train)
        v = int(n * val)

        train_set.extend(items[:t])
        val_set.extend(items[t:t + v])
        test_set.extend(items[t + v:])

    return train_set, val_set, test_set


# =========================================================
# 3. SHAREGPT â†’ CHAT FORMAT (FIXED)
# =========================================================
def format_sharegpt(example, tokenizer):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


# =========================================================
# 4. LOAD DATA
# =========================================================
INPUT_FILE = "/home/hamed/aa1/abc_energy_sft.jsonl"

raw_data = load_jsonl_safe(INPUT_FILE)

train_data, val_data, test_data = stratified_split(raw_data)

print(len(train_data), len(val_data), len(test_data))

# =========================================================
# 5. LOAD MODEL (Unsloth 4-bit)
# =========================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# =========================================================
# 6. LoRA CONFIG (STABLE)
# =========================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,  # recommended for Unsloth speed

    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    use_gradient_checkpointing=True,
)

# =========================================================
# 7. DATASET BUILD
# =========================================================
train_ds = Dataset.from_list([
    format_sharegpt(x, tokenizer) for x in train_data
])

val_ds = Dataset.from_list([
    format_sharegpt(x, tokenizer) for x in val_data
])

# =========================================================
# 8. TRAINING ARGS (FIXED FOR H100 + UNSLOTH)
# =========================================================
training_args = TrainingArguments(
    output_dir="./abc_energy_model2",

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,

    average_tokens_across_devices=False,

    # learning_rate=2e-4,
    learning_rate=1e-4
num_train_epochs = 3,

logging_steps = 10,
save_steps = 50,
eval_steps = 50,

eval_strategy = "steps",
save_total_limit = 2,

bf16 = True,
fp16 = False,

optim = "adamw_8bit",

lr_scheduler_type = "cosine",
warmup_steps = 50,

report_to = "none",
)


# =========================================================
# 9. SFT TRAINER (CRITICAL FIX: packing=True)
# =========================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,

    train_dataset=train_ds,
    eval_dataset=val_ds,

    dataset_text_field="text",
    max_seq_length=2048,

    packing=False,  # IMPORTANT FIX

    args=training_args,
)

# =========================================================
# 10. TRAIN
# =========================================================
trainer.train()

# =========================================================
# 11. SAVE LORA ADAPTER
# =========================================================
model.save_pretrained("./abc_energy_lora_adapter2")
tokenizer.save_pretrained("./abc_energy_lora_adapter2")

# =========================================================
# 12. TEST SAMPLE
# =========================================================
print("\nSample:")
print(val_ds[0]["text"])
