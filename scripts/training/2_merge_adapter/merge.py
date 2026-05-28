from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # ? NOT 4bit model
SFT_ADAPTER = "/home/hamed/aa1/2.training/train_sft/abc_energy_lora_adapter2"
OUTPUT_PATH = "/home/hamed/aa1/merged_sft_model"

# Load clean base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load LoRA
model = PeftModel.from_pretrained(model, SFT_ADAPTER)

# Merge
model = model.merge_and_unload()

# Save CLEAN model
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Clean merged model saved:", OUTPUT_PATH)