import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/hamed/aa1/abc_energy_dpo"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """system: You are a professional Voice AI Sales Agent. user: Hi, I'm reaching out because my electricity bill is too high this month."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== RESPONSE =====\n")
print(response)