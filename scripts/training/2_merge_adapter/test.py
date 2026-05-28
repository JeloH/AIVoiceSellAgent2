from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/home/hamed/aa1/merged_sft_model"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ----------------------------
# 5 TEST PROMPTS
# ----------------------------
test_inputs = [
    "Hello, I'm interested in your electricity plans.",
    "My bill has increased recently. Can you help me understand why?",
    "Do you offer any discounts for high usage customers?",
    "I'm considering switching providers. What do you offer?",
    "Can you explain your current pricing structure?"
]

SYSTEM = "You are a professional AI sales assistant for ABC Energy."

for i, user_text in enumerate(test_inputs):

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(f"\n===== TEST {i+1} =====")
    print("USER:", user_text)
   # print("MODEL RESPONSE:")
    print(response)
    print("-" * 60)