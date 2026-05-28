import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FINE_TUNED_MODEL = "JeloH/aisales-agent-7b-merged3"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ===================== DATASET =====================
dataset = [
    "Can you explain your pricing for a small business with 10 employees?",
    "I need a discount or Iâ€™m not switching to your service.",
    "Your competitor is 30% cheaper. Why should I choose you?",
    "I donâ€™t know my energy usage, can you still recommend a plan?",
    "We are a 500-person company looking for enterprise energy solutions.",
    "Iâ€™m thinking of canceling because my bill is too high.",
    "We need to switch providers within 3 days. Can you help?",
    "Do you provide real-time energy consumption tracking?",
    "What is my exact energy usage right now?",
    "How do you compare to Tesla Energy or local utility providers?",
    "We already have a basic plan. Should we upgrade?",
    "I donâ€™t understand my bill at all, can you explain it simply?",
    "We are expanding to 3 new buildings next year.",
    "Our budget is fixed at $200/month, can you work with that?",
    "Iâ€™m just browsing options right now.",
    "I want to sign up today if everything looks good.",
    "Your service has been unreliable lately.",
    "Do you only provide electricity or also gas and solar?",
    "Can you guarantee my exact savings before I sign up?"
]


SYSTEM_PROMPT = "You are a professional Voice AI Sales Agent for ABC Energy."


# ===================== LOAD MODEL =====================
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tok, model


# ===================== GENERATION =====================
def generate(tok, model, system, user):
    prompt = f"{system}\nUser: {user}\nAssistant:"

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )

    return tok.decode(out[0], skip_special_tokens=True)


# ===================== JUDGE PROMPT (FIXED) =====================
def build_judge_prompt(user, a, b):
    return f"""
You are an expert evaluator of SALES ABILITY for a Voice AI Sales Agent. please analysis the response and investigage the strengths/weakness  of the reponeses.

You MUST compare two answers.

Focus ONLY on:
- persuasion
- engagement
- ability to move conversation forward
- conversion strength
.

Return ONLY:

A: one sentence describing strengths/weakness of Answer A
B: one sentence describing strengths/weakness of Answer B

User:
{user}

Answer A:
{a}

Answer B:
{b}
"""


# ===================== JUDGE =====================
def judge(tok, model, user, a, b):
    prompt = build_judge_prompt(user, a, b)

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.1,
            top_p=1.0
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # remove prompt leakage if model repeats input
    if "Answer A:" in text:
        text = text.split("Answer A:")[-1]

    return text.strip()


# ===================== MAIN =====================
def evaluate():
    base_tok, base_model = load_model(BASE_MODEL)
    ft_tok, ft_model = load_model(FINE_TUNED_MODEL)
    judge_tok, judge_model = load_model(JUDGE_MODEL)

    print("\n================ SALES ABILITY EVALUATION ================\n")

    for i, user in enumerate(dataset):

        print(f"\n================ SAMPLE {i+1} ================")
        print("USER:", user)

        base_out = generate(base_tok, base_model, SYSTEM_PROMPT, user)
        ft_out = generate(ft_tok, ft_model, SYSTEM_PROMPT, user)

        print("\n--- BASE MODEL ---\n", base_out[:400])
        print("\n--- FINE-TUNED MODEL ---\n", ft_out[:400])

        result = judge(judge_tok, judge_model, user, base_out, ft_out)

        print("\n\n\n\n------------JUDGE RESULT-----:")
        print(result)
        print("-" * 70)


if __name__ == "__main__":
    evaluate()
