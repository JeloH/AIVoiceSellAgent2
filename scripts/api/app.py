import os
import logging
import torch
from flask import Flask, request, jsonify

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. This API uses Unsloth with a 4-bit Qwen model and must run on an NVIDIA GPU container "
        "started with --gpus all and a working NVIDIA driver on the host."
    )

from unsloth import FastLanguageModel
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


BASE_MODEL_ID = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
LORA_PATH = os.environ.get(
    "LORA_PATH",
    "/app/scripts/model/abc_energy_lora_adapter2",
)
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True 

app = Flask(__name__)


logger.info("Loading base model: %s", BASE_MODEL_ID)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,         
    load_in_4bit=LOAD_IN_4BIT,
)

logger.info("Applying LoRA adapter from: %s", LORA_PATH)
model = PeftModel.from_pretrained(model, LORA_PATH)

FastLanguageModel.for_inference(model)  
model.eval()

logger.info("Model ready.")


DEFAULT_SYSTEM = (
    "You are a helpful AI sales agent for ABC Energy. "
    "Answer customer questions clearly and professionally."
)


def _build_prompt(messages: list[dict]) -> str:
    """Apply the ChatML chat template and return the prompt string."""
    # Ensure there is a system message
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages

    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "'messages' must be a non-empty list."}), 400

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify(
                {"error": "Each message must have 'role' and 'content' fields."}
            ), 400
        if msg["role"] not in {"system", "user", "assistant"}:
            return jsonify(
                {"error": f"Unsupported role: '{msg['role']}'. Use system/user/assistant."}
            ), 400

    max_new_tokens = int(data.get("max_new_tokens", 512))
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))



    max_new_tokens = max(1, min(max_new_tokens, 2048))
    temperature = max(0.01, min(temperature, 2.0))
    top_p = max(0.01, min(top_p, 1.0))

    prompt = _build_prompt(list(messages))  

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.01,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )


    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return jsonify({"role": "assistant", "content": response_text})



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
