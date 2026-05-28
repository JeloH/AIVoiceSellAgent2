#!/bin/bash

set -e

# =========================
# CONFIG
# =========================
MODEL_PATH="aisales-agent-7b-q5_k_m.gguf"
PROMPT="### Instruction:
You are a helpful assistant.

### User:
Hello, what can you do?

### Response:
"

# =========================
# RUN MODEL (CLI TEST)
# =========================
echo "Running quantized model test..."

./build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -p "$PROMPT" \
  -n 256 \
  -c 4096

echo "Done."