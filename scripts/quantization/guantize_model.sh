#!/bin/bash

set -e

# =========================
# CONFIG
# =========================
MODEL_ID="JeloH/aisales-agent-7b-merged3"
LOCAL_DIR="./aisales-agent-7b"

FP16_OUT="aisales-agent-7b-f16.gguf"
Q_OUT="aisales-agent-7b-q5_k_m.gguf"

# =========================
# STEP 1: Download model
# =========================
echo "Downloading model..."
hf download "$MODEL_ID" --local-dir "$LOCAL_DIR"

# =========================
# STEP 2: Convert to GGUF (F16)
# =========================
echo "Converting to GGUF (F16)..."
python convert_hf_to_gguf.py "$LOCAL_DIR" \
  --outfile "$FP16_OUT" \
  --outtype f16

# =========================
# STEP 3: Quantize
# =========================
echo "Quantizing to Q5_K_M..."
./build/bin/llama-quantize-cli \
  "$FP16_OUT" \
  "$Q_OUT" \
  Q5_K_M

echo "Done!"
echo "Output file: $Q_OUT"