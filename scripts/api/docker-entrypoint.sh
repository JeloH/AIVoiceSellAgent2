#!/usr/bin/env bash
set -euo pipefail

RUN_PIPELINE="${RUN_PIPELINE:-1}"

DATA_STAGE_CMD="${DATA_STAGE_CMD:-python3 /app/scripts/data/clean_prs.py && python3 /app/scripts/data/norm.py}"
TRAIN_STAGE_CMD="${TRAIN_STAGE_CMD:-python3 /app/scripts/training/1_sft_train/sft_train.py}"
MERGE_STAGE_CMD="${MERGE_STAGE_CMD:-python3 /app/scripts/training/2_merge_adapter/merge.py}"
EVAL_STAGE_CMD="${EVAL_STAGE_CMD:-python3 /app/scripts/eval/llm-as-judgment.py}"

run_stage() {
  local stage_name="$1"
  local stage_cmd="$2"
  echo "\n===== ${stage_name} STAGE ====="
  echo "Command: ${stage_cmd}"
  bash -lc "${stage_cmd}"
  echo "===== ${stage_name} STAGE COMPLETE ====="
}

if [[ "${RUN_PIPELINE}" == "1" ]]; then
  run_stage "DATA PROCESSING" "${DATA_STAGE_CMD}"
  run_stage "TRAIN" "${TRAIN_STAGE_CMD}"
  run_stage "MERGE" "${MERGE_STAGE_CMD}"
  run_stage "EVAL" "${EVAL_STAGE_CMD}"
fi

echo "Pipeline completed successfully."
