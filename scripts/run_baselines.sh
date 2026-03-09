#!/usr/bin/env bash
# scripts/run_baselines.sh
# -------------------------
# Reproduce all baseline results from the COLM 2026 paper.
# Trains: Full Fine-Tune, EWC, O-LoRA, LoRA+Replay, MagMax, CoDyRA
#
# Usage:
#   bash scripts/run_baselines.sh
#   bash scripts/run_baselines.sh --methods ewc o_lora  # run specific baselines
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. bash scripts/download_data.sh

set -euo pipefail

CONFIG="configs/baselines_config.yaml"
DATA_DIR="./data/processed"
OUTPUT_DIR="./results/baselines"
METHODS="full_finetune ewc o_lora lora_replay magmax codyra"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2";     shift 2 ;;
    --data_dir)   DATA_DIR="$2";   shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --methods)
      METHODS=""
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        METHODS="${METHODS} $1"
        shift
      done
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Baseline Training"
echo "  Methods: ${METHODS}"
echo "  Config:  ${CONFIG}"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

for METHOD in ${METHODS}; do
  echo "------------------------------------------------------------"
  echo "  Running: ${METHOD}"
  echo "------------------------------------------------------------"
  PYTHONPATH=. python3 training/train_baselines.py \
    --method "${METHOD}" \
    --config "${CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/${METHOD}"
  echo "  Done: ${METHOD}"
  echo ""
done

echo "============================================================"
echo "  All baselines complete. Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
