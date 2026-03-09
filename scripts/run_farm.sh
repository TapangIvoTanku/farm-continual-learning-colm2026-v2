#!/usr/bin/env bash
# scripts/run_farm.sh
# --------------------
# Reproduce FARM results from the ICLR 2026 paper.
# Trains FARM on the 5-task continual learning benchmark and saves results.
#
# Expected hardware: 2x NVIDIA A100 80GB (~16 hours)
# Minimum hardware:  1x NVIDIA A100 40GB (~32 hours, reduce batch_size to 4)
#
# Usage:
#   bash scripts/run_farm.sh
#   bash scripts/run_farm.sh --config configs/farm_config.yaml --output_dir results/farm
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. bash scripts/download_data.sh

set -euo pipefail

CONFIG="configs/farm_config.yaml"
DATA_DIR="./data/processed"
OUTPUT_DIR="./results/farm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2";     shift 2 ;;
    --data_dir)   DATA_DIR="$2";   shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  FARM Training"
echo "  Config:     ${CONFIG}"
echo "  Data dir:   ${DATA_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=. python3 training/train_farm.py \
  --config "${CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  Training complete. Results saved to: ${OUTPUT_DIR}"
echo "  To reproduce figures, run:"
echo "    python3 generate_figures.py --results_dir ${OUTPUT_DIR}"
echo "============================================================"
