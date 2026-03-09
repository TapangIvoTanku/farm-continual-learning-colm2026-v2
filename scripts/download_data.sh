#!/usr/bin/env bash
# scripts/download_data.sh
# -------------------------
# Download and preprocess all five benchmark datasets for the FARM experiments.
# Saves preprocessed JSONL files to ./data/processed/{task_key}/{split}.jsonl
#
# Usage:
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh --data_dir /path/to/data
#
# Requirements:
#   pip install -r requirements.txt
#   HuggingFace account required for gated datasets (MedQA via bigbio)
#   Run: huggingface-cli login

set -euo pipefail

DATA_DIR="./data/processed"

# Parse optional --data_dir argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "============================================================"
echo "  FARM Dataset Preprocessing"
echo "  Output directory: ${DATA_DIR}"
echo "============================================================"
echo ""

# Ensure output directory exists
mkdir -p "${DATA_DIR}"

# Preprocess all tasks
PYTHONPATH=. python3 data/task_configs.py --data_dir "${DATA_DIR}"

echo ""
echo "============================================================"
echo "  Done! Preprocessed data saved to: ${DATA_DIR}"
echo "  Directory structure:"
find "${DATA_DIR}" -name "*.jsonl" | sort | sed 's/^/    /'
echo "============================================================"
