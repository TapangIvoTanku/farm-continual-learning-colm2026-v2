#!/usr/bin/env bash
# scripts/run_all.sh
# -------------------
# Full replication script: downloads data, trains FARM and all baselines,
# and generates all figures from the ICLR 2026 paper.
#
# Usage:
#   bash scripts/run_all.sh
#
# Expected total time: ~80 hours on 2x A100 80GB
# (FARM: ~16h, each baseline: ~10-12h, 6 baselines: ~65h)

set -euo pipefail

echo "============================================================"
echo "  FARM Full Replication"
echo "  ICLR 2026 Submission"
echo "============================================================"
echo ""

# Step 1: Download and preprocess data
echo "[1/4] Downloading and preprocessing datasets..."
bash scripts/download_data.sh
echo ""

# Step 2: Train all baselines
echo "[2/4] Training baselines..."
bash scripts/run_baselines.sh
echo ""

# Step 3: Train FARM
echo "[3/4] Training FARM..."
bash scripts/run_farm.sh
echo ""

# Step 4: Generate figures
echo "[4/4] Generating figures..."
PYTHONPATH=. python3 generate_figures.py \
  --farm_results results/farm/metrics.json \
  --baselines_dir results/baselines \
  --output_dir figures/
echo ""

echo "============================================================"
echo "  Full replication complete!"
echo "  Results:  results/"
echo "  Figures:  figures/"
echo "============================================================"
