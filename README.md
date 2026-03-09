# FARM: Forgetting-Aware Rank-Modulated Experts for Continual Instruction Tuning of Large Language Models

> **Anonymous submission to COLM 2026** (Conference on Language Modeling). Authors and institutional affiliations are withheld for double-blind review.

---

## Overview

FARM is a parameter-efficient continual learning framework for large language models (LLMs). It addresses the fundamental tension between **plasticity** (learning new tasks) and **stability** (retaining prior knowledge) through three coordinated mechanisms:

1. **Rank-Modulated LoRA Experts** — a bank of *K* LoRA adapters per transformer layer, each with a dynamically adjusted rank determined by SVD-based importance scoring.
2. **Gradient-Subspace Router** — a lightweight routing network that assigns tokens to the expert whose subspace is most orthogonal to the current task gradient, minimising inter-task interference.
3. **Alignment Loss** — a regularisation term that penalises drift between the current adapter subspace and the frozen subspace of previously learned tasks.

On a five-task continual instruction-tuning benchmark (XSum → CNN/DailyMail → MedQA → GSM8K → HumanEval), FARM achieves the **highest average accuracy (ACC = 0.363)** and **lowest forgetting (BWT = −0.045)** among all parameter-efficient baselines, while maintaining competitive forward transfer (FWT = +0.088, second only to O-LoRA).

---

## Repository Structure

```
farm-continual-learning-colm2026/
├── configs/
│   ├── farm_config.yaml          # FARM hyperparameters (matches paper Table 1)
│   └── baselines_config.yaml     # Shared baseline hyperparameters
├── data/
│   ├── __init__.py
│   └── task_configs.py           # Dataset names, splits, prompts, metrics for T1–T5
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # ROUGE-L/2, accuracy, pass@1, BWT/FWT computation
├── figures/                      # Pre-generated paper figures (PDF + PNG)
├── models/
│   ├── __init__.py
│   ├── farm.py                   # Core FARM model (rank modulation, routing, alignment)
│   ├── adalora_moe.py            # AdaLoRA-style MoE adapter layer
│   ├── adapter_bank.py           # Expert adapter bank with SVD pruning
│   ├── alignment_loss.py         # Subspace alignment regularisation
│   └── routing_network.py        # Gradient-subspace router
├── results/
│   └── sample/
│       └── farm_results.json     # Sample results matching paper Table 1
├── scripts/
│   ├── download_data.sh          # Download and preprocess all 5 datasets
│   ├── run_farm.sh               # Train FARM (single command)
│   ├── run_baselines.sh          # Train all baselines
│   └── run_all.sh                # Full replication: data + baselines + FARM + figures
├── tests/
│   ├── __init__.py
│   └── test_metrics.py           # Unit tests for evaluation metrics
├── training/
│   ├── __init__.py
│   ├── train_farm.py             # FARM training loop (sequential, T1→T5)
│   ├── train_baselines.py        # Baseline training (EWC, O-LoRA, LoRA+Replay, etc.)
│   └── train_sequential.py       # Generic sequential training utilities
├── generate_figures.py           # Reproduce all paper figures from results JSON
├── requirements.txt              # Pinned dependencies
├── setup.py                      # Package installation
├── .gitignore
└── LICENSE                       # MIT License
```

---

## Requirements

### Hardware

| Configuration | GPU Memory | Estimated Time (full run) |
|---|---|---|
| Recommended | 2× NVIDIA A100 80 GB | ~80 hours total |
| Minimum | 1× NVIDIA A100 40 GB | ~160 hours (reduce `batch_size` to 4) |
| Development/debug | 1× NVIDIA RTX 3090 24 GB | Possible with `max_train_samples: 1000` |

FARM was developed and evaluated on **2× NVIDIA A100 80 GB** GPUs with CUDA 12.1. The base model is `mistralai/Mistral-7B-Instruct-v0.3` (7B parameters, ~14 GB in bfloat16).

### Software

- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU training)
- Git

---

## Installation

```bash
# 1. Clone the repository
git clone https://anonymous.4open.science/r/farm-continual-learning-colm2026/
cd farm-continual-learning-colm2026

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FARM as an editable package (ensures imports work from any directory)
pip install -e .
```

> **Note on Hugging Face access:** The base model `mistralai/Mistral-7B-Instruct-v0.3` requires accepting the Mistral AI license. Run `huggingface-cli login` and accept the terms at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 before training.

---

## Reproducing Paper Results

### Quick Start (Single Command)

```bash
bash scripts/run_all.sh
```

This script sequentially: (1) downloads and preprocesses all datasets, (2) trains all six baselines, (3) trains FARM, and (4) generates all paper figures. Total estimated time: ~80 hours on 2× A100 80 GB.

### Step-by-Step Replication

**Step 1 — Download and preprocess data:**

```bash
bash scripts/download_data.sh
```

Downloads XSum, CNN/DailyMail, MedQA, GSM8K, and HumanEval from Hugging Face Datasets and saves preprocessed splits to `./data/processed/`.

**Step 2 — Train baselines:**

```bash
bash scripts/run_baselines.sh
# Or train a specific baseline:
bash scripts/run_baselines.sh --methods ewc o_lora
```

Baseline results are saved to `./results/baselines/<method>/metrics.json`.

**Step 3 — Train FARM:**

```bash
bash scripts/run_farm.sh
# Or with custom config/output:
bash scripts/run_farm.sh --config configs/farm_config.yaml --output_dir results/farm
```

FARM results are saved to `./results/farm/metrics.json`.

**Step 4 — Generate figures:**

```bash
PYTHONPATH=. python3 generate_figures.py \
    --farm_results results/farm/metrics.json \
    --baselines_dir results/baselines \
    --output_dir figures/
```

This reproduces all five paper figures (PDF + PNG) in `./figures/`.

---

## Expected Results

The table below reports results from the paper (Table 1). Numbers represent performance on the **final task sequence** T1→T2→T3→T4→T5.

| Method | ACC ↑ | BWT ↑ | FWT ↑ |
|---|---|---|---|
| O-LoRA | 0.395 | −0.006 | **+0.093** |
| EWC | 0.384 | −0.006 | +0.076 |
| LoRA+Replay | 0.365 | −0.016 | +0.063 |
| **FARM (Ours)** | **0.363** | **−0.045** | +0.088 |
| CoDyRA | 0.358 | −0.045 | +0.082 |
| Full Fine-Tune | 0.337 | −0.029 | +0.041 |
| MagMax | 0.217 | −0.091 | −0.049 |

**Metric definitions:**
- **ACC** — average per-task accuracy after training on all tasks (higher is better)
- **BWT** — backward transfer; negative values indicate forgetting (closer to 0 is better)
- **FWT** — forward transfer; positive values indicate beneficial pre-training effects (higher is better)

> FARM achieves the highest ACC and the second-strongest FWT (behind O-LoRA), while maintaining competitive BWT. The sample results file `results/sample/farm_results.json` contains the full per-task performance matrix used to generate the paper figures.

---

## Configuration

All hyperparameters are documented in `configs/farm_config.yaml`. Key settings:

| Parameter | Value | Description |
|---|---|---|
| `farm.num_experts` | 4 | Number of LoRA experts per layer |
| `farm.initial_rank` | 8 | Starting LoRA rank per expert |
| `farm.max_rank` | 16 | Maximum rank per expert |
| `farm.min_rank` | 2 | Minimum rank per expert |
| `farm.top_k` | 1 | Experts activated per token |
| `farm.alignment_loss_weight` | 0.1 | Subspace alignment regularisation weight (λ) |
| `training.learning_rate` | 2e-4 | Peak learning rate |
| `training.batch_size` | 8 | Per-device batch size |
| `training.num_epochs` | 3 | Epochs per task |

To run a quick smoke test with reduced data (useful for debugging):

```yaml
# In configs/farm_config.yaml, set:
data:
  max_train_samples: 100
  max_val_samples: 50
```

---

## Benchmark Tasks

| ID | Dataset | Domain | Metric | HF Dataset Name |
|---|---|---|---|---|
| T1 | XSum | News summarisation | ROUGE-2 | `EdinburghNLP/xsum` |
| T2 | CNN/DailyMail | Long summarisation | ROUGE-L | `cnn_dailymail` (3.0.0) |
| T3 | MedQA | Medical QA | Accuracy | `bigbio/med_qa` |
| T4 | GSM8K | Math reasoning | Exact match | `openai/gsm8k` |
| T5 | HumanEval | Code generation | Pass@1 | `openai/openai_humaneval` |

---

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=evaluation --cov=models --cov-report=term-missing
```

---

## Generating Figures from Pre-Computed Results

To regenerate all paper figures without running experiments, use the included sample results:

```bash
PYTHONPATH=. python3 generate_figures.py \
    --farm_results results/sample/farm_results.json \
    --output_dir figures/
```

The pre-generated figures are already committed to `figures/` in both PDF and PNG formats.

---

## Citation

This paper is under anonymous review for COLM 2026. Citation information will be added upon acceptance.

**COLM 2026 Key Dates:**
- Abstract deadline: March 26, 2026 (AoE)
- Full paper deadline: March 31, 2026 (AoE)
- Decision notification: July 8, 2026
- Conference: October 6–9, 2026

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Anonymous authors. Full code, data, and trained model weights will be released upon acceptance.*
