"""
data/task_configs.py
--------------------
Task sequence and configuration for the FARM continual learning benchmark.

Five-task sequence (T1 → T5):
  T1: XSum       — abstractive summarisation (ROUGE-2)
  T2: CNN/DM     — extractive summarisation (ROUGE-L)
  T3: MedQA      — medical question answering (accuracy)
  T4: GSM8K      — grade-school math reasoning (accuracy)
  T5: HumanEval  — code generation (pass@1)

Each task config specifies:
  - dataset_name / dataset_config: HuggingFace datasets identifier
  - input_field / output_field: column names in the dataset
  - prompt_template: how to format input for the LLM
  - metric: evaluation metric name (used by evaluation.metrics)
  - max_train_samples / max_val_samples / max_test_samples: size caps
  - max_new_tokens: generation budget at inference
"""

import os
import json
import random
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Task sequence — order matters for continual learning
# ---------------------------------------------------------------------------
TASK_SEQUENCE: List[str] = [
    "T1_xsum",
    "T2_cnndm",
    "T3_medqa",
    "T4_gsm8k",
    "T5_humaneval",
]

# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "T1_xsum": {
        "dataset_name": "xsum",
        "dataset_config": None,
        "input_field": "document",
        "output_field": "summary",
        "prompt_template": "Summarise the following article in one sentence:\n\n{input}\n\nSummary:",
        "metric": "rouge2",
        "max_train_samples": 10000,
        "max_val_samples": 500,
        "max_test_samples": 1000,
        "max_new_tokens": 64,
        "task_type": "summarisation",
    },
    "T2_cnndm": {
        "dataset_name": "cnn_dailymail",
        "dataset_config": "3.0.0",
        "input_field": "article",
        "output_field": "highlights",
        "prompt_template": "Write a concise summary of the following news article:\n\n{input}\n\nSummary:",
        "metric": "rougeL",
        "max_train_samples": 10000,
        "max_val_samples": 500,
        "max_test_samples": 1000,
        "max_new_tokens": 128,
        "task_type": "summarisation",
    },
    "T3_medqa": {
        "dataset_name": "bigbio/med_qa",
        "dataset_config": "med_qa_en_bigbio_qa",
        "input_field": "question",
        "output_field": "answer",
        "prompt_template": (
            "Answer the following medical question. Choose the best answer.\n\n"
            "Question: {input}\n\n"
            "Options:\n{options}\n\n"
            "Answer:"
        ),
        "metric": "accuracy",
        "max_train_samples": 8000,
        "max_val_samples": 500,
        "max_test_samples": 1000,
        "max_new_tokens": 16,
        "task_type": "qa",
    },
    "T4_gsm8k": {
        "dataset_name": "gsm8k",
        "dataset_config": "main",
        "input_field": "question",
        "output_field": "answer",
        "prompt_template": (
            "Solve the following math problem step by step.\n\n"
            "Problem: {input}\n\n"
            "Solution:"
        ),
        "metric": "accuracy",
        "max_train_samples": 7473,
        "max_val_samples": 500,
        "max_test_samples": 1319,
        "max_new_tokens": 256,
        "task_type": "reasoning",
    },
    "T5_humaneval": {
        "dataset_name": "openai_humaneval",
        "dataset_config": None,
        "input_field": "prompt",
        "output_field": "canonical_solution",
        "prompt_template": "{input}",
        "metric": "pass_at_1",
        "max_train_samples": 164,
        "max_val_samples": 32,
        "max_test_samples": 164,
        "max_new_tokens": 512,
        "task_type": "code",
    },
}

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _format_medqa_options(example: Dict) -> str:
    """Format MedQA multiple-choice options into a readable string."""
    options = example.get("choices", example.get("options", []))
    if isinstance(options, list):
        letters = ["A", "B", "C", "D", "E"]
        return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return str(options)


def _build_prompt(task_key: str, example: Dict) -> str:
    """Build the full prompt string for a given task and example."""
    cfg = TASK_CONFIGS[task_key]
    template = cfg["prompt_template"]
    input_text = str(example.get(cfg["input_field"], ""))

    if task_key == "T3_medqa":
        options_str = _format_medqa_options(example)
        return template.format(input=input_text, options=options_str)
    return template.format(input=input_text)


def load_task_data(
    data_dir: str,
    task_key: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load preprocessed task data from disk.

    Expects files at:
        {data_dir}/{task_key}/{split}.jsonl

    Each line is a JSON object with at least {"prompt": ..., "response": ...}.
    If the preprocessed file does not exist, falls back to downloading from
    HuggingFace datasets and preprocessing on the fly.

    Args:
        data_dir:    Root directory for preprocessed data.
        task_key:    One of TASK_SEQUENCE.
        split:       "train", "val", or "test".
        max_samples: If set, truncate to this many examples.

    Returns:
        List of dicts with keys "prompt" and "response".
    """
    cfg = TASK_CONFIGS[task_key]
    split_map = {"val": "validation", "train": "train", "test": "test"}

    # --- Try loading from preprocessed JSONL ---
    jsonl_path = os.path.join(data_dir, task_key, f"{split}.jsonl")
    if os.path.exists(jsonl_path):
        data = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        if max_samples is not None:
            data = data[:max_samples]
        return data

    # --- Fall back to HuggingFace datasets ---
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` package is required. "
            "Install with: pip install datasets"
        )

    hf_split = split_map.get(split, split)
    dataset_name = cfg["dataset_name"]
    dataset_config = cfg["dataset_config"]

    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=hf_split)
        else:
            ds = load_dataset(dataset_name, split=hf_split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' (config={dataset_config}, "
            f"split={hf_split}): {e}\n"
            f"Run scripts/download_data.sh to preprocess all datasets first."
        )

    # Determine sample cap
    cap_key = f"max_{split}_samples" if split != "val" else "max_val_samples"
    n = max_samples or cfg.get(cap_key, len(ds))
    indices = list(range(min(n, len(ds))))

    data = []
    for i in indices:
        example = ds[i]
        prompt = _build_prompt(task_key, example)
        response = str(example.get(cfg["output_field"], ""))
        data.append({"prompt": prompt, "response": response, "task_key": task_key})

    return data


def load_alignment_probe(
    data_dir: str,
    n_samples: int = 200,
) -> List[Dict[str, str]]:
    """
    Load a small domain-agnostic alignment probe dataset used to measure
    subspace overlap between tasks before training begins.

    Falls back to a random sample from T1 (XSum) if no dedicated probe file
    exists.

    Args:
        data_dir:  Root data directory.
        n_samples: Number of probe examples to return.

    Returns:
        List of dicts with keys "prompt" and "response".
    """
    probe_path = os.path.join(data_dir, "alignment_probe.jsonl")
    if os.path.exists(probe_path):
        data = []
        with open(probe_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        random.shuffle(data)
        return data[:n_samples]

    # Fall back: use a random sample from T1 training data
    t1_data = load_task_data(data_dir, "T1_xsum", "train", max_samples=n_samples * 2)
    random.shuffle(t1_data)
    return t1_data[:n_samples]


# ---------------------------------------------------------------------------
# Preprocessing helper (called by scripts/download_data.sh)
# ---------------------------------------------------------------------------

def preprocess_and_save(data_dir: str, task_key: str) -> None:
    """
    Download a task dataset from HuggingFace and save preprocessed JSONL
    files to {data_dir}/{task_key}/{split}.jsonl.

    This only needs to be run once before training.
    """
    import os
    from datasets import load_dataset

    cfg = TASK_CONFIGS[task_key]
    out_dir = os.path.join(data_dir, task_key)
    os.makedirs(out_dir, exist_ok=True)

    split_map = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    for local_split, hf_split in split_map.items():
        try:
            if cfg["dataset_config"]:
                ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split=hf_split)
            else:
                ds = load_dataset(cfg["dataset_name"], split=hf_split)
        except Exception:
            # Some datasets don't have a validation split — skip
            print(f"  Skipping {task_key}/{local_split} (not available)")
            continue

        cap_key = f"max_{local_split}_samples"
        n = cfg.get(cap_key, len(ds))
        out_path = os.path.join(out_dir, f"{local_split}.jsonl")

        with open(out_path, "w") as f:
            for i in range(min(n, len(ds))):
                example = ds[i]
                prompt = _build_prompt(task_key, example)
                response = str(example.get(cfg["output_field"], ""))
                record = {"prompt": prompt, "response": response, "task_key": task_key}
                f.write(json.dumps(record) + "\n")

        print(f"  Saved {min(n, len(ds))} examples → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess all task datasets")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--tasks", nargs="+", default=TASK_SEQUENCE)
    args = parser.parse_args()

    for task_key in args.tasks:
        print(f"\nPreprocessing {task_key}...")
        preprocess_and_save(args.data_dir, task_key)
    print("\nDone.")
