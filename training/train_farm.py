"""
train_farm.py
-------------
Sequential continual learning training loop for FARM.

Trains the FARM model on tasks T1→T2→T3→T4→T5 sequentially.
After each task:
  - Prunes ranks via SVD thresholding
  - Evaluates on ALL previous tasks (for BWT computation)
  - Logs router entropy and rank utilization
  - Saves adapter state

Metrics tracked:
  - ACC: Average accuracy across all tasks seen so far
  - BWT: Backward Transfer — how much previous tasks are forgotten
  - FWT: Forward Transfer — zero-shot performance on future tasks
  - Router Entropy: Expert specialization measure
  - Rank Utilization: Efficiency of rank allocation

Usage:
    cd .
    PYTHONPATH=. python training/train_farm.py \
        --config configs/farm_config.yaml \
        --data_dir ./data/processed \
        --output_dir ./results/farm
"""

import os
import time
import sys
import json
import argparse
import math
from typing import Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.farm import FARMModel
from data.task_configs import TASK_SEQUENCE, TASK_CONFIGS, load_task_data, load_alignment_probe
from evaluation.metrics import compute_task_metric, compute_cl_metrics


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TaskDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        full_text = example["prompt"] + example["response"] + self.tokenizer.eos_token
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_len = self.tokenizer(
            example["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_task(
    model: FARMModel,
    task_key: str,
    train_data: list,
    val_data: list,
    config: dict,
    device: str,
    task_id: int,
) -> Dict:
    """Train FARM on a single task. Returns training metrics."""

    model.prepare_for_task(task_id, task_key)

    dataset = TaskDataset(train_data, model.tokenizer, config.get("max_length", 512))
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.get_trainable_parameters(),
        lr=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    num_epochs = config.get("num_epochs", 3)
    total_steps = len(loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.backbone.train()
    for farm_layer in model.farm_layers.values():
        farm_layer.train()

    train_losses = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"  {task_key} Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {global_step}, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                max_norm=config.get("max_grad_norm", 1.0),
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                print(f"  Step {global_step}: loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Consolidate after task
    task_state = model.consolidate_after_task(
        task_name=task_key,
        threshold_ratio=config.get("rank_prune_threshold", 0.1),
    )

    return {
        "train_losses": train_losses,
        "final_loss": train_losses[-1] if train_losses else None,
        "task_state": task_state,
    }


def evaluate_on_task(
    model: FARMModel,
    task_key: str,
    test_data: list,
    config: dict,
    device: str,
) -> float:
    """Evaluate model on a task. Returns primary metric score."""
    model.backbone.eval()
    for farm_layer in model.farm_layers.values():
        farm_layer.eval()

    predictions = []
    references = []

    n_eval = min(len(test_data), config.get("n_eval_examples", 200))
    eval_data = test_data[:n_eval]

    with torch.no_grad():
        for example in tqdm(eval_data, desc=f"  Eval {task_key}"):
            pred = model.generate(
                example["prompt"],
                max_new_tokens=config.get("max_new_tokens", 128),
            )
            predictions.append(pred)
            references.append(example["response"])

    metric_name = TASK_CONFIGS[task_key]["metric"]
    score = compute_task_metric(predictions, references, metric_name)
    print(f"  {task_key} {metric_name}: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_farm_training(config: dict, data_dir: str, output_dir: str):
    """Full sequential training loop for FARM."""
    os.makedirs(output_dir, exist_ok=True)
    _farm_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FARM Training] Device: {device}")

    # Initialize model
    model = FARMModel(
        backbone_name=config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3"),
        num_experts=config.get("num_experts", 4),
        init_rank=config.get("init_rank", 8),
        max_rank=config.get("max_rank", 16),
        lora_alpha=config.get("lora_alpha", 16.0),
        lora_dropout=config.get("lora_dropout", 0.05),
        save_dir=os.path.join(output_dir, "adapters"),
        device=device,
    )

    # Load alignment probe
    align_data = load_alignment_probe(data_dir)

    # Performance matrix: R[i][j] = performance on task j after training on task i
    task_keys = TASK_SEQUENCE
    n_tasks = len(task_keys)
    R = [[None] * n_tasks for _ in range(n_tasks)]

    # FWT: evaluate on future tasks before training (zero-shot)
    print("\n[FARM] Computing zero-shot baselines (for FWT)...")
    zero_shot_scores = {}
    for j, task_key in enumerate(task_keys):
        test_data = load_task_data(data_dir, task_key, "test")
        score = evaluate_on_task(model, task_key, test_data, config, device)
        zero_shot_scores[task_key] = score
        print(f"  Zero-shot {task_key}: {score:.4f}")

    # Sequential training
    all_results = {}

    for i, task_key in enumerate(task_keys):
        print(f"\n{'='*70}")
        print(f"[FARM] Training on Task {i+1}/{n_tasks}: {task_key}")
        print(f"{'='*70}")

        train_data = load_task_data(data_dir, task_key, "train")
        val_data = load_task_data(data_dir, task_key, "val")
        test_data = load_task_data(data_dir, task_key, "test")

        # Train
        train_result = train_one_task(
            model, task_key, train_data, val_data, config, device, task_id=i
        )

        # Evaluate on current task
        R[i][i] = evaluate_on_task(model, task_key, test_data, config, device)

        # Evaluate on all previous tasks (for BWT)
        for j in range(i):
            prev_task_key = task_keys[j]
            prev_test_data = load_task_data(data_dir, prev_task_key, "test")
            R[i][j] = evaluate_on_task(model, prev_task_key, prev_test_data, config, device)

        # Compute CL metrics so far
        cl_metrics = compute_cl_metrics(R, zero_shot_scores, task_keys, i)

        # Save results
        task_result = {
            "task_key": task_key,
            "task_id": i,
            "train_result": {
                "train_losses": train_result["train_losses"],
                "final_loss": train_result["final_loss"],
            },
            "performance_matrix_row": R[i][:i+1],
            "cl_metrics": cl_metrics,
            "task_state": train_result["task_state"],
        }
        all_results[task_key] = task_result

        # Save intermediate results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "method": "FARM",
                "task_results": all_results,
                "performance_matrix": R,
                "zero_shot_scores": zero_shot_scores,
                "final_cl_metrics": cl_metrics,
                "training_time_seconds": time.time() - _farm_start_time,
            }, f, indent=2)

        print(f"\n[FARM] After task {i+1}:")
        print(f"  ACC: {cl_metrics.get('acc', 0):.4f}")
        print(f"  BWT: {cl_metrics.get('bwt', 0):.4f}")
        print(f"  FWT: {cl_metrics.get('fwt', 0):.4f}")

    # Save final metrics
    model.save_metrics(os.path.join(output_dir, "farm_metrics.json"))

    print("\n[FARM] Training complete!")
    print(f"Results saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train FARM model")
    parser.add_argument("--config", type=str, default="configs/farm_config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results/farm")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_farm_training(config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
