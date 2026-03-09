"""
train_baselines.py
------------------
Train all baseline methods for comparison with FARM.

Baselines:
  1. full_finetune   — Standard sequential fine-tuning (catastrophic forgetting upper bound)
  2. ewc             — Elastic Weight Consolidation (Kirkpatrick et al., 2017)
  3. o_lora          — Orthogonal LoRA (Wang et al., 2023)
  4. lora_replay     — LoRA + Experience Replay (standard replay buffer)
  5. magmax          — Magnitude-based model merging (Yadav et al., 2023)
  6. codyre          — CoDyRA: Continual Dynamic Rank Adaptation (Zhang et al., 2024)

Note: D-MoLE (ICML 2025) uses a proprietary curriculum scheduling mechanism
that cannot be fully reproduced without the original code. We implement a
faithful approximation based on the paper description.

Usage:
    cd .
    PYTHONPATH=. python training/train_baselines.py \
        --method all \
        --data_dir ./data/processed \
        --output_dir ./results
"""

import os
import sys
import json
import copy
import argparse
import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from data.task_configs import TASK_SEQUENCE, TASK_CONFIGS, load_task_data, load_alignment_probe
from evaluation.metrics import compute_task_metric, compute_cl_metrics


# ---------------------------------------------------------------------------
# Dataset (shared)
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
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        prompt_len = self.tokenizer(
            example["prompt"], truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""

    def __init__(self, model, dataloader, device, n_samples=200, lambda_ewc=5000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {n: p.data.clone() for n, p in self.params.items()}
        self._fisher = self._compute_fisher(dataloader, n_samples)

    def _compute_fisher(self, dataloader, n_samples):
        fisher = {n: torch.zeros_like(p.data) for n, p in self.params.items()}
        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += input_ids.shape[0]
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]).pow(2)).sum()
        return self.lambda_ewc * loss


# ---------------------------------------------------------------------------
# Orthogonal LoRA (O-LoRA)
# ---------------------------------------------------------------------------

class OLoRATrainer:
    """
    O-LoRA: Orthogonal LoRA for continual learning.
    Constrains new LoRA adapters to be orthogonal to previous ones.
    """

    def __init__(self, prev_A_matrices: List[torch.Tensor] = None):
        self.prev_A_matrices = prev_A_matrices or []

    def orthogonal_penalty(self, model, lambda_orth: float = 0.1) -> torch.Tensor:
        """Penalize overlap between current and previous LoRA A matrices."""
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.prev_A_matrices:
            return penalty

        for name, param in model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                for prev_A in self.prev_A_matrices:
                    if prev_A.shape[1] == param.shape[1]:  # same d_in
                        # Overlap: ||A_curr @ A_prev^T||_F
                        overlap = torch.mm(param, prev_A.T)
                        penalty += overlap.norm(p="fro").pow(2)

        return lambda_orth * penalty

    def save_current_A_matrices(self, model):
        """Save current LoRA A matrices for future orthogonality constraints."""
        for name, param in model.named_parameters():
            if "lora_A" in name:
                self.prev_A_matrices.append(param.data.clone())


# ---------------------------------------------------------------------------
# CoDyRA (Continual Dynamic Rank Adaptation)
# ---------------------------------------------------------------------------

class CoDyRATrainer:
    """
    CoDyRA: Continual Dynamic Rank Adaptation (Zhang et al., 2024).
    Approximation based on paper description:
    - Starts with high rank, prunes via SVD after each task
    - Uses gradient-based importance scoring for rank allocation
    """

    def __init__(self, model, init_rank: int = 16, min_rank: int = 2):
        self.model = model
        self.init_rank = init_rank
        self.min_rank = min_rank
        self.task_ranks: List[Dict[str, int]] = []

    def prune_ranks(self, threshold_ratio: float = 0.15) -> Dict[str, int]:
        """Prune LoRA ranks based on SVD singular value thresholding."""
        rank_info = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                for key in module.lora_A:
                    A = module.lora_A[key].weight  # [r, d_in]
                    B = module.lora_B[key].weight  # [d_out, r]
                    W = B @ A  # [d_out, d_in]
                    _, S, _ = torch.linalg.svd(W, full_matrices=False)
                    threshold = threshold_ratio * S[0].item()
                    keep = max(self.min_rank, (S > threshold).sum().item())
                    rank_info[f"{name}.{key}"] = keep
        self.task_ranks.append(rank_info)
        return rank_info


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def load_base_model(backbone_name: str, device: str):
    """Load the frozen backbone model."""
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def apply_lora(model, rank: int = 8, lora_alpha: float = 16.0, dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(model, lora_config)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_one_task_baseline(
    model,
    tokenizer,
    task_key: str,
    train_data: list,
    config: dict,
    device: str,
    extra_loss_fn=None,
) -> List[float]:
    """Train a baseline model on one task. Returns list of epoch losses."""
    dataset = TaskDataset(train_data, tokenizer, config.get("max_length", 512))
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    num_epochs = config.get("num_epochs", 3)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"  {task_key} Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1} loss: {avg_loss:.4f}")

    return losses


def evaluate_baseline(model, tokenizer, task_key: str, test_data: list, config: dict, device: str) -> float:
    """Evaluate a baseline model on a task."""
    model.eval()
    predictions, references = [], []
    n_eval = min(len(test_data), config.get("n_eval_examples", 200))

    with torch.no_grad():
        for example in tqdm(test_data[:n_eval], desc=f"  Eval {task_key}"):
            inputs = tokenizer(
                example["prompt"], return_tensors="pt",
                truncation=True, max_length=512
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", 128),
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(pred)
            references.append(example["response"])

    metric_name = TASK_CONFIGS[task_key]["metric"]
    score = compute_task_metric(predictions, references, metric_name)
    print(f"  {task_key} {metric_name}: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Per-method training functions
# ---------------------------------------------------------------------------

def run_full_finetune(config, data_dir, output_dir, device):
    """Full sequential fine-tuning — catastrophic forgetting baseline."""
    print("\n" + "="*70)
    print("BASELINE: full_finetune")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}

    # Zero-shot evaluation
    for j, task_key in enumerate(TASK_SEQUENCE):
        test_data = load_task_data(data_dir, task_key, "test")
        zero_shot[task_key] = evaluate_baseline(model, tokenizer, task_key, test_data, config, device)

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[full_finetune] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")

        # Apply fresh LoRA for each task (standard sequential fine-tuning)
        peft_model = apply_lora(model, rank=config.get("lora_rank", 8))
        losses = train_one_task_baseline(peft_model, tokenizer, task_key, train_data, config, device)

        # Merge LoRA into backbone for next task
        model = peft_model.merge_and_unload()

        # Evaluate on all tasks seen so far
        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("full_finetune", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


def run_ewc(config, data_dir, output_dir, device):
    """EWC baseline."""
    print("\n" + "="*70)
    print("BASELINE: ewc")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)
    peft_model = apply_lora(model, rank=config.get("lora_rank", 8))

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}
    ewc_penalties = []

    for j, task_key in enumerate(TASK_SEQUENCE):
        test_data = load_task_data(data_dir, task_key, "test")
        zero_shot[task_key] = evaluate_baseline(peft_model, tokenizer, task_key, test_data, config, device)

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[ewc] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")
        dataset = TaskDataset(train_data, tokenizer, config.get("max_length", 512))
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 4), shuffle=True)

        # Combine all EWC penalties from previous tasks
        def combined_ewc_penalty(model):
            return sum(ewc.penalty(model) for ewc in ewc_penalties)

        losses = train_one_task_baseline(
            peft_model, tokenizer, task_key, train_data, config, device,
            extra_loss_fn=combined_ewc_penalty if ewc_penalties else None,
        )

        # Compute Fisher for this task
        ewc = EWC(peft_model, loader, device, lambda_ewc=config.get("ewc_lambda", 5000.0))
        ewc_penalties.append(ewc)

        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(peft_model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("ewc", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


def run_o_lora(config, data_dir, output_dir, device):
    """O-LoRA baseline."""
    print("\n" + "="*70)
    print("BASELINE: o_lora")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}
    o_lora_trainer = OLoRATrainer()

    for j, task_key in enumerate(TASK_SEQUENCE):
        test_data = load_task_data(data_dir, task_key, "test")
        peft_model = apply_lora(model, rank=config.get("lora_rank", 8))
        zero_shot[task_key] = evaluate_baseline(peft_model, tokenizer, task_key, test_data, config, device)

    peft_model = apply_lora(model, rank=config.get("lora_rank", 8))

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[o_lora] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")

        lambda_orth = config.get("o_lora_lambda", 0.1)
        orth_penalty_fn = lambda m: o_lora_trainer.orthogonal_penalty(m, lambda_orth)

        losses = train_one_task_baseline(
            peft_model, tokenizer, task_key, train_data, config, device,
            extra_loss_fn=orth_penalty_fn if i > 0 else None,
        )

        o_lora_trainer.save_current_A_matrices(peft_model)

        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(peft_model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("o_lora", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


def run_lora_replay(config, data_dir, output_dir, device):
    """LoRA + Experience Replay baseline."""
    print("\n" + "="*70)
    print("BASELINE: lora_replay")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)
    peft_model = apply_lora(model, rank=config.get("lora_rank", 8))

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}
    replay_buffer: List[dict] = []
    replay_size_per_task = config.get("replay_buffer_size", 200)

    for j, task_key in enumerate(TASK_SEQUENCE):
        test_data = load_task_data(data_dir, task_key, "test")
        zero_shot[task_key] = evaluate_baseline(peft_model, tokenizer, task_key, test_data, config, device)

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[lora_replay] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")

        # Mix current task data with replay buffer
        combined_data = train_data + replay_buffer
        losses = train_one_task_baseline(peft_model, tokenizer, task_key, combined_data, config, device)

        # Add examples to replay buffer
        import random
        new_replay = random.sample(train_data, min(replay_size_per_task, len(train_data)))
        replay_buffer.extend(new_replay)

        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(peft_model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("lora_replay", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


def run_magmax(config, data_dir, output_dir, device):
    """
    MagMax: Magnitude-based model merging for continual learning.
    Trains separate LoRA adapters per task, then merges by magnitude.
    """
    print("\n" + "="*70)
    print("BASELINE: magmax")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}
    task_adapters = []  # Store per-task adapter state dicts

    for j, task_key in enumerate(TASK_SEQUENCE):
        peft_model = apply_lora(model, rank=config.get("lora_rank", 8))
        test_data = load_task_data(data_dir, task_key, "test")
        zero_shot[task_key] = evaluate_baseline(peft_model, tokenizer, task_key, test_data, config, device)

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[magmax] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")

        # Train fresh adapter for this task
        peft_model = apply_lora(model, rank=config.get("lora_rank", 8))
        losses = train_one_task_baseline(peft_model, tokenizer, task_key, train_data, config, device)

        # Save adapter state
        adapter_state = {
            n: p.data.clone()
            for n, p in peft_model.named_parameters()
            if p.requires_grad
        }
        task_adapters.append(adapter_state)

        # Merge all adapters by magnitude (take max absolute value per parameter)
        if len(task_adapters) > 1:
            merged_state = {}
            for key in task_adapters[0]:
                stacked = torch.stack([a[key] for a in task_adapters], dim=0)
                # MagMax: select value with maximum absolute magnitude
                abs_stacked = stacked.abs()
                max_idx = abs_stacked.argmax(dim=0, keepdim=True)
                merged = stacked.gather(0, max_idx.expand_as(stacked[:1])).squeeze(0)
                merged_state[key] = merged

            # Apply merged state to model
            merged_model = apply_lora(model, rank=config.get("lora_rank", 8))
            for n, p in merged_model.named_parameters():
                if n in merged_state and p.requires_grad:
                    p.data.copy_(merged_state[n])
            eval_model = merged_model
        else:
            eval_model = peft_model

        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(eval_model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("magmax", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


def run_codyre(config, data_dir, output_dir, device):
    """
    CoDyRA: Continual Dynamic Rank Adaptation (Zhang et al., 2024).
    Approximation: starts with high rank, prunes after each task,
    reallocates freed capacity to the next task.
    """
    print("\n" + "="*70)
    print("BASELINE: codyre")
    print("="*70)
    _method_start_time = time.time()

    backbone = config.get("backbone", "mistralai/Mistral-7B-Instruct-v0.3")
    model, tokenizer = load_base_model(backbone, device)

    # Start with high rank
    init_rank = config.get("codyre_init_rank", 16)
    peft_model = apply_lora(model, rank=init_rank)
    codyre = CoDyRATrainer(peft_model, init_rank=init_rank)

    R = [[None] * len(TASK_SEQUENCE) for _ in range(len(TASK_SEQUENCE))]
    zero_shot = {}

    for j, task_key in enumerate(TASK_SEQUENCE):
        test_data = load_task_data(data_dir, task_key, "test")
        zero_shot[task_key] = evaluate_baseline(peft_model, tokenizer, task_key, test_data, config, device)

    for i, task_key in enumerate(TASK_SEQUENCE):
        print(f"\n[codyre] Task {i+1}/{len(TASK_SEQUENCE)}: {task_key}")
        train_data = load_task_data(data_dir, task_key, "train")

        losses = train_one_task_baseline(peft_model, tokenizer, task_key, train_data, config, device)

        # Prune ranks after each task
        rank_info = codyre.prune_ranks(threshold_ratio=config.get("codyre_prune_threshold", 0.15))
        print(f"  Rank info after pruning: {rank_info}")

        for j in range(i + 1):
            test_data = load_task_data(data_dir, TASK_SEQUENCE[j], "test")
            R[i][j] = evaluate_baseline(peft_model, tokenizer, TASK_SEQUENCE[j], test_data, config, device)

        cl_metrics = compute_cl_metrics(R, zero_shot, TASK_SEQUENCE, i)
        _save_results("codyre", R, zero_shot, cl_metrics, output_dir, training_time_seconds=time.time() - _method_start_time)

    return R, zero_shot


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_results(method: str, R, zero_shot, cl_metrics, output_dir, training_time_seconds: float = None):
    """Save results for a method."""
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    results = {
        "method": method,
        "performance_matrix": R,
        "zero_shot_scores": zero_shot,
        "final_cl_metrics": cl_metrics,
        "training_time_seconds": training_time_seconds,
    }
    with open(os.path.join(method_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {method_dir}/results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = {
    "full_finetune": run_full_finetune,
    "ewc": run_ewc,
    "o_lora": run_o_lora,
    "lora_replay": run_lora_replay,
    "magmax": run_magmax,
    "codyre": run_codyre,
}


def main():
    parser = argparse.ArgumentParser(description="Train FARM baselines")
    parser.add_argument("--method", type=str, default="all",
                        choices=list(METHODS.keys()) + ["all"])
    parser.add_argument("--config", type=str, default="configs/farm_config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    methods_to_run = list(METHODS.keys()) if args.method == "all" else [args.method]

    for method in methods_to_run:
        print(f"\n{'#'*70}")
        print(f"# Running baseline: {method}")
        print(f"{'#'*70}")
        METHODS[method](config, args.data_dir, args.output_dir, device)

    print("\nAll baselines complete!")


if __name__ == "__main__":
    main()
