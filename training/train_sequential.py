"""
train_sequential.py
-------------------
Main sequential training loop for AdaLoRA-MoE.

This script trains the model on 6 tasks sequentially, saving checkpoints
and evaluating after each task. It implements the full AdaLoRA-MoE training
procedure including:
  - Adapter allocation via AdapterBank
  - Task loss + Alignment Anchor Loss
  - Routing network updates
  - Periodic adapter consolidation

Usage:
    python training/train_sequential.py --config configs/adalora_moe_config.yaml
"""

import os
import json
import argparse
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from models.adalora_moe import AdaLoRAMoE
from data.task_configs import TASK_SEQUENCE, load_task_data, load_alignment_probe


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class TaskDataset(Dataset):
    """Simple dataset for a single task."""

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

        # Labels: -100 for prompt tokens (don't compute loss on prompt)
        prompt_encoding = self.tokenizer(
            example["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_one_task(
    model: AdaLoRAMoE,
    task_key: str,
    task_config: dict,
    train_data: list,
    val_data: list,
    align_data: list,
    config: dict,
    device: str,
) -> dict:
    """
    Train the model on a single task and return evaluation metrics.

    Returns:
        metrics: dict with train/val losses and evaluation scores
    """
    print(f"\n{'='*70}")
    print(f"Training on task: {task_key}")
    print(f"Description: {task_config['description']}")
    print(f"{'='*70}")

    # Prepare model for this task
    active_model = model.prepare_for_task(
        task_name=task_key,
        task_description=task_config["description"],
    )
    active_model.train()

    # Datasets and loaders
    train_dataset = TaskDataset(train_data, model.tokenizer, config["max_length"])
    align_dataset = TaskDataset(align_data, model.tokenizer, config["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    align_loader = DataLoader(
        align_dataset,
        batch_size=config["align_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    align_iter = iter(align_loader)

    # Optimizer and scheduler
    optimizer = AdamW(
        [p for p in active_model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_val_loss = float("inf")
    metrics_history = []

    for epoch in range(config["num_epochs"]):
        epoch_task_loss = 0.0
        epoch_align_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config['num_epochs']}")

        for batch in progress_bar:
            # Move task batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get alignment probe batch
            try:
                align_batch = next(align_iter)
            except StopIteration:
                align_iter = iter(align_loader)
                align_batch = next(align_iter)

            align_input_ids = align_batch["input_ids"].to(device)
            align_attention_mask = align_batch["attention_mask"].to(device)

            # Forward pass — task loss
            outputs = active_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            task_loss = outputs.loss

            # Update alignment lambda
            model.alignment_loss_fn.update_lambda(global_step, total_steps)

            # Combined loss
            total_loss, loss_components = model.total_loss_fn(
                task_loss=task_loss,
                current_model=active_model,
                align_input_ids=align_input_ids,
                align_attention_mask=align_attention_mask,
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in active_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_task_loss += loss_components["task_loss"]
            epoch_align_loss += loss_components["align_loss"]
            epoch_total_loss += loss_components["total_loss"]
            n_batches += 1
            global_step += 1

            progress_bar.set_postfix({
                "task_loss": f"{loss_components['task_loss']:.4f}",
                "align_loss": f"{loss_components['align_loss']:.4f}",
                "λ": f"{loss_components['lambda_align']:.3f}",
            })

            if config.get("use_wandb") and global_step % 10 == 0:
                wandb.log({
                    f"{task_key}/task_loss": loss_components["task_loss"],
                    f"{task_key}/align_loss": loss_components["align_loss"],
                    f"{task_key}/total_loss": loss_components["total_loss"],
                    f"{task_key}/lambda_align": loss_components["lambda_align"],
                    "global_step": global_step,
                })

        avg_task_loss = epoch_task_loss / n_batches
        avg_align_loss = epoch_align_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches

        print(f"  Epoch {epoch+1} | task_loss={avg_task_loss:.4f} | "
              f"align_loss={avg_align_loss:.4f} | total={avg_total_loss:.4f}")

        metrics_history.append({
            "epoch": epoch + 1,
            "task_loss": avg_task_loss,
            "align_loss": avg_align_loss,
            "total_loss": avg_total_loss,
        })

    # Save the trained adapter
    model.save_current_task()

    return {
        "task_key": task_key,
        "epochs": metrics_history,
        "final_task_loss": metrics_history[-1]["task_loss"],
        "final_align_loss": metrics_history[-1]["align_loss"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/adalora_moe_config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/adalora_moe")
    parser.add_argument("--resume_from_task", type=int, default=0,
                        help="Resume training from this task index (0-based)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize W&B
    if config.get("use_wandb"):
        wandb.init(
            project="AdaLoRA-MoE",
            name=f"adalora_moe_{config.get('run_name', 'run')}",
            config=config,
        )

    # Initialize model
    model = AdaLoRAMoE(
        backbone_name=config["backbone_name"],
        save_dir=args.output_dir,
        similarity_threshold=config["similarity_threshold"],
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        lambda_align_start=config["lambda_align_start"],
        lambda_align_end=config["lambda_align_end"],
        consolidate_every=config["consolidate_every"],
    )

    # Load alignment probe
    align_data = load_alignment_probe(args.data_dir)
    print(f"Loaded {len(align_data)} alignment probe examples.")

    # Sequential training loop
    all_results = []
    for task_idx, task_key in enumerate(TASK_SEQUENCE):
        if task_idx < args.resume_from_task:
            print(f"Skipping task {task_idx}: {task_key} (resuming from task {args.resume_from_task})")
            continue

        from data.task_configs import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_key]

        train_data = load_task_data(args.data_dir, task_key, "train")
        val_data = load_task_data(args.data_dir, task_key, "val")

        result = train_one_task(
            model=model,
            task_key=task_key,
            task_config=task_config,
            train_data=train_data,
            val_data=val_data,
            align_data=align_data,
            config=config,
            device=device,
        )
        all_results.append(result)

        # Save intermediate results
        results_path = os.path.join(args.output_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {results_path}")

    print("\n✅ Sequential training complete!")
    print(f"All results saved to {args.output_dir}/training_results.json")

    if config.get("use_wandb"):
        wandb.finish()


if __name__ == "__main__":
    main()
