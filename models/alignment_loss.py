"""
alignment_loss.py
-----------------
Alignment Anchor Loss for AdaLoRA-MoE.

Core innovation: During fine-tuning on any new task, this loss penalizes
deviation from the frozen RLHF-aligned reference model's output distribution.
This prevents the "alignment tax" — the degradation of safety/helpfulness
that occurs during continual fine-tuning.

The loss is a KL divergence between the reference model and the current model
on a small, fixed alignment probe dataset (500 Alpaca examples).

The weight lambda_align is annealed linearly from 0.1 to 1.0 during training,
ensuring the model first learns the new task and then consolidates alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel


class AlignmentAnchorLoss(nn.Module):
    """
    KL-divergence loss between the current model and a frozen reference model.

    This is computed on a fixed alignment probe dataset, not the task data.
    The reference model is the original RLHF-aligned backbone (frozen).

    Args:
        reference_model: Frozen RLHF-aligned reference model (e.g., Mistral-7B-Instruct)
        lambda_align_start: Initial weight for the alignment loss
        lambda_align_end: Final weight for the alignment loss (after annealing)
        temperature: Temperature for softening the distributions
    """

    def __init__(
        self,
        reference_model: PreTrainedModel,
        lambda_align_start: float = 0.1,
        lambda_align_end: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.reference_model = reference_model
        self.lambda_align_start = lambda_align_start
        self.lambda_align_end = lambda_align_end
        self.temperature = temperature

        # Freeze the reference model completely
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        # Current lambda (updated by the trainer)
        self.current_lambda = lambda_align_start

    def forward(
        self,
        current_model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the alignment anchor loss.

        Args:
            current_model: The model being fine-tuned
            input_ids: Token IDs from the alignment probe batch [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Optional labels (not used for KL loss, but kept for API consistency)

        Returns:
            Weighted KL divergence loss (scalar)
        """
        # Get logits from the current (fine-tuned) model
        with torch.enable_grad():
            current_outputs = current_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            current_logits = current_outputs.logits  # [batch, seq_len, vocab_size]

        # Get logits from the frozen reference model
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits  # [batch, seq_len, vocab_size]

        # Apply temperature scaling
        current_log_probs = F.log_softmax(current_logits / self.temperature, dim=-1)
        ref_probs = F.softmax(ref_logits / self.temperature, dim=-1)

        # KL divergence: KL(ref || current) = sum(ref * (log_ref - log_current))
        # We use F.kl_div which expects log-probabilities for input and probabilities for target
        # Reduction over vocab dimension, then mean over sequence and batch
        kl_loss = F.kl_div(
            current_log_probs,
            ref_probs,
            reduction="batchmean",
            log_target=False,
        )

        return self.current_lambda * kl_loss

    def update_lambda(self, current_step: int, total_steps: int) -> float:
        """
        Linearly anneal lambda from lambda_align_start to lambda_align_end.

        Args:
            current_step: Current training step
            total_steps: Total training steps for this task

        Returns:
            Updated lambda value
        """
        progress = min(current_step / max(total_steps, 1), 1.0)
        self.current_lambda = (
            self.lambda_align_start +
            progress * (self.lambda_align_end - self.lambda_align_start)
        )
        return self.current_lambda

    def get_alignment_score(
        self,
        current_model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """
        Compute a normalized alignment score (0-1, higher = better aligned).
        Used for evaluation, not training.

        Returns:
            alignment_score: 1 - normalized_kl_divergence
        """
        with torch.no_grad():
            current_outputs = current_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_outputs = self.reference_model(input_ids=input_ids, attention_mask=attention_mask)

            current_probs = F.softmax(current_outputs.logits, dim=-1)
            ref_probs = F.softmax(ref_outputs.logits, dim=-1)

            # Jensen-Shannon divergence (symmetric, bounded [0,1])
            m = 0.5 * (current_probs + ref_probs)
            js_div = 0.5 * (
                F.kl_div(m.log(), current_probs, reduction="batchmean") +
                F.kl_div(m.log(), ref_probs, reduction="batchmean")
            )
            # JS divergence is in [0, log(2)], normalize to [0, 1]
            alignment_score = float(1.0 - js_div.item() / 0.693)
            return max(0.0, min(1.0, alignment_score))


class TotalLoss(nn.Module):
    """
    Combined loss for AdaLoRA-MoE training:
        L_total = L_task + L_align

    where:
        L_task  = standard cross-entropy loss on the task data
        L_align = alignment anchor KL loss on the alignment probe data
    """

    def __init__(self, alignment_loss: AlignmentAnchorLoss):
        super().__init__()
        self.alignment_loss = alignment_loss

    def forward(
        self,
        task_loss: torch.Tensor,
        current_model: PreTrainedModel,
        align_input_ids: torch.Tensor,
        align_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            task_loss: Cross-entropy loss from the task batch
            current_model: The model being fine-tuned
            align_input_ids: Input IDs from the alignment probe batch
            align_attention_mask: Attention mask from the alignment probe batch

        Returns:
            (total_loss, loss_components): total loss and dict of components for logging
        """
        align_loss = self.alignment_loss(
            current_model=current_model,
            input_ids=align_input_ids,
            attention_mask=align_attention_mask,
        )

        total = task_loss + align_loss

        components = {
            "task_loss": task_loss.item(),
            "align_loss": align_loss.item(),
            "total_loss": total.item(),
            "lambda_align": self.alignment_loss.current_lambda,
        }

        return total, components
