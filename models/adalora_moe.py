"""
adalora_moe.py
--------------
Main AdaLoRA-MoE Model Wrapper.

This class ties together:
  1. AdapterBank  — dynamic LoRA adapter allocation
  2. RoutingNetwork — soft mixture-of-experts routing at inference
  3. AlignmentAnchorLoss — RLHF alignment preservation during training

It wraps a frozen pre-trained backbone (Mistral-7B-Instruct) and exposes
a clean interface for sequential task training and inference.
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer

from models.adapter_bank import AdapterBank
from models.routing_network import RoutingNetwork
from models.alignment_loss import AlignmentAnchorLoss, TotalLoss


class AdaLoRAMoE(nn.Module):
    """
    AdaLoRA-MoE: Adaptive Mixture-of-Expert LoRA Routing for
    Alignment-Preserving Continual Learning.

    Args:
        backbone_name: HuggingFace model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        save_dir: Directory for saving adapters and routing network
        similarity_threshold: Cosine similarity threshold for adapter reuse
        lora_rank: LoRA rank r
        lora_alpha: LoRA scaling alpha
        lambda_align_start: Initial alignment loss weight
        lambda_align_end: Final alignment loss weight
        consolidate_every: Consolidate adapters every N tasks
        device: torch device
    """

    def __init__(
        self,
        backbone_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        save_dir: str = "./checkpoints",
        similarity_threshold: float = 0.75,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lambda_align_start: float = 0.1,
        lambda_align_end: float = 1.0,
        consolidate_every: int = 3,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.save_dir = save_dir
        self.consolidate_every = consolidate_every
        self.task_count = 0
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AdaLoRAMoE] Using device: {self.device}")

        # ---- Load backbone (frozen) ----
        print(f"[AdaLoRAMoE] Loading backbone: {backbone_name}")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            backbone_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("[AdaLoRAMoE] Backbone frozen.")

        # ---- Adapter Bank ----
        self.adapter_bank = AdapterBank(
            similarity_threshold=similarity_threshold,
            save_dir=os.path.join(save_dir, "adapter_bank"),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # ---- Routing Network ----
        self.routing_network = RoutingNetwork(
            input_dim=384,   # all-MiniLM-L6-v2 output dim
            hidden_dim=256,
            num_adapters=max(1, len(self.adapter_bank)),
        )
        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---- Alignment Loss ----
        self.alignment_loss_fn = AlignmentAnchorLoss(
            reference_model=self.backbone,
            lambda_align_start=lambda_align_start,
            lambda_align_end=lambda_align_end,
        )
        self.total_loss_fn = TotalLoss(self.alignment_loss_fn)

        # ---- Current active model (set during training) ----
        self.active_model: Optional[PeftModel] = None
        self.active_adapter_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def prepare_for_task(
        self,
        task_name: str,
        task_description: str,
    ) -> PeftModel:
        """
        Prepare the model for training on a new task.
        Allocates or reuses an adapter from the bank.

        Returns:
            The PEFT model ready for fine-tuning.
        """
        adapter_id, is_new = self.adapter_bank.allocate_adapter(
            task_name=task_name,
            task_description=task_description,
        )
        self.active_adapter_id = adapter_id

        if is_new:
            # Create a fresh LoRA adapter on top of the frozen backbone
            lora_config = self.adapter_bank.get_lora_config()
            self.active_model = get_peft_model(self.backbone, lora_config)
            # Expand routing network for new adapter
            if self.task_count > 0:
                self.routing_network.expand_for_new_adapter()
        else:
            # Load the existing adapter for fine-tuning (positive transfer)
            self.active_model = self.adapter_bank.load_adapter(
                adapter_id, self.backbone
            )

        self.active_model.print_trainable_parameters()
        self.task_count += 1

        # Trigger consolidation if needed
        if self.task_count % self.consolidate_every == 0:
            print(f"[AdaLoRAMoE] Triggering adapter consolidation after task {self.task_count}.")
            self.adapter_bank.consolidate(merge_threshold=0.85)

        return self.active_model

    def save_current_task(self):
        """Save the currently active adapter after training."""
        if self.active_model is None or self.active_adapter_id is None:
            raise RuntimeError("No active model to save. Call prepare_for_task() first.")
        self.adapter_bank.save_adapter(self.active_adapter_id, self.active_model)
        # Save routing network
        routing_path = os.path.join(self.save_dir, "routing_network.pt")
        self.routing_network.save(routing_path)

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        use_routing: bool = True,
        **generate_kwargs,
    ) -> str:
        """
        Generate text using the mixture-of-experts routing.

        If use_routing=True, uses the routing network to select the best adapter.
        If use_routing=False, uses the active adapter directly (for task-specific eval).
        """
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        if use_routing and len(self.adapter_bank) > 1:
            # Encode input for routing
            embedding = self.sentence_encoder.encode(
                input_text, normalize_embeddings=True
            )
            embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(self.device)
            _, top_adapter_idx = self.routing_network.get_top_k_adapters(
                embedding_tensor, k=1
            )
            best_adapter_id = int(top_adapter_idx[0, 0].item())

            # Load best adapter
            model = self.adapter_bank.load_adapter(best_adapter_id, self.backbone)
        else:
            model = self.active_model or self.backbone

        model.eval()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        # Decode only the newly generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_alignment_score(self, probe_batch: dict) -> float:
        """Evaluate alignment score on a probe batch."""
        return self.alignment_loss_fn.get_alignment_score(
            current_model=self.active_model or self.backbone,
            input_ids=probe_batch["input_ids"].to(self.device),
            attention_mask=probe_batch["attention_mask"].to(self.device),
        )

    def __repr__(self) -> str:
        return (
            f"AdaLoRAMoE(\n"
            f"  backbone={self.backbone_name},\n"
            f"  tasks_seen={self.task_count},\n"
            f"  {repr(self.adapter_bank)}\n"
            f")"
        )
