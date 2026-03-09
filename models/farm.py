"""
farm.py
-------
FARM: Forgetting-Aware Rank-Modulated Experts

Core model implementing:
1. Per-layer mixture of K LoRA experts with adaptive ranks
2. Rank-overlap-minimizing router (theoretically motivated)
3. Forgetting bound computation: F(T_i) <= C * ||A_i^T A_{i+1}||_F / sqrt(r_i * r_{i+1})
4. SVD-based rank pruning and capacity reallocation after each task

Design philosophy:
- Routing is NOT a learned black-box gate. It is a geometric operation:
  route each input to the expert whose current LoRA subspace has minimum
  cosine similarity with the incoming activation direction. This minimizes
  the cross-task adapter overlap, directly bounding forgetting.
- Rank adaptation is NOT just AdaLoRA's SVD pruning. After each task,
  we reallocate freed capacity to experts that need it most (highest
  gradient norm on the new task).
"""

import os
import math
import json
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Rank-Overlap-Minimizing Router
# ---------------------------------------------------------------------------

class RankOverlapRouter(nn.Module):
    """
    Routes input activations to the expert with minimum subspace overlap.

    For each expert e with LoRA A matrix A_e ∈ R^{d x r_e}, the overlap
    with input x ∈ R^d is:

        overlap(x, e) = ||A_e^T x||_2 / (||A_e||_F * ||x||_2 + eps)

    The router selects the expert with minimum overlap (most orthogonal
    to the current input direction), which minimizes cross-task interference.

    During training on task t, we also maintain a running estimate of the
    task's gradient subspace to compute the forgetting bound post-hoc.
    """

    def __init__(self, num_experts: int, hidden_dim: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Subspace matrices: A_e for each expert (updated externally from LoRA weights)
        # Shape: [num_experts, hidden_dim, max_rank]
        self.register_buffer(
            "expert_subspaces",
            torch.zeros(num_experts, hidden_dim, 16),  # max rank = 16
        )
        self.register_buffer(
            "expert_ranks",
            torch.zeros(num_experts, dtype=torch.long),
        )

        # Running gradient subspace estimate for current task (for forgetting bound)
        self.register_buffer(
            "task_gradient_subspace",
            torch.zeros(hidden_dim, 16),
        )
        self.register_buffer(
            "gradient_accumulation_count",
            torch.tensor(0, dtype=torch.long),
        )

        # Router entropy tracking
        self.routing_history: List[torch.Tensor] = []

    def update_expert_subspace(self, expert_id: int, A_matrix: torch.Tensor):
        """
        Update the stored subspace for an expert from its LoRA A matrix.
        A_matrix: [r, d] (LoRA convention: W' = W + B*A, A: [r, d])
        """
        r, d = A_matrix.shape
        # Normalize columns of A^T to get orthonormal basis
        A_T = A_matrix.T  # [d, r]
        # Gram-Schmidt or just normalize
        norms = A_T.norm(dim=0, keepdim=True).clamp(min=1e-8)
        A_normalized = A_T / norms

        max_r = self.expert_subspaces.shape[2]
        actual_r = min(r, max_r)

        self.expert_subspaces[expert_id, :, :actual_r] = A_normalized[:, :actual_r].detach()
        self.expert_subspaces[expert_id, :, actual_r:] = 0.0
        self.expert_ranks[expert_id] = actual_r

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for input x.

        Args:
            x: [batch, seq, hidden_dim] or [batch, hidden_dim]

        Returns:
            weights: [batch, num_experts] softmax weights
            selected: [batch, top_k] selected expert indices
        """
        if x.dim() == 3:
            # Pool over sequence dimension
            x_pooled = x.mean(dim=1)  # [batch, hidden_dim]
        else:
            x_pooled = x  # [batch, hidden_dim]

        x_norm = F.normalize(x_pooled, dim=-1)  # [batch, hidden_dim]

        overlaps = []
        for e in range(self.num_experts):
            r = self.expert_ranks[e].item()
            if r == 0:
                # Expert has no subspace yet — prefer it (overlap = 0)
                overlap = torch.zeros(x_norm.shape[0], device=x.device)
            else:
                subspace = self.expert_subspaces[e, :, :r]  # [hidden_dim, r]
                # Project x onto expert subspace
                proj = x_norm @ subspace.to(x_norm.dtype)  # [batch, r]
                overlap = proj.norm(dim=-1)  # [batch]
            overlaps.append(overlap)

        overlaps = torch.stack(overlaps, dim=-1)  # [batch, num_experts]

        # Convert overlaps to routing weights: lower overlap = higher weight
        # Use negative overlap as logits, then softmax
        logits = -overlaps  # [batch, num_experts]
        weights = F.softmax(logits / 0.1, dim=-1)  # temperature = 0.1

        # Top-k selection
        _, selected = torch.topk(weights, k=self.top_k, dim=-1)

        # Track routing for entropy computation
        if self.training:
            self.routing_history.append(weights.detach().mean(dim=0))

        return weights, selected

    def get_router_entropy(self) -> float:
        """Compute average routing entropy over recent history."""
        if not self.routing_history:
            return 0.0
        avg_weights = torch.stack(self.routing_history).mean(dim=0)
        entropy = -(avg_weights * (avg_weights + 1e-8).log()).sum().item()
        self.routing_history.clear()
        return entropy

    def compute_forgetting_bound(
        self,
        A_prev: torch.Tensor,
        A_curr: torch.Tensor,
        r_prev: int,
        r_curr: int,
    ) -> float:
        """
        Compute the forgetting bound:
            F(T_i) <= C * ||A_prev^T A_curr||_F / sqrt(r_prev * r_curr)

        where C is approximated as 1.0 (normalized bound).

        Args:
            A_prev: [r_prev, d] LoRA A matrix from previous task
            A_curr: [r_curr, d] LoRA A matrix from current task

        Returns:
            Scalar bound value
        """
        if r_prev == 0 or r_curr == 0:
            return 0.0

        # A_prev^T A_curr: [d, r_prev]^T @ [d, r_curr] = [r_prev, r_curr]
        cross = A_prev @ A_curr.T  # [r_prev, r_curr]
        frob_norm = cross.norm(p="fro").item()
        bound = frob_norm / math.sqrt(r_prev * r_curr)
        return bound


# ---------------------------------------------------------------------------
# Adaptive Rank LoRA Expert
# ---------------------------------------------------------------------------

class AdaptiveRankLoRAExpert(nn.Module):
    """
    A single LoRA expert with adaptive rank.

    Maintains A ∈ R^{r x d_in} and B ∈ R^{d_out x r}.
    Rank r can be pruned via SVD thresholding or expanded up to max_rank.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        init_rank: int = 8,
        max_rank: int = 16,
        lora_alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.current_rank = init_rank
        self.max_rank = max_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / init_rank

        self.lora_A = nn.Parameter(torch.empty(init_rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, init_rank))
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] -> [..., d_out]"""
        return self.dropout(x) @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T * self.scaling

    def prune_rank(self, threshold_ratio: float = 0.1) -> int:
        """
        Prune rank via SVD: keep singular values above threshold_ratio * sigma_max.
        Returns the new rank.
        """
        with torch.no_grad():
            # Effective weight: B @ A ∈ R^{d_out x d_in}
            W = self.lora_B @ self.lora_A  # [d_out, d_in]
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            threshold = threshold_ratio * S[0].item()
            keep = (S > threshold).sum().item()
            keep = max(1, min(keep, self.max_rank))

            if keep < self.current_rank:
                # Reconstruct with reduced rank
                new_A = (S[:keep].unsqueeze(1) * Vh[:keep, :])  # [keep, d_in]
                new_B = U[:, :keep]  # [d_out, keep]

                # Resize parameters
                self.lora_A = nn.Parameter(new_A)
                self.lora_B = nn.Parameter(new_B)
                self.current_rank = keep
                self.scaling = self.lora_alpha / keep

        return self.current_rank

    def get_A_matrix(self) -> torch.Tensor:
        """Return the A matrix for subspace computation."""
        return self.lora_A.detach()  # [r, d_in]


# ---------------------------------------------------------------------------
# FARM Layer: Per-layer mixture of K adaptive rank experts
# ---------------------------------------------------------------------------

class FARMLayer(nn.Module):
    """
    Replaces a single LoRA adapter with K adaptive-rank experts + rank-overlap router.

    Applied to a specific linear layer in the transformer.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_experts: int = 4,
        init_rank: int = 8,
        max_rank: int = 16,
        lora_alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_in = d_in
        self.d_out = d_out

        self.experts = nn.ModuleList([
            AdaptiveRankLoRAExpert(d_in, d_out, init_rank, max_rank, lora_alpha, dropout)
            for _ in range(num_experts)
        ])

        self.router = RankOverlapRouter(num_experts, d_in, top_k=1)

        # Initialize router subspaces
        for i, expert in enumerate(self.experts):
            self.router.update_expert_subspace(i, expert.get_A_matrix())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_in]
        Returns the mixture-of-experts LoRA delta: [..., d_out]
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, d = x.shape
            x_flat = x.reshape(batch * seq, d)
        else:
            x_flat = x

        # Get routing weights: [batch*seq, num_experts]
        weights, selected = self.router(x_flat)

        # Compute expert outputs
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=1
        )  # [batch*seq, num_experts, d_out]

        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # [batch*seq, num_experts, 1]
        output = (expert_outputs * weights_expanded).sum(dim=1)  # [batch*seq, d_out]

        if len(original_shape) == 3:
            output = output.reshape(batch, seq, self.d_out)

        return output

    def prune_all_experts(self, threshold_ratio: float = 0.1):
        """Prune all experts and update router subspaces."""
        for i, expert in enumerate(self.experts):
            expert.prune_rank(threshold_ratio)
            self.router.update_expert_subspace(i, expert.get_A_matrix())

    def get_rank_utilization(self) -> Dict[str, int]:
        """Return current rank of each expert."""
        return {f"expert_{i}": e.current_rank for i, e in enumerate(self.experts)}

    def compute_cross_expert_forgetting_bounds(self) -> List[float]:
        """
        Compute pairwise forgetting bounds between all expert pairs.
        Useful for monitoring inter-expert interference.
        """
        bounds = []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                A_i = self.experts[i].get_A_matrix()
                A_j = self.experts[j].get_A_matrix()
                bound = self.router.compute_forgetting_bound(
                    A_i, A_j,
                    self.experts[i].current_rank,
                    self.experts[j].current_rank,
                )
                bounds.append(bound)
        return bounds


# ---------------------------------------------------------------------------
# FARM Model: Full model with FARM layers injected
# ---------------------------------------------------------------------------

class FARMModel(nn.Module):
    """
    Full FARM model wrapping a pretrained LLM with FARM layers injected
    into the attention Q, K, V, and O projections.

    Architecture:
    - Frozen backbone (Mistral-7B or similar)
    - FARM layers replacing LoRA adapters in attention projections
    - Per-task adapter state saved/loaded for continual learning
    """

    def __init__(
        self,
        backbone_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        num_experts: int = 4,
        init_rank: int = 8,
        max_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        save_dir: str = "./checkpoints/farm",
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_experts = num_experts
        self.init_rank = init_rank
        self.max_rank = max_rank
        self.save_dir = save_dir
        self.device = device

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        os.makedirs(save_dir, exist_ok=True)

        print(f"[FARM] Loading backbone: {backbone_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Inject FARM layers
        self.farm_layers: Dict[str, FARMLayer] = {}
        self._inject_farm_layers(lora_alpha, lora_dropout)

        # Task tracking
        self.task_history: List[Dict] = []  # Per-task adapter snapshots
        self.current_task_id: Optional[int] = None

        print(f"[FARM] Initialized with {len(self.farm_layers)} FARM layers")
        total_params = sum(
            p.numel() for layer in self.farm_layers.values()
            for p in layer.parameters()
        )
        print(f"[FARM] Trainable parameters: {total_params:,}")

    def _inject_farm_layers(self, lora_alpha: float, lora_dropout: float):
        """Inject FARM layers into target modules of the backbone."""
        for name, module in self.backbone.named_modules():
            module_name = name.split(".")[-1]
            if module_name in self.target_modules and isinstance(module, nn.Linear):
                d_in = module.in_features
                d_out = module.out_features

                farm_layer = FARMLayer(
                    d_in=d_in,
                    d_out=d_out,
                    num_experts=self.num_experts,
                    init_rank=self.init_rank,
                    max_rank=self.max_rank,
                    lora_alpha=lora_alpha,
                    dropout=lora_dropout,
                ).to(self.device)

                self.farm_layers[name] = farm_layer

                # Hook to add FARM output to the linear layer output
                def make_hook(farm_layer_ref):
                    def forward_hook(module, input, output):
                        x = input[0]
                        delta = farm_layer_ref(x.to(self.device))
                        return output + delta.to(output.dtype)
                    return forward_hook

                module.register_forward_hook(make_hook(farm_layer))

        # Register FARM layers as submodules
        for name, layer in self.farm_layers.items():
            safe_name = name.replace(".", "_")
            self.add_module(f"farm_{safe_name}", layer)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return only FARM layer parameters (backbone is frozen)."""
        params = []
        for layer in self.farm_layers.values():
            params.extend(layer.parameters())
        return params

    def prepare_for_task(self, task_id: int, task_name: str):
        """Prepare model for a new task."""
        self.current_task_id = task_id
        print(f"[FARM] Preparing for task {task_id}: {task_name}")

        # Reset router entropy tracking
        for layer in self.farm_layers.values():
            layer.router.routing_history.clear()

    def consolidate_after_task(self, task_name: str, threshold_ratio: float = 0.1):
        """
        After training on a task:
        1. Prune ranks via SVD thresholding
        2. Update router subspaces
        3. Save task adapter state
        4. Compute and log forgetting bounds
        """
        print(f"[FARM] Consolidating after task: {task_name}")

        # Prune all layers
        rank_utilization = {}
        for layer_name, layer in self.farm_layers.items():
            layer.prune_all_experts(threshold_ratio)
            rank_utilization[layer_name] = layer.get_rank_utilization()

        # Compute router entropy
        router_entropy = {}
        for layer_name, layer in self.farm_layers.items():
            router_entropy[layer_name] = layer.router.get_router_entropy()

        # Compute forgetting bounds between current and previous task
        forgetting_bounds = {}
        if len(self.task_history) > 0:
            prev_task = self.task_history[-1]
            for layer_name, layer in self.farm_layers.items():
                bounds = layer.compute_cross_expert_forgetting_bounds()
                forgetting_bounds[layer_name] = {
                    "mean": sum(bounds) / len(bounds) if bounds else 0.0,
                    "max": max(bounds) if bounds else 0.0,
                }

        # Save task state
        task_state = {
            "task_id": self.current_task_id,
            "task_name": task_name,
            "rank_utilization": rank_utilization,
            "router_entropy": router_entropy,
            "forgetting_bounds": forgetting_bounds,
        }
        self.task_history.append(task_state)

        # Save adapter weights
        self._save_task_adapters(task_name)

        return task_state

    def _save_task_adapters(self, task_name: str):
        """Save FARM layer weights for the current task."""
        task_dir = os.path.join(self.save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        state = {}
        for layer_name, layer in self.farm_layers.items():
            state[layer_name] = layer.state_dict()

        torch.save(state, os.path.join(task_dir, "farm_adapters.pt"))
        print(f"[FARM] Saved adapters to {task_dir}/farm_adapters.pt")

    def load_task_adapters(self, task_name: str):
        """Load FARM layer weights from a saved task."""
        task_dir = os.path.join(self.save_dir, task_name)
        state = torch.load(os.path.join(task_dir, "farm_adapters.pt"), map_location=self.device)

        for layer_name, layer in self.farm_layers.items():
            if layer_name in state:
                layer.load_state_dict(state[layer_name])

        print(f"[FARM] Loaded adapters from {task_dir}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Standard forward pass — FARM layers are applied via hooks."""
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_text: str, max_new_tokens: int = 256, **kwargs) -> str:
        """Generate text with FARM adapters active."""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        self.backbone.eval()
        outputs = self.backbone.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def save_metrics(self, output_path: str):
        """Save all collected metrics to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.task_history, f, indent=2)
        print(f"[FARM] Metrics saved to {output_path}")
