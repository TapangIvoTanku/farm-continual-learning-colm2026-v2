"""
routing_network.py
------------------
Semantic Routing Network for AdaLoRA-MoE.

At inference time, given an input, the routing network computes a soft
mixture-of-experts weighting over all adapters in the bank. This allows
the model to blend knowledge from multiple adapters when inputs span
multiple domains.

Architecture: 2-layer MLP with load-balancing loss to prevent routing collapse.
Input: sentence embedding of the input text (384-dim from all-MiniLM-L6-v2)
Output: softmax weights over K adapters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RoutingNetwork(nn.Module):
    """
    Lightweight 2-layer MLP that routes inputs to adapters.

    Args:
        input_dim: Dimension of the sentence embedding (384 for MiniLM)
        hidden_dim: Hidden layer size
        num_adapters: Number of adapters in the bank (can grow dynamically)
        temperature: Softmax temperature (lower = sharper routing)
        load_balance_weight: Weight for the load-balancing auxiliary loss
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_adapters: int = 1,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_adapters = num_adapters
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight

        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_adapters)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]
        Returns:
            routing_weights: Softmax weights [batch_size, num_adapters]
        """
        h = self.fc1(x)
        h = self.layer_norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        routing_weights = F.softmax(logits / self.temperature, dim=-1)
        return routing_weights

    def expand_for_new_adapter(self):
        """
        Expand the output layer when a new adapter is added to the bank.
        Preserves existing weights and initializes new output neuron to zero.
        """
        old_weight = self.fc2.weight.data  # [num_adapters, hidden_dim]
        old_bias = self.fc2.bias.data      # [num_adapters]

        self.num_adapters += 1
        new_fc2 = nn.Linear(self.hidden_dim, self.num_adapters)
        nn.init.zeros_(new_fc2.weight)
        nn.init.zeros_(new_fc2.bias)

        # Copy old weights
        new_fc2.weight.data[:self.num_adapters - 1] = old_weight
        new_fc2.bias.data[:self.num_adapters - 1] = old_bias

        self.fc2 = new_fc2
        print(f"[RoutingNetwork] Expanded to {self.num_adapters} adapters.")

    def shrink_for_merged_adapter(self, removed_indices: list[int]):
        """
        Shrink the output layer after adapter consolidation.
        Removes output neurons corresponding to merged adapters.
        """
        keep_indices = [i for i in range(self.num_adapters) if i not in removed_indices]
        old_weight = self.fc2.weight.data
        old_bias = self.fc2.bias.data

        self.num_adapters = len(keep_indices)
        new_fc2 = nn.Linear(self.hidden_dim, self.num_adapters)
        new_fc2.weight.data = old_weight[keep_indices]
        new_fc2.bias.data = old_bias[keep_indices]
        self.fc2 = new_fc2
        print(f"[RoutingNetwork] Shrunk to {self.num_adapters} adapters after consolidation.")

    def load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary load-balancing loss to prevent routing collapse
        (all inputs being routed to a single adapter).

        Encourages uniform usage across all adapters by penalizing
        the variance of mean routing weights.

        Args:
            routing_weights: [batch_size, num_adapters]
        Returns:
            scalar loss
        """
        # Mean routing weight per adapter across the batch
        mean_weights = routing_weights.mean(dim=0)  # [num_adapters]
        # Ideal uniform distribution
        uniform = torch.ones_like(mean_weights) / self.num_adapters
        # L2 distance from uniform
        loss = F.mse_loss(mean_weights, uniform)
        return self.load_balance_weight * loss

    def get_top_k_adapters(self, x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k adapters and their weights for a given input.
        Useful for sparse routing (only activate top-k adapters).

        Returns:
            (top_k_weights, top_k_indices): both [batch_size, k]
        """
        weights = self.forward(x)
        top_k_weights, top_k_indices = torch.topk(weights, k=min(k, self.num_adapters), dim=-1)
        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_indices

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_adapters": self.num_adapters,
            "temperature": self.temperature,
            "load_balance_weight": self.load_balance_weight,
        }, path)
        print(f"[RoutingNetwork] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RoutingNetwork":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_adapters=checkpoint["num_adapters"],
            temperature=checkpoint["temperature"],
            load_balance_weight=checkpoint["load_balance_weight"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[RoutingNetwork] Loaded from {path}")
        return model
