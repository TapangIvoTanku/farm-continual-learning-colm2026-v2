"""
adapter_bank.py
---------------
Dynamic LoRA Adapter Bank with Task-Aware Allocation.

Core innovation: Instead of training a single LoRA adapter for all tasks,
AdaLoRA-MoE maintains a bank of specialized adapters. When a new task arrives,
the bank checks semantic similarity to previous tasks:
  - If similarity > threshold: reuse the most similar adapter (positive transfer)
  - If similarity <= threshold: spawn a new adapter (prevent interference)

This is the first component of AdaLoRA-MoE.
"""

import os
import json
import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass, field, asdict
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import PreTrainedModel


@dataclass
class AdapterEntry:
    """Metadata for a single adapter in the bank."""
    adapter_id: int
    task_name: str
    task_embedding: list          # Sentence embedding of the task description
    adapter_path: str             # Path to saved adapter weights
    task_index: int               # Sequential index of the task (0-based)
    routing_count: int = 0        # How many times this adapter has been selected
    merged_from: list = field(default_factory=list)  # IDs of adapters merged into this one


class AdapterBank:
    """
    Dynamic bank of LoRA adapters with semantic task-aware allocation.

    Usage:
        bank = AdapterBank(similarity_threshold=0.75, save_dir="./adapter_bank")
        adapter_id = bank.allocate_adapter(task_name="news_summarization",
                                            task_description="Summarize news articles into one sentence.")
        # ... fine-tune the selected adapter ...
        bank.save_adapter(adapter_id, model)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        save_dir: str = "./adapter_bank",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.save_dir = save_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj"]
        self.adapters: list[AdapterEntry] = []
        self.embedding_model = SentenceTransformer(embedding_model)

        os.makedirs(save_dir, exist_ok=True)
        self._load_bank_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate_adapter(self, task_name: str, task_description: str) -> tuple[int, bool]:
        """
        Allocate an adapter for a new task.

        Returns:
            (adapter_id, is_new): adapter_id is the index in self.adapters,
                                   is_new=True if a new adapter was created.
        """
        task_embedding = self._embed(task_description)

        if len(self.adapters) == 0:
            return self._create_new_adapter(task_name, task_description, task_embedding), True

        similarities = self._compute_similarities(task_embedding)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.similarity_threshold:
            # Reuse existing adapter — positive forward transfer
            print(f"[AdapterBank] Task '{task_name}' is similar to "
                  f"'{self.adapters[best_idx].task_name}' (sim={best_sim:.3f}). "
                  f"Reusing adapter {best_idx}.")
            self.adapters[best_idx].routing_count += 1
            self._save_bank_state()
            return best_idx, False
        else:
            # Create new adapter — prevent interference
            print(f"[AdapterBank] Task '{task_name}' is novel "
                  f"(max_sim={best_sim:.3f} < {self.similarity_threshold}). "
                  f"Creating new adapter.")
            return self._create_new_adapter(task_name, task_description, task_embedding), True

    def get_lora_config(self) -> LoraConfig:
        """Return a LoraConfig for creating a new adapter."""
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def save_adapter(self, adapter_id: int, model: PeftModel) -> None:
        """Save a fine-tuned adapter to disk."""
        path = self.adapters[adapter_id].adapter_path
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        print(f"[AdapterBank] Saved adapter {adapter_id} to {path}")
        self._save_bank_state()

    def load_adapter(self, adapter_id: int, base_model: PreTrainedModel) -> PeftModel:
        """Load a saved adapter onto a base model."""
        path = self.adapters[adapter_id].adapter_path
        model = PeftModel.from_pretrained(base_model, path)
        print(f"[AdapterBank] Loaded adapter {adapter_id} from {path}")
        return model

    def get_all_adapter_paths(self) -> list[str]:
        """Return paths of all saved adapters."""
        return [a.adapter_path for a in self.adapters]

    def get_task_embeddings_tensor(self) -> torch.Tensor:
        """Return all task embeddings as a tensor for the routing network."""
        if not self.adapters:
            return torch.empty(0)
        embeddings = [torch.tensor(a.task_embedding) for a in self.adapters]
        return torch.stack(embeddings)

    def consolidate(self, merge_threshold: float = 0.85) -> int:
        """
        Merge adapters with cosine similarity above merge_threshold.
        Returns the number of merges performed.
        See consolidate_adapters.py for the full merging logic.
        """
        n_merges = 0
        embeddings = np.array([a.task_embedding for a in self.adapters])
        merged = set()

        for i in range(len(self.adapters)):
            if i in merged:
                continue
            for j in range(i + 1, len(self.adapters)):
                if j in merged:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]) /
                            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8))
                if sim >= merge_threshold:
                    print(f"[AdapterBank] Merging adapter {j} into {i} (sim={sim:.3f})")
                    self.adapters[i].merged_from.append(j)
                    merged.add(j)
                    n_merges += 1

        # Remove merged adapters (keep only non-merged ones)
        self.adapters = [a for idx, a in enumerate(self.adapters) if idx not in merged]
        self._save_bank_state()
        print(f"[AdapterBank] Consolidation complete. {n_merges} merges. "
              f"Bank size: {len(self.adapters)}")
        return n_merges

    def __len__(self) -> int:
        return len(self.adapters)

    def __repr__(self) -> str:
        lines = [f"AdapterBank (size={len(self.adapters)}, threshold={self.similarity_threshold})"]
        for i, a in enumerate(self.adapters):
            lines.append(f"  [{i}] {a.task_name} | task_idx={a.task_index} | "
                         f"routed={a.routing_count}x")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list:
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _compute_similarities(self, query_embedding: list) -> np.ndarray:
        q = np.array(query_embedding)
        bank_embeddings = np.array([a.task_embedding for a in self.adapters])
        # Cosine similarity (embeddings are already normalized)
        similarities = bank_embeddings @ q
        return similarities

    def _create_new_adapter(self, task_name: str, task_description: str,
                             task_embedding: list) -> int:
        adapter_id = len(self.adapters)
        adapter_path = os.path.join(self.save_dir, f"adapter_{adapter_id}_{task_name}")
        entry = AdapterEntry(
            adapter_id=adapter_id,
            task_name=task_name,
            task_embedding=task_embedding,
            adapter_path=adapter_path,
            task_index=adapter_id,
            routing_count=1,
        )
        self.adapters.append(entry)
        self._save_bank_state()
        return adapter_id

    def _save_bank_state(self) -> None:
        state_path = os.path.join(self.save_dir, "bank_state.json")
        state = [asdict(a) for a in self.adapters]
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_bank_state(self) -> None:
        state_path = os.path.join(self.save_dir, "bank_state.json")
        if not os.path.exists(state_path):
            return
        with open(state_path, "r") as f:
            state = json.load(f)
        self.adapters = [AdapterEntry(**a) for a in state]
        print(f"[AdapterBank] Loaded {len(self.adapters)} adapters from {state_path}")
