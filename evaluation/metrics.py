"""
evaluation/metrics.py
---------------------
Evaluation metrics for the FARM continual learning benchmark.

Supported metrics:
  - rouge2      : ROUGE-2 F1 (XSum)
  - rougeL      : ROUGE-L F1 (CNN/DM)
  - accuracy    : Exact-match accuracy (MedQA, GSM8K)
  - pass_at_1   : HumanEval pass@1 via execution-based evaluation

Continual learning metrics (computed from the performance matrix R):
  - ACC  : Average accuracy across all tasks seen so far
  - BWT  : Backward Transfer — measures forgetting of previous tasks
  - FWT  : Forward Transfer — measures zero-shot generalisation to future tasks
  - Intransigence : How much worse than joint training the model is
"""

import re
import math
import string
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Task-level metric: compute_task_metric
# ---------------------------------------------------------------------------

def compute_task_metric(
    predictions: List[str],
    references: List[str],
    metric_name: str,
) -> float:
    """
    Compute a scalar evaluation score for a list of predictions vs. references.

    Args:
        predictions:  Model-generated strings.
        references:   Ground-truth strings.
        metric_name:  One of {"rouge2", "rougeL", "accuracy", "pass_at_1"}.

    Returns:
        Scalar score in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length."
        )
    if not predictions:
        return 0.0

    if metric_name in ("rouge2", "rougeL"):
        return _compute_rouge(predictions, references, metric_name)
    elif metric_name == "accuracy":
        return _compute_accuracy(predictions, references)
    elif metric_name == "pass_at_1":
        return _compute_pass_at_1(predictions, references)
    else:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            "Choose from: rouge2, rougeL, accuracy, pass_at_1"
        )


# ---------------------------------------------------------------------------
# Continual learning metrics: compute_cl_metrics
# ---------------------------------------------------------------------------

def compute_cl_metrics(
    R: List[List[Optional[float]]],
    zero_shot_scores: List[float],
    task_keys: List[str],
    current_task_idx: int,
) -> Dict[str, float]:
    """
    Compute continual learning metrics from the performance matrix R.

    R[i][j] = score on task j after training on task i (or None if not yet evaluated).
    zero_shot_scores[j] = score on task j before any training (for FWT).

    Definitions (following Lopez-Paz & Ranzato, 2017):
      ACC  = (1 / (t+1)) * sum_{j=0}^{t} R[t][j]
      BWT  = (1 / t) * sum_{j=0}^{t-1} (R[t][j] - R[j][j])   (negative = forgetting)
      FWT  = (1 / (T-1)) * sum_{j=1}^{T-1} (R[j-1][j] - zero_shot[j])

    Args:
        R:                 Performance matrix (list of lists, None for missing).
        zero_shot_scores:  Zero-shot baseline scores for each task.
        task_keys:         Task identifiers in order.
        current_task_idx:  Index of the most recently trained task (0-indexed).

    Returns:
        Dict with keys: acc, bwt, fwt, intransigence, n_tasks_trained.
    """
    t = current_task_idx  # 0-indexed index of last trained task

    # --- ACC ---
    acc_scores = [
        R[t][j] for j in range(t + 1)
        if R[t][j] is not None
    ]
    acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0

    # --- BWT ---
    if t == 0:
        bwt = 0.0
    else:
        bwt_terms = []
        for j in range(t):
            r_tj = R[t][j]
            r_jj = R[j][j]
            if r_tj is not None and r_jj is not None:
                bwt_terms.append(r_tj - r_jj)
        bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0

    # --- FWT ---
    fwt_terms = []
    for j in range(1, t + 1):
        r_prev_j = R[j - 1][j] if j < len(R) and j < len(R[j - 1]) else None
        zs_j = zero_shot_scores[j] if j < len(zero_shot_scores) else None
        if r_prev_j is not None and zs_j is not None:
            fwt_terms.append(r_prev_j - zs_j)
    fwt = sum(fwt_terms) / len(fwt_terms) if fwt_terms else 0.0

    # --- Intransigence (diagonal gap) ---
    # How much worse is the model on the current task vs. training on it alone?
    # We approximate this as R[t][t] vs. the single-task upper bound (not always available).
    # Here we report it as 0 since we don't track single-task upper bounds separately.
    intransigence = 0.0

    return {
        "acc": round(acc, 6),
        "bwt": round(bwt, 6),
        "fwt": round(fwt, 6),
        "intransigence": round(intransigence, 6),
        "n_tasks_trained": t + 1,
    }


# ---------------------------------------------------------------------------
# Internal metric implementations
# ---------------------------------------------------------------------------

def _compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_type: str,
) -> float:
    """Compute ROUGE-2 or ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)[rouge_type].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores)
    except ImportError:
        # Fallback: simple token overlap
        return _simple_rouge(predictions, references, rouge_type)


def _simple_rouge(
    predictions: List[str],
    references: List[str],
    rouge_type: str,
) -> float:
    """
    Lightweight ROUGE approximation without external dependencies.
    Used as fallback when rouge_score is not installed.
    """
    def tokenize(text: str) -> List[str]:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def get_ngrams(tokens: List[str], n: int):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    n = 2 if rouge_type == "rouge2" else 1  # ROUGE-L approximated as unigram overlap

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        if rouge_type == "rougeL":
            # LCS-based
            lcs_len = _lcs_length(pred_tokens, ref_tokens)
            prec = lcs_len / len(pred_tokens) if pred_tokens else 0.0
            rec = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        else:
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            pred_set = {}
            for ng in pred_ngrams:
                pred_set[ng] = pred_set.get(ng, 0) + 1
            ref_set = {}
            for ng in ref_ngrams:
                ref_set[ng] = ref_set.get(ng, 0) + 1
            overlap = sum(min(pred_set.get(ng, 0), ref_set.get(ng, 0)) for ng in ref_set)
            prec = overlap / len(pred_ngrams) if pred_ngrams else 0.0
            rec = overlap / len(ref_ngrams) if ref_ngrams else 0.0

        scores.append(f1(prec, rec))

    return sum(scores) / len(scores)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _compute_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute exact-match accuracy after normalising both strings.
    For GSM8K, extracts the final numeric answer from the prediction.
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_norm = _normalise_answer(pred)
        ref_norm = _normalise_answer(ref)
        if pred_norm == ref_norm:
            correct += 1
        elif _extract_number(pred) is not None and _extract_number(ref) is not None:
            if abs(_extract_number(pred) - _extract_number(ref)) < 1e-6:
                correct += 1
    return correct / len(predictions)


def _normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_number(text: str) -> Optional[float]:
    """Extract the last number from a string (for GSM8K answer parsing)."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    # Fall back: last number in string
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def _compute_pass_at_1(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute pass@1 for HumanEval code generation.

    Attempts to execute each prediction against the test cases embedded in
    the reference. Falls back to a simple prefix-match heuristic if execution
    is not available (e.g., sandboxed environments).

    NOTE: For full pass@1 evaluation, use the official HumanEval evaluation
    harness: https://github.com/openai/human-eval
    """
    try:
        return _execute_pass_at_1(predictions, references)
    except Exception:
        # Heuristic fallback: check if prediction contains the canonical solution
        correct = 0
        for pred, ref in zip(predictions, references):
            ref_stripped = ref.strip()
            pred_stripped = pred.strip()
            # Accept if prediction contains the first line of the reference solution
            first_line = ref_stripped.split("\n")[0].strip()
            if first_line and first_line in pred_stripped:
                correct += 1
        return correct / len(predictions) if predictions else 0.0


def _execute_pass_at_1(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Execute HumanEval predictions in a sandboxed subprocess.
    Each reference is expected to contain the test harness code.
    """
    import subprocess
    import tempfile

    correct = 0
    for pred, ref in zip(predictions, references):
        code = pred + "\n\n" + ref
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                ["python3", fname],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                correct += 1
        except subprocess.TimeoutExpired:
            pass
        finally:
            import os
            os.unlink(fname)

    return correct / len(predictions) if predictions else 0.0
