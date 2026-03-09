"""
tests/test_metrics.py
---------------------
Unit tests for evaluation/metrics.py
"""

import pytest
from evaluation.metrics import (
    compute_task_metric,
    compute_cl_metrics,
    _compute_accuracy,
    _extract_number,
    _normalise_answer,
)


class TestComputeTaskMetric:
    def test_rouge2_perfect(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = compute_task_metric(preds, refs, "rouge2")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_rouge2_no_overlap(self):
        preds = ["completely different text here"]
        refs = ["the cat sat on the mat"]
        score = compute_task_metric(preds, refs, "rouge2")
        assert score < 0.1

    def test_rougeL_partial(self):
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        score = compute_task_metric(preds, refs, "rougeL")
        assert 0.3 < score < 1.0

    def test_accuracy_exact_match(self):
        preds = ["Paris", "Berlin", "Tokyo"]
        refs = ["Paris", "Berlin", "Tokyo"]
        score = compute_task_metric(preds, refs, "accuracy")
        assert score == pytest.approx(1.0)

    def test_accuracy_partial(self):
        preds = ["Paris", "London", "Tokyo"]
        refs = ["Paris", "Berlin", "Tokyo"]
        score = compute_task_metric(preds, refs, "accuracy")
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_accuracy_case_insensitive(self):
        preds = ["paris"]
        refs = ["Paris"]
        score = compute_task_metric(preds, refs, "accuracy")
        assert score == pytest.approx(1.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_task_metric(["a"], ["b"], "bleu")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_task_metric(["a", "b"], ["c"], "accuracy")

    def test_empty_inputs(self):
        score = compute_task_metric([], [], "accuracy")
        assert score == 0.0


class TestComputeCLMetrics:
    def test_single_task_acc(self):
        R = [[0.8, None, None], [None, None, None], [None, None, None]]
        zero_shot = [0.0, 0.0, 0.0]
        metrics = compute_cl_metrics(R, zero_shot, ["T1", "T2", "T3"], 0)
        assert metrics["acc"] == pytest.approx(0.8)
        assert metrics["bwt"] == pytest.approx(0.0)
        assert metrics["n_tasks_trained"] == 1

    def test_two_tasks_bwt_forgetting(self):
        # After T1: R[0][0] = 0.8
        # After T2: R[1][0] = 0.6 (forgot T1), R[1][1] = 0.9
        R = [
            [0.8, None],
            [0.6, 0.9],
        ]
        zero_shot = [0.0, 0.0]
        metrics = compute_cl_metrics(R, zero_shot, ["T1", "T2"], 1)
        assert metrics["acc"] == pytest.approx(0.75)   # (0.6 + 0.9) / 2
        assert metrics["bwt"] == pytest.approx(-0.2)   # 0.6 - 0.8

    def test_fwt_positive(self):
        # R[0][1] = 0.3 means after training T1, model got 0.3 on T2 zero-shot
        # zero_shot[1] = 0.1 means random baseline on T2 is 0.1
        # FWT = 0.3 - 0.1 = 0.2
        R = [
            [0.8, 0.3],
            [0.6, 0.9],
        ]
        zero_shot = [0.0, 0.1]
        metrics = compute_cl_metrics(R, zero_shot, ["T1", "T2"], 1)
        assert metrics["fwt"] == pytest.approx(0.2)


class TestExtractNumber:
    def test_gsm8k_format(self):
        text = "The answer is #### 42"
        assert _extract_number(text) == pytest.approx(42.0)

    def test_plain_number(self):
        text = "The total is 3.14"
        assert _extract_number(text) == pytest.approx(3.14)

    def test_no_number(self):
        text = "No numbers here"
        assert _extract_number(text) is None

    def test_negative_number(self):
        text = "The result is -7"
        assert _extract_number(text) == pytest.approx(-7.0)


class TestNormaliseAnswer:
    def test_lowercase(self):
        assert _normalise_answer("PARIS") == "paris"

    def test_strip_punctuation(self):
        assert _normalise_answer("Paris!") == "paris"

    def test_collapse_whitespace(self):
        assert _normalise_answer("  Paris   France  ") == "paris france"
