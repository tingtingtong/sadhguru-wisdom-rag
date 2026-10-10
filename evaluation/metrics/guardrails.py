"""Guardrails evaluation metrics: Topic classification, I/O validation, prompt injection."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailMetrics:
    # Topic classification
    topic_true_positives: int
    topic_true_negatives: int
    topic_false_positives: int
    topic_false_negatives: int
    topic_precision: float
    topic_recall: float
    topic_f1: float
    topic_accuracy: float

    # Input validation
    input_valid_accepted: int
    input_invalid_rejected: int
    input_total: int

    # Output validation
    output_clean_count: int
    output_leaked_count: int
    prompt_injection_blocked: int


def compute_classification_metrics(
    predictions: list[bool],
    ground_truth: list[bool],
) -> dict:
    """Compute precision, recall, F1, accuracy for boolean classification."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_input_validation(test_inputs: list[dict]) -> dict:
    """
    Evaluate input validator against test cases.
    Each test_input: {"text": str, "should_pass": bool}
    """
    from guardrails.input_validator import validate_input

    results = {"accepted": 0, "rejected": 0, "correct": 0, "total": len(test_inputs)}

    for case in test_inputs:
        is_valid, _ = validate_input(case["text"])
        if is_valid:
            results["accepted"] += 1
        else:
            results["rejected"] += 1

        expected = case["should_pass"]
        if is_valid == expected:
            results["correct"] += 1

    results["accuracy"] = round(results["correct"] / results["total"], 4) if results["total"] else 0.0
    return results


def evaluate_output_validation(test_outputs: list[dict]) -> dict:
    """
    Evaluate output validator against test cases.
    Each test_output: {"text": str, "has_leak": bool}
    """
    from guardrails.output_validator import validate_output

    results = {"clean": 0, "modified": 0, "total": len(test_outputs)}

    for case in test_outputs:
        cleaned = validate_output(case["text"])
        if cleaned == case["text"]:
            results["clean"] += 1
        else:
            results["modified"] += 1

    return results


# Pre-built test cases for input validation
INPUT_VALIDATION_CASES = [
    {"text": "", "should_pass": False},
    {"text": "   ", "should_pass": False},
    {"text": "ab", "should_pass": False},
    {"text": "a" * 2001, "should_pass": False},
    {"text": "What is yoga?", "should_pass": True},
    {"text": "How does meditation help?", "should_pass": True},
    {"text": "x" * 2000, "should_pass": True},
    {"text": "   hi   ", "should_pass": True},
]

# Pre-built test cases for output validation
OUTPUT_VALIDATION_CASES = [
    {"text": "Sadhguru teaches about meditation.", "has_leak": False},
    {"text": "As an AI language model, I cannot...", "has_leak": True},
    {"text": "The system prompt says...", "has_leak": True},
    {"text": "You are a helpful AI assistant that...", "has_leak": True},
    {"text": "", "has_leak": False},
    {"text": "Yoga is a path to inner wellbeing.", "has_leak": False},
]
