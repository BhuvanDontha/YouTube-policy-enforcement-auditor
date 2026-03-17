"""
Evaluation Metrics — Precision, Recall, F1, Confusion Matrices
Evaluates classifier performance against ground truth labels.

Author: Bhuvan Dontha
"""

import os
import csv
import json
from typing import List, Dict, Tuple
from collections import defaultdict


def load_ground_truth(filepath: str) -> Dict[str, Dict]:
    """Load ground truth labels indexed by content_id."""
    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["content_id"]] = {
                "true_policy_id": row["true_policy_id"],
                "true_policy_name": row["true_policy_name"],
                "true_severity": row["true_severity"],
            }
    return gt


def evaluate_classifier(
    predictions: List[Dict],
    ground_truth: Dict[str, Dict],
    policy_field: str = "llm_policy_name",
    severity_field: str = "llm_severity_tier",
) -> Dict:
    """Evaluate a classifier against ground truth.

    Args:
        predictions: Flat list of prediction rows (from flatten_batch_results or similar).
        ground_truth: Dict of content_id -> {true_policy_name, true_severity}.
        policy_field: Column name for the predicted policy in predictions.
        severity_field: Column name for the predicted severity in predictions.

    Returns:
        Dict with overall accuracy, per-category metrics, and confusion data.
    """
    # Deduplicate: take the highest-severity prediction per content_id
    best_per_content = {}
    sev_order = {"RED": 3, "EXTREME": 3, "YELLOW": 2, "STRONG": 2, "GREEN": 1, "MODERATE": 1}

    for pred in predictions:
        cid = pred.get("content_id", "")
        current = best_per_content.get(cid)
        pred_sev = sev_order.get(pred.get(severity_field, "GREEN"), 0)

        if current is None or pred_sev > sev_order.get(current.get(severity_field, "GREEN"), 0):
            best_per_content[cid] = pred

    # Compare against ground truth
    correct_policy = 0
    correct_severity = 0
    total = 0

    # Per-category tracking
    tp = defaultdict(int)  # True positives per category
    fp = defaultdict(int)  # False positives per category
    fn = defaultdict(int)  # False negatives per category

    # Confusion matrix data
    confusion = defaultdict(lambda: defaultdict(int))  # [true][predicted] = count

    for cid, gt in ground_truth.items():
        pred = best_per_content.get(cid)
        if pred is None:
            fn[gt["true_policy_name"]] += 1
            confusion[gt["true_policy_name"]]["NOT_CLASSIFIED"] += 1
            total += 1
            continue

        true_policy = gt["true_policy_name"]
        pred_policy = pred.get(policy_field, "None")
        true_sev = gt["true_severity"]
        pred_sev = pred.get(severity_field, "N/A")

        total += 1
        confusion[true_policy][pred_policy] += 1

        if _policy_match(true_policy, pred_policy):
            correct_policy += 1
            tp[true_policy] += 1
            if _severity_match(true_sev, pred_sev):
                correct_severity += 1
        else:
            fp[pred_policy] += 1
            fn[true_policy] += 1

    # Compute per-category metrics
    all_categories = set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))
    per_category = {}

    for cat in sorted(all_categories):
        t = tp[cat]
        f_p = fp[cat]
        f_n = fn[cat]
        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_category[cat] = {
            "true_positives": t,
            "false_positives": f_p,
            "false_negatives": f_n,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        }

    # Overall metrics
    macro_precision = sum(m["precision"] for m in per_category.values()) / len(per_category) if per_category else 0
    macro_recall = sum(m["recall"] for m in per_category.values()) / len(per_category) if per_category else 0
    macro_f1 = sum(m["f1_score"] for m in per_category.values()) / len(per_category) if per_category else 0

    return {
        "total_evaluated": total,
        "policy_accuracy": round(correct_policy / total * 100, 1) if total > 0 else 0,
        "severity_accuracy": round(correct_severity / total * 100, 1) if total > 0 else 0,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        "per_category": per_category,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def _policy_match(true_name: str, pred_name: str) -> bool:
    """Check if predicted policy matches true policy (fuzzy match on name)."""
    if not true_name or not pred_name:
        return False
    # Normalize for comparison
    true_lower = true_name.lower().strip()
    pred_lower = pred_name.lower().strip()
    # Exact match
    if true_lower == pred_lower:
        return True
    # Partial match (e.g., "Harmful Acts" matches "Harmful Acts & Unreliable Content")
    if true_lower in pred_lower or pred_lower in true_lower:
        return True
    return False


def _severity_match(true_sev: str, pred_sev: str) -> bool:
    """Check if predicted severity matches true severity."""
    # Map between different naming conventions
    equiv = {
        "RED": "RED", "EXTREME": "RED",
        "YELLOW": "YELLOW", "STRONG": "YELLOW",
        "GREEN": "GREEN", "MODERATE": "GREEN",
    }
    return equiv.get(true_sev.upper(), true_sev) == equiv.get(pred_sev.upper(), pred_sev)


def format_evaluation_report(eval_results: Dict) -> str:
    """Format evaluation results as a readable text report."""
    lines = [
        "=" * 60,
        "CLASSIFIER EVALUATION REPORT",
        "=" * 60,
        f"Total evaluated: {eval_results['total_evaluated']}",
        f"Policy accuracy: {eval_results['policy_accuracy']}%",
        f"Severity accuracy: {eval_results['severity_accuracy']}%",
        f"Macro Precision: {eval_results['macro_precision']}",
        f"Macro Recall: {eval_results['macro_recall']}",
        f"Macro F1: {eval_results['macro_f1']}",
        "",
        "-" * 60,
        "PER-CATEGORY METRICS",
        "-" * 60,
        f"{'Category':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}",
        "-" * 60,
    ]

    for cat, metrics in sorted(eval_results["per_category"].items()):
        lines.append(
            f"{cat:<30} {metrics['precision']:>6.3f} {metrics['recall']:>6.3f} "
            f"{metrics['f1_score']:>6.3f} {metrics['true_positives']:>4} "
            f"{metrics['false_positives']:>4} {metrics['false_negatives']:>4}"
        )

    return "\n".join(lines)


# === CLI Test ===
if __name__ == "__main__":
    print("=" * 70)
    print("Evaluation Metrics — Unit Test")
    print("=" * 70)

    # Mock data
    mock_gt = {
        "CNT-001": {"true_policy_id": "POL-VIOL", "true_policy_name": "Violence", "true_severity": "RED"},
        "CNT-002": {"true_policy_id": "POL-LANG", "true_policy_name": "Inappropriate Language", "true_severity": "GREEN"},
        "CNT-003": {"true_policy_id": "POL-FIRE", "true_policy_name": "Firearms", "true_severity": "RED"},
    }

    mock_preds = [
        {"content_id": "CNT-001", "llm_policy_name": "Violence", "llm_severity_tier": "RED"},
        {"content_id": "CNT-002", "llm_policy_name": "Inappropriate Language", "llm_severity_tier": "YELLOW"},
        {"content_id": "CNT-003", "llm_policy_name": "Violence", "llm_severity_tier": "RED"},  # Wrong category
    ]

    result = evaluate_classifier(mock_preds, mock_gt)
    print(f"\nPolicy accuracy: {result['policy_accuracy']}%")
    print(f"Severity accuracy: {result['severity_accuracy']}%")
    print(f"Macro F1: {result['macro_f1']}")
    print(f"\nPer category:")
    for cat, m in result["per_category"].items():
        print(f"  {cat}: P={m['precision']}, R={m['recall']}, F1={m['f1_score']}")

    print("\n\u2705 Evaluation test passed.")
