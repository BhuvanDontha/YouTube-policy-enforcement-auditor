"""
Consistency Audit — Do similar inputs get the same classification?
Tests enforcement fairness by checking if semantically similar content
descriptions produce consistent classification results.

Author: Bhuvan Dontha
"""

from typing import List, Dict, Tuple
from collections import defaultdict


def generate_similarity_pairs(descriptions: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate pairs of content descriptions within the same policy category.

    These pairs SHOULD receive the same policy classification (though
    severity may differ). If they do not, that is an inconsistency.

    Args:
        descriptions: List of dicts with content_id, description, true_policy_id.

    Returns:
        List of (desc_a, desc_b) tuples from the same policy category.
    """
    by_policy = defaultdict(list)
    for d in descriptions:
        by_policy[d.get("true_policy_id", "UNKNOWN")].append(d)

    pairs = []
    for policy_id, items in by_policy.items():
        # Generate pairs within same policy (limit to avoid explosion)
        for i in range(min(len(items), 10)):
            for j in range(i + 1, min(len(items), 10)):
                pairs.append((items[i], items[j]))

    return pairs


def audit_consistency(
    pairs: List[Tuple[Dict, Dict]],
    classify_fn,
) -> List[Dict]:
    """Run consistency audit on content pairs.

    Args:
        pairs: List of (desc_a, desc_b) tuples.
        classify_fn: A classification function that takes a description string
                     and returns a list of classification dicts.

    Returns:
        List of audit result dicts.
    """
    results = []

    for desc_a, desc_b in pairs:
        class_a = classify_fn(desc_a["description"])
        class_b = classify_fn(desc_b["description"])

        # Get primary classification for each
        primary_a = _get_primary(class_a)
        primary_b = _get_primary(class_b)

        policy_match = _policies_match(primary_a, primary_b)
        severity_match = _severities_match(primary_a, primary_b)

        results.append({
            "content_id_a": desc_a.get("content_id", ""),
            "content_id_b": desc_b.get("content_id", ""),
            "description_a": desc_a["description"][:60],
            "description_b": desc_b["description"][:60],
            "true_policy": desc_a.get("true_policy_name", desc_a.get("true_policy_id", "")),
            "predicted_policy_a": primary_a.get("policy_name", "None") if primary_a else "None",
            "predicted_policy_b": primary_b.get("policy_name", "None") if primary_b else "None",
            "predicted_severity_a": primary_a.get("severity_tier", "N/A") if primary_a else "N/A",
            "predicted_severity_b": primary_b.get("severity_tier", "N/A") if primary_b else "N/A",
            "policy_consistent": policy_match,
            "severity_consistent": severity_match,
            "fully_consistent": policy_match and severity_match,
        })

    return results


def compute_consistency_summary(audit_results: List[Dict]) -> Dict:
    """Compute consistency summary statistics."""
    total = len(audit_results)
    if total == 0:
        return {"total_pairs": 0}

    policy_consistent = sum(1 for r in audit_results if r["policy_consistent"])
    severity_consistent = sum(1 for r in audit_results if r["severity_consistent"])
    fully_consistent = sum(1 for r in audit_results if r["fully_consistent"])

    # Per-policy consistency
    by_policy = defaultdict(lambda: {"total": 0, "consistent": 0})
    for r in audit_results:
        p = r["true_policy"]
        by_policy[p]["total"] += 1
        if r["fully_consistent"]:
            by_policy[p]["consistent"] += 1

    per_policy = {}
    for policy, counts in sorted(by_policy.items()):
        rate = counts["consistent"] / counts["total"] * 100 if counts["total"] > 0 else 0
        per_policy[policy] = {
            "total_pairs": counts["total"],
            "consistent_pairs": counts["consistent"],
            "consistency_rate": round(rate, 1),
        }

    return {
        "total_pairs": total,
        "policy_consistency_rate": round(policy_consistent / total * 100, 1),
        "severity_consistency_rate": round(severity_consistent / total * 100, 1),
        "full_consistency_rate": round(fully_consistent / total * 100, 1),
        "per_policy": per_policy,
        "inconsistent_examples": [
            r for r in audit_results if not r["fully_consistent"]
        ][:10],
    }


def _get_primary(classifications: List[Dict]):
    """Get highest-severity classification."""
    sev_order = {"RED": 3, "EXTREME": 3, "YELLOW": 2, "STRONG": 2, "GREEN": 1, "MODERATE": 1}
    non_none = [c for c in classifications if c.get("policy_id") not in ("NONE", "ERROR", None)]
    if not non_none:
        return None
    return max(non_none, key=lambda c: sev_order.get(c.get("severity_tier", "GREEN"), 0))


def _policies_match(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get("policy_name", "").lower() == b.get("policy_name", "").lower()


def _severities_match(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get("severity_tier", "") == b.get("severity_tier", "")
