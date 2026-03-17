"""
Disagreement Engine — The Intellectual Core of the Project
Runs both classifiers on the same content and analyzes WHERE they disagree.

The insight: Disagreement zones between classifiers are exactly where
human reviewers add the most value. This is a resource allocation problem,
not an accuracy problem.

Author: Bhuvan Dontha
"""

import json
import os
import csv
from typing import List, Dict, Optional
from enum import Enum


class DisagreementType(Enum):
    """Four types of disagreement between LLM and rules classifiers."""
    LLM_CATCHES_RULES_MISS = "LLM finds violation, rules miss it"
    RULES_CATCH_LLM_MISS = "Rules find violation, LLM misses it"
    SAME_POLICY_DIFF_SEVERITY = "Same policy, different severity tier"
    DIFFERENT_POLICIES = "Different policies detected entirely"
    AGREEMENT = "Both classifiers agree"


SEVERITY_ORDER = {"EXTREME": 3, "RED": 3, "STRONG": 2, "YELLOW": 2, "MODERATE": 1, "GREEN": 1}


def _get_primary_classification(classifications: List[Dict]) -> Optional[Dict]:
    """Get the highest-severity classification from a list."""
    non_none = [c for c in classifications if c.get("policy_id") not in ("NONE", "ERROR", None)]
    if not non_none:
        return None
    return max(non_none, key=lambda c: SEVERITY_ORDER.get(c.get("severity_tier", "GREEN"), 0))


def _severity_gap(tier_a: str, tier_b: str) -> int:
    """Calculate the numeric gap between two severity tiers."""
    return abs(SEVERITY_ORDER.get(tier_a, 0) - SEVERITY_ORDER.get(tier_b, 0))


def compare_classifications(
    content_id: str,
    description: str,
    llm_results: List[Dict],
    rules_results: List[Dict],
) -> Dict:
    """Compare LLM and rules classifications for a single content description.

    Args:
        content_id: The content identifier.
        description: The content description text.
        llm_results: Classification results from the LLM classifier.
        rules_results: Classification results from the rules classifier.

    Returns:
        A dict containing the comparison analysis.
    """
    llm_primary = _get_primary_classification(llm_results)
    rules_primary = _get_primary_classification(rules_results)

    llm_found = llm_primary is not None
    rules_found = rules_primary is not None

    # Determine disagreement type
    if not llm_found and not rules_found:
        d_type = DisagreementType.AGREEMENT
        severity_gap = 0
        priority_score = 0.0
    elif llm_found and not rules_found:
        d_type = DisagreementType.LLM_CATCHES_RULES_MISS
        severity_gap = SEVERITY_ORDER.get(llm_primary.get("severity_tier", "GREEN"), 0)
        priority_score = severity_gap * 0.8  # LLM catching nuance is moderately important
    elif rules_found and not llm_found:
        d_type = DisagreementType.RULES_CATCH_LLM_MISS
        severity_gap = SEVERITY_ORDER.get(rules_primary.get("severity_tier", "GREEN"), 0)
        priority_score = severity_gap * 1.0  # LLM missing a keyword match is concerning
    elif llm_primary["policy_id"] == rules_primary["policy_id"]:
        if llm_primary.get("severity_tier") == rules_primary.get("severity_tier"):
            d_type = DisagreementType.AGREEMENT
            severity_gap = 0
            priority_score = 0.0
        else:
            d_type = DisagreementType.SAME_POLICY_DIFF_SEVERITY
            severity_gap = _severity_gap(
                llm_primary.get("severity_tier", "GREEN"),
                rules_primary.get("severity_tier", "GREEN"),
            )
            priority_score = severity_gap * 0.9  # Calibration gaps are important
    else:
        d_type = DisagreementType.DIFFERENT_POLICIES
        severity_gap = max(
            SEVERITY_ORDER.get(llm_primary.get("severity_tier", "GREEN"), 0),
            SEVERITY_ORDER.get(rules_primary.get("severity_tier", "GREEN"), 0),
        )
        priority_score = severity_gap * 1.2  # Completely different = highest concern

    return {
        "content_id": content_id,
        "description": description[:80],
        "disagreement_type": d_type.value,
        "is_disagreement": d_type != DisagreementType.AGREEMENT,
        "severity_gap": severity_gap,
        "human_review_priority": round(priority_score, 2),
        "llm_policy": llm_primary.get("policy_name", "None") if llm_primary else "None",
        "llm_severity": llm_primary.get("severity_tier", "N/A") if llm_primary else "N/A",
        "llm_confidence": llm_primary.get("confidence", "N/A") if llm_primary else "N/A",
        "llm_reasoning": llm_primary.get("reasoning", "")[:120] if llm_primary else "",
        "rules_policy": rules_primary.get("policy_name", "None") if rules_primary else "None",
        "rules_severity": rules_primary.get("severity_tier", "N/A") if rules_primary else "N/A",
    }


def run_ensemble(
    descriptions: List[Dict],
    llm_batch_results: List[Dict],
    rules_classify_fn,
) -> List[Dict]:
    """Run the full ensemble comparison across all descriptions.

    Args:
        descriptions: List of dicts with content_id and description.
        llm_batch_results: Output from LLMClassifier.classify_batch().
        rules_classify_fn: The rules classifier function (classify_rules).

    Returns:
        List of comparison dicts for all descriptions.
    """
    # Index LLM results by content_id
    llm_index = {item["content_id"]: item["classifications"] for item in llm_batch_results}

    comparisons = []
    for item in descriptions:
        cid = item["content_id"]
        desc = item["description"]

        llm_results = llm_index.get(cid, [])
        rules_results = rules_classify_fn(desc)

        comparison = compare_classifications(cid, desc, llm_results, rules_results)
        comparisons.append(comparison)

    return comparisons


def compute_ensemble_summary(comparisons: List[Dict]) -> Dict:
    """Compute summary statistics from ensemble comparisons.

    Returns:
        Dict with disagreement rates, type breakdown, and priority rankings.
    """
    total = len(comparisons)
    if total == 0:
        return {"total": 0}

    disagreements = [c for c in comparisons if c["is_disagreement"]]
    d_count = len(disagreements)

    # Type breakdown
    type_counts = {}
    for c in comparisons:
        t = c["disagreement_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Policy-level disagreement rates
    policy_disagreements = {}
    for c in disagreements:
        for policy in [c["llm_policy"], c["rules_policy"]]:
            if policy and policy != "None":
                if policy not in policy_disagreements:
                    policy_disagreements[policy] = 0
                policy_disagreements[policy] += 1

    # Top priority cases
    top_priority = sorted(disagreements, key=lambda x: x["human_review_priority"], reverse=True)[:10]

    return {
        "total_descriptions": total,
        "total_disagreements": d_count,
        "disagreement_rate": round(d_count / total * 100, 1) if total > 0 else 0,
        "type_breakdown": type_counts,
        "policy_disagreement_counts": dict(
            sorted(policy_disagreements.items(), key=lambda x: x[1], reverse=True)
        ),
        "top_priority_cases": [
            {
                "content_id": c["content_id"],
                "description": c["description"],
                "type": c["disagreement_type"],
                "priority": c["human_review_priority"],
                "llm_says": f"{c['llm_policy']} ({c['llm_severity']})",
                "rules_says": f"{c['rules_policy']} ({c['rules_severity']})",
            }
            for c in top_priority
        ],
    }


def save_comparisons_csv(comparisons: List[Dict], filepath: str):
    """Save comparison results to CSV."""
    if not comparisons:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"Saved {len(comparisons)} comparisons to {filepath}")


# === CLI Test ===
if __name__ == "__main__":
    print("=" * 70)
    print("Disagreement Engine — Unit Test")
    print("=" * 70)

    # Simulate LLM and rules outputs for testing
    mock_llm = [{"policy_id": "POL-VIOL", "policy_name": "Violence", "severity_tier": "RED",
                  "confidence": "HIGH", "reasoning": "Graphic beheading in gaming content", "classifier": "llm"}]
    mock_rules = [{"policy_id": "NONE", "policy_name": "No Violation Detected", "severity_tier": "GREEN",
                    "confidence": "LOW", "reasoning": "No keywords matched", "classifier": "rules"}]

    result = compare_classifications("TEST-001", "Gaming video with graphic beheading", mock_llm, mock_rules)
    print(f"\nTest 1 (LLM catches, rules miss):")
    print(f"  Type: {result['disagreement_type']}")
    print(f"  Priority: {result['human_review_priority']}")
    print(f"  LLM: {result['llm_policy']} ({result['llm_severity']})")
    print(f"  Rules: {result['rules_policy']} ({result['rules_severity']})")

    # Test agreement
    mock_llm2 = [{"policy_id": "POL-HARM", "policy_name": "Harmful Acts", "severity_tier": "RED",
                   "confidence": "HIGH", "reasoning": "Anti-vaccination content", "classifier": "llm"}]
    mock_rules2 = [{"policy_id": "POL-HARM", "policy_name": "Harmful Acts", "severity_tier": "RED",
                     "confidence": "LOW", "reasoning": "Keyword match", "classifier": "rules"}]

    result2 = compare_classifications("TEST-002", "Anti-vaccination conspiracy video", mock_llm2, mock_rules2)
    print(f"\nTest 2 (Agreement):")
    print(f"  Type: {result2['disagreement_type']}")
    print(f"  Is disagreement: {result2['is_disagreement']}")

    # Test severity disagreement
    mock_llm3 = [{"policy_id": "POL-LANG", "policy_name": "Inappropriate Language", "severity_tier": "RED",
                   "confidence": "MEDIUM", "reasoning": "Strong profanity detected", "classifier": "llm"}]
    mock_rules3 = [{"policy_id": "POL-LANG", "policy_name": "Inappropriate Language", "severity_tier": "GREEN",
                     "confidence": "LOW", "reasoning": "Matched 'damn'", "classifier": "rules"}]

    result3 = compare_classifications("TEST-003", "Comedy with language", mock_llm3, mock_rules3)
    print(f"\nTest 3 (Same policy, different severity):")
    print(f"  Type: {result3['disagreement_type']}")
    print(f"  Severity gap: {result3['severity_gap']}")
    print(f"  Priority: {result3['human_review_priority']}")

    print("\n\u2705 All tests passed.")
