"""
Rules-Based Classifier — Deterministic Baseline
Classifies content descriptions against YouTube's 14 policy categories
using keyword matching and pattern rules.

This is intentionally simple. Its purpose is to be the BASELINE
that the LLM classifier should outperform.

Author: Bhuvan Dontha
"""

import json
import re
import os
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

with open(os.path.join(DATA_DIR, "policy_taxonomy.json"), "r") as f:
    taxonomy = json.load(f)


def _match_keywords(text: str, keywords: list) -> int:
    """Count how many keywords match in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def classify_rules(description: str) -> List[Dict]:
    """
    Classify a content description using keyword matching rules.

    Returns a list of policy violations detected, each with:
    - policy_id, policy_name, severity_tier, confidence, reasoning
    """
    results = []
    text = description.lower()

    for category in taxonomy["categories"]:
        best_tier = None
        best_score = 0
        best_tier_label = None

        # Check tiers from most severe to least (RED > YELLOW > GREEN)
        for tier_name in ["RED", "YELLOW", "GREEN"]:
            if tier_name not in category["tiers"]:
                continue

            tier = category["tiers"][tier_name]
            score = _match_keywords(text, tier["keywords"])

            if score > 0 and (best_tier is None or
                              _tier_priority(tier_name) > _tier_priority(best_tier)):
                best_tier = tier_name
                best_score = score
                best_tier_label = tier["label"]

        if best_tier is not None:
            confidence = "HIGH" if best_score >= 3 else "MEDIUM" if best_score >= 2 else "LOW"
            results.append({
                "policy_id": category["policy_id"],
                "policy_name": category["policy_name"],
                "severity_tier": best_tier,
                "confidence": confidence,
                "match_count": best_score,
                "reasoning": f"Keyword match ({best_score} keywords matched for {best_tier} tier: {best_tier_label})",
                "classifier": "rules"
            })

    # If nothing matched, return no violation
    if not results:
        results.append({
            "policy_id": "NONE",
            "policy_name": "No Violation Detected",
            "severity_tier": "GREEN",
            "confidence": "LOW",
            "match_count": 0,
            "reasoning": "No policy keywords matched in content description",
            "classifier": "rules"
        })

    return results


def _tier_priority(tier: str) -> int:
    """Return numeric priority for severity tier comparison."""
    return {"RED": 3, "YELLOW": 2, "GREEN": 1}.get(tier, 0)


# === Quick test ===
if __name__ == "__main__":
    test_cases = [
        "Gaming video with graphic beheading scene in first 10 seconds",
        "Comedy sketch where host says 'what the hell' repeatedly",
        "Tutorial on how to 3D print a functional firearm at home",
        "Family-friendly cooking tutorial for kids making cupcakes",
        "Video promoting anti-vaccination conspiracy theories as facts",
        "Video titled 'F*ck This Product' with uncensored profanity",
        "Documentary about World War 2 showing fully blurred archival bodies",
    ]

    print("=" * 70)
    print("Rules-Based Classifier — Test Results")
    print("=" * 70)

    for desc in test_cases:
        print(f"\n📝 \"{desc[:60]}...\"")
        results = classify_rules(desc)
        for r in results:
            emoji = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(r["severity_tier"], "⚪")
            print(f"   {emoji} {r['policy_name']} | {r['severity_tier']} | {r['confidence']} | {r['reasoning']}")
