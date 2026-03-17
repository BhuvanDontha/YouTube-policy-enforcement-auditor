"""
Policy Enforcement Auditor — Main Pipeline Runner
Orchestrates the full analysis: data generation → classification → 
disagreement detection → evaluation → output.

Run: python run_pipeline.py

Author: Bhuvan Dontha
"""

import os
import sys
import csv
import json
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from classifiers.rules_classifier import classify_rules
from classifiers.llm_classifier import LLMClassifier, flatten_batch_results
from classifiers.ensemble import run_ensemble, compute_ensemble_summary, save_comparisons_csv
from analysis.metrics import load_ground_truth, evaluate_classifier, format_evaluation_report
from analysis.consistency_audit import generate_similarity_pairs, audit_consistency, compute_consistency_summary

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_descriptions(filepath: str):
    """Load content descriptions from CSV."""
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  POLICY ENFORCEMENT AUDITOR — Full Pipeline")
    print("=" * 70)

    # --- Step 1: Load data ---
    print("\n[1/6] Loading data...")
    descriptions = load_descriptions(os.path.join(DATA_DIR, "synthetic_content.csv"))
    ground_truth = load_ground_truth(os.path.join(DATA_DIR, "ground_truth.csv"))
    print(f"  Loaded {len(descriptions)} descriptions, {len(ground_truth)} ground truth labels")

    # --- Step 2: Run rules classifier ---
    print("\n[2/6] Running rules-based classifier on all descriptions...")
    rules_results = []
    for item in descriptions:
        classifications = classify_rules(item["description"])
        rules_results.append({
            "content_id": item["content_id"],
            "description": item["description"][:80],
            "classifications": classifications,
        })
    print(f"  Rules classifier complete: {len(rules_results)} descriptions classified")

    # --- Step 3: Run LLM classifier ---
    print("\n[3/6] Running LLM classifier (Gemini 2.5 Flash)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  \u26A0 GEMINI_API_KEY not set. Skipping LLM classification.")
        print("    Set it with: export GEMINI_API_KEY='your-key'")
        print("    Get a free key at: https://aistudio.google.com/apikey")
        print("    Running with rules-only mode for now.\n")
        llm_batch_results = rules_results  # Fallback: use rules as both
        llm_flat = []
    else:
        classifier = LLMClassifier(api_key=api_key)
        llm_batch_results = classifier.classify_batch(descriptions, delay=0.3)
        llm_flat = flatten_batch_results(llm_batch_results)

        # Save LLM results
        if llm_flat:
            with open(os.path.join(OUTPUT_DIR, "llm_classifications.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=llm_flat[0].keys())
                writer.writeheader()
                writer.writerows(llm_flat)
            print(f"  Saved LLM results to outputs/llm_classifications.csv")

    # --- Step 4: Run disagreement engine ---
    print("\n[4/6] Running disagreement engine...")
    comparisons = run_ensemble(descriptions, llm_batch_results, classify_rules)
    save_comparisons_csv(comparisons, os.path.join(OUTPUT_DIR, "disagreements.csv"))

    summary = compute_ensemble_summary(comparisons)
    print(f"  Total disagreements: {summary['total_disagreements']}/{summary['total_descriptions']} "
          f"({summary['disagreement_rate']}%)")
    print(f"  Type breakdown:")
    for dtype, count in summary.get("type_breakdown", {}).items():
        print(f"    {dtype}: {count}")

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Step 5: Evaluate classifier accuracy ---
    print("\n[5/6] Evaluating classifier accuracy against ground truth...")
    if llm_flat:
        eval_results = evaluate_classifier(llm_flat, ground_truth)
        report = format_evaluation_report(eval_results)
        print(report)

        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w") as f:
            f.write(report)
    else:
        print("  Skipped (no LLM results available)")

    # --- Step 6: Run consistency audit ---
    print("\n[6/6] Running consistency audit...")
    pairs = generate_similarity_pairs(descriptions)
    print(f"  Generated {len(pairs)} similarity pairs")

    if len(pairs) > 100:
        pairs = pairs[:100]  # Limit for API rate reasons
        print(f"  Limited to 100 pairs for rate limiting")

    audit_results = audit_consistency(pairs, classify_rules)  # Using rules for consistency (no API cost)
    consistency = compute_consistency_summary(audit_results)
    print(f"  Full consistency rate: {consistency['full_consistency_rate']}%")

    with open(os.path.join(OUTPUT_DIR, "consistency_audit.json"), "w") as f:
        json.dump(consistency, f, indent=2, default=str)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  - llm_classifications.csv    (LLM classification results)")
    print(f"  - disagreements.csv          (LLM vs rules comparison)")
    print(f"  - ensemble_summary.json      (Disagreement statistics)")
    print(f"  - evaluation_report.json     (Accuracy metrics)")
    print(f"  - evaluation_report.txt      (Readable report)")
    print(f"  - consistency_audit.json     (Consistency analysis)")
    print(f"\n  Next: Run the Streamlit app:")
    print(f"    streamlit run app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
