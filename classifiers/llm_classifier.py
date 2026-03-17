"""
LLM-Based Content Policy Classifier
Uses Google Gemini 2.5 Flash via the google-genai SDK (googleapis/python-genai)
for structured content classification against YouTube's 14 advertiser-friendly
policy categories.

Source: https://support.google.com/youtube/answer/6162278
SDK: https://github.com/googleapis/python-genai

Author: Bhuvan Dontha
"""

import json
import os
import time
from typing import List, Dict, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai package required. Install: pip install google-genai"
    )

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Load policy taxonomy for prompt context
with open(os.path.join(DATA_DIR, "policy_taxonomy.json"), "r") as f:
    TAXONOMY = json.load(f)


def _build_taxonomy_prompt() -> str:
    """Build the policy taxonomy section for the system prompt."""
    lines = []
    for cat in TAXONOMY["categories"]:
        lines.append(f"\n### {cat['policy_name']} (ID: {cat['policy_id']})")
        for tier_name in ["GREEN", "YELLOW", "RED"]:
            if tier_name in cat["tiers"]:
                tier = cat["tiers"][tier_name]
                lines.append(f"  - {tier_name} ({tier['label']}): {tier['description'][:200]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a YouTube Trust & Safety content policy classifier.

Your task: Given a content description, classify it against YouTube's 14 advertiser-friendly policy categories.

POLICY TAXONOMY:
{_build_taxonomy_prompt()}

INSTRUCTIONS:
1. Read the content description carefully.
2. Identify ALL applicable policy violations (a single piece of content can violate MULTIPLE policies).
3. For each violation, assign:
   - policy_id: The policy ID from the taxonomy above
   - policy_name: The human-readable policy name
   - severity_tier: GREEN, YELLOW, or RED based on the tier descriptions
   - confidence: HIGH, MEDIUM, or LOW based on how clearly the content matches the tier
   - reasoning: 1-2 sentence explanation of WHY this classification was chosen

4. If the content does not violate any policy, return a single entry with policy_id "NONE".

RESPOND WITH VALID JSON ONLY. No markdown, no explanation outside the JSON.

Response format:
{{
  "classifications": [
    {{
      "policy_id": "POL-XXXX",
      "policy_name": "Category Name",
      "severity_tier": "GREEN|YELLOW|RED",
      "confidence": "HIGH|MEDIUM|LOW",
      "reasoning": "Explanation"
    }}
  ]
}}"""


class LLMClassifier:
    """Classifies content descriptions using Gemini API via google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """Initialize the classifier with a Gemini API key.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use. Default: gemini-2.5-flash (free tier).
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=key)
        self.model_name = model_name
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for deterministic classification
            response_mime_type="application/json",
        )

    def classify(self, description: str, retries: int = 2) -> List[Dict]:
        """Classify a single content description.

        Args:
            description: The content description to classify.
            retries: Number of retries on failure.

        Returns:
            List of classification dicts with policy_id, policy_name,
            severity_tier, confidence, reasoning, and classifier="llm".
        """
        prompt = f'Classify this content description:\n\n"{description}"'

        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.config,
                )
                result = json.loads(response.text)

                classifications = result.get("classifications", [])
                if not classifications:
                    return [self._no_violation()]

                # Validate and normalize each classification
                validated = []
                for c in classifications:
                    validated.append({
                        "policy_id": c.get("policy_id", "UNKNOWN"),
                        "policy_name": c.get("policy_name", "Unknown"),
                        "severity_tier": c.get("severity_tier", "GREEN").upper(),
                        "confidence": c.get("confidence", "LOW").upper(),
                        "reasoning": c.get("reasoning", "No reasoning provided"),
                        "classifier": "llm",
                    })
                return validated

            except json.JSONDecodeError:
                if attempt < retries:
                    time.sleep(1)
                    continue
                return [self._error_result("LLM returned non-JSON response")]

            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                return [self._error_result(f"API error: {str(e)[:100]}")]

    def classify_batch(
        self, descriptions: List[Dict], delay: float = 0.5, progress: bool = True
    ) -> List[Dict]:
        """Classify a batch of content descriptions.

        Args:
            descriptions: List of dicts with 'content_id' and 'description' keys.
            delay: Seconds to wait between API calls (rate limiting).
            progress: Whether to print progress updates.

        Returns:
            List of result dicts with content_id and classifications.
        """
        results = []
        total = len(descriptions)

        for i, item in enumerate(descriptions):
            content_id = item.get("content_id", f"UNKNOWN-{i}")
            desc = item.get("description", "")

            classifications = self.classify(desc)

            results.append({
                "content_id": content_id,
                "description": desc[:80],
                "classifications": classifications,
            })

            if progress and (i + 1) % 25 == 0:
                print(f"  [{i+1}/{total}] classified...")

            if delay > 0 and i < total - 1:
                time.sleep(delay)

        if progress:
            print(f"  [{total}/{total}] complete.")

        return results

    @staticmethod
    def _no_violation() -> Dict:
        return {
            "policy_id": "NONE",
            "policy_name": "No Violation Detected",
            "severity_tier": "GREEN",
            "confidence": "HIGH",
            "reasoning": "Content does not appear to violate any advertiser-friendly policy.",
            "classifier": "llm",
        }

    @staticmethod
    def _error_result(message: str) -> Dict:
        return {
            "policy_id": "ERROR",
            "policy_name": "Classification Error",
            "severity_tier": "UNKNOWN",
            "confidence": "NONE",
            "reasoning": message,
            "classifier": "llm",
        }


def flatten_batch_results(batch_results: List[Dict]) -> List[Dict]:
    """Flatten batch results into one row per classification for CSV/DataFrame export.

    Args:
        batch_results: Output from classify_batch().

    Returns:
        List of flat dicts suitable for pandas DataFrame.
    """
    rows = []
    for item in batch_results:
        for c in item["classifications"]:
            rows.append({
                "content_id": item["content_id"],
                "description": item["description"],
                "llm_policy_id": c["policy_id"],
                "llm_policy_name": c["policy_name"],
                "llm_severity_tier": c["severity_tier"],
                "llm_confidence": c["confidence"],
                "llm_reasoning": c["reasoning"],
            })
    return rows


# === CLI Test ===
if __name__ == "__main__":
    print("=" * 70)
    print("LLM Classifier — Test Run (Gemini 2.5 Flash)")
    print("SDK: google-genai (googleapis/python-genai)")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n\u26A0  GEMINI_API_KEY not set.")
        print("   Get a free key at: https://aistudio.google.com/apikey")
        print("   Then: export GEMINI_API_KEY='your-key-here'\n")
        exit(1)

    classifier = LLMClassifier(api_key=api_key)

    test_cases = [
        "Gaming video with graphic beheading scene in first 10 seconds",
        "Comedy sketch where host says 'what the hell' repeatedly",
        "Tutorial on how to 3D print a functional firearm at home",
        "Family-friendly cooking tutorial for kids making cupcakes",
        "Video promoting anti-vaccination conspiracy theories as facts",
        "Documentary about the history of the war on drugs in America",
        "Video titled 'F*ck This Product' with uncensored profanity in thumbnail",
    ]

    for desc in test_cases:
        print(f'\n\U0001F4DD "{desc[:60]}..."')
        results = classifier.classify(desc)
        for r in results:
            emoji = {"RED": "\U0001F534", "YELLOW": "\U0001F7E1", "GREEN": "\U0001F7E2"}.get(
                r["severity_tier"], "\u26AA"
            )
            print(f"   {emoji} {r['policy_name']} | {r['severity_tier']} | {r['confidence']}")
            print(f"      \u2192 {r['reasoning'][:100]}")
        time.sleep(0.5)  # Rate limit
