"""
YouTube Video Content Extractor
Primary: Gemini native YouTube understanding (no scraping, no blocking)
Fallback: youtube-transcript-api (local only, blocked on cloud)

Gemini natively processes YouTube URLs — audio, visual, captions — and
returns a content summary. This bypasses all YouTube scraping restrictions.
Works in any language without translation libraries.

Author: Bhuvan Dontha
"""

import re
import os
from typing import Optional, Dict

# Primary: Gemini (always available if API key is set)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Fallback: youtube-transcript-api (may be blocked on cloud)
YTT_AVAILABLE = False
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YTT_AVAILABLE = True
except ImportError:
    pass


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None

    url = url.strip()

    # Direct video ID (11 chars)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    # Parse URL properly
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)

        # Standard: youtube.com/watch?v=ID
        if parsed.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
            query = parse_qs(parsed.query)
            if "v" in query:
                return query["v"][0]
            # Handle /shorts/ID and /embed/ID
            path_match = re.match(r'^/(shorts|embed)/([a-zA-Z0-9_-]{11})', parsed.path)
            if path_match:
                return path_match.group(2)

        # Short URL: youtu.be/ID
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")[:11]
    except Exception:
        pass

    # Fallback regex
    match = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


def _get_youtube_url(url_or_id: str) -> Optional[str]:
    """Convert any input to a full YouTube URL."""
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def get_transcript_via_gemini(url_or_id: str, api_key: Optional[str] = None) -> Dict:
    """Extract video content using Gemini's native YouTube understanding.

    Gemini processes the actual video (audio + visual + captions) directly.
    No scraping. No IP blocking. Works with any language natively.

    Args:
        url_or_id: YouTube URL or video ID.
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        Dict with full_text, video_id, char_count, source, original_language.
    """
    if not GENAI_AVAILABLE:
        return {"error": "google-genai package required. Install: pip install google-genai"}

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return {"error": "GEMINI_API_KEY required for Gemini-based extraction."}

    video_id = extract_video_id(url_or_id)
    if not video_id:
        return {"error": f"Could not extract video ID from: {url_or_id}"}

    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        client = genai.Client(api_key=key)

        # Gemini natively understands YouTube URLs passed as file URIs
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_uri(
                            file_uri=yt_url,
                            mime_type="video/*",
                        ),
                        types.Part.from_text(
                            text=(
                                "Extract the complete spoken content of this video verbatim. "
                                "Include ALL words spoken, including profanity, slurs, and controversial statements. "
                                "Do NOT censor, bleep, or replace any words with asterisks. "
                                "Do not add timestamps, speaker labels, or commentary. "
                                "If the video contains no speech, describe ALL visual content in detail "
                                "including any violence, nudity, drug use, weapons, or graphic imagery shown. "
                                "Return the full unedited transcript in the original language."
                            )
                        ),
                    ]
                )
            ],
            config=types.GenerateContentConfig(temperature=0.1),
        )

        full_text = response.text.strip()

        if not full_text:
            return {"error": "Gemini returned empty response for this video."}

        return {
            "full_text": full_text,
            "video_id": video_id,
            "char_count": len(full_text),
            "truncated": False,
            "duration_minutes": None,
            "original_language": "auto-detected",
            "translated_to": None,
            "source": "gemini_native",
        }

    except Exception as e:
        return {"error": f"Gemini video processing failed: {str(e)}"}


def get_transcript_via_api(url_or_id: str, max_chars: int = 100000) -> Dict:
    """Fallback: Fetch transcript via youtube-transcript-api.

    Works locally but may be blocked on cloud deployments.
    """
    if not YTT_AVAILABLE:
        return {"error": "youtube-transcript-api not installed."}

    video_id = extract_video_id(url_or_id)
    if not video_id:
        return {"error": f"Could not extract video ID from: {url_or_id}"}

    try:
        ytt_api = YouTubeTranscriptApi()

        # List available transcripts
        available = ytt_api.list(video_id)
        if not available:
            return {"error": "No transcripts available for this video."}

        # Pick English if available, otherwise first available
        languages = [t.language_code for t in available]
        selected_lang = "en" if "en" in languages else languages[0]

        transcript = ytt_api.fetch(video_id, languages=[selected_lang])

        segments = []
        total_duration = 0.0
        for snippet in transcript:
            segments.append(snippet.text)
            total_duration = max(total_duration, snippet.start + snippet.duration)

        full_text = " ".join(segments)
        truncated = len(full_text) > max_chars
        if truncated:
            full_text = full_text[:max_chars]

        return {
            "full_text": full_text,
            "video_id": video_id,
            "char_count": len(full_text),
            "truncated": truncated,
            "duration_minutes": round(total_duration / 60, 1),
            "original_language": selected_lang,
            "translated_to": None,
            "source": f"transcript_api_{selected_lang}",
        }

    except Exception as e:
        return {"error": f"Transcript API failed: {str(e)}"}


def get_transcript(url_or_id: str, max_chars: int = 100000) -> Dict:
    """Smart transcript extraction with automatic fallback.

    Order:
      1. Gemini native (works everywhere, any language, no blocking)
      2. youtube-transcript-api (local fallback)

    Args:
        url_or_id: YouTube URL or video ID.
        max_chars: Max characters for API fallback (Gemini handles its own limits).

    Returns:
        Dict with full_text, video_id, char_count, source, etc.
        On failure: Dict with 'error' key.
    """
    errors = []

    # Attempt 1: Gemini native
    if GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        result = get_transcript_via_gemini(url_or_id)
        if "error" not in result:
            return result
        errors.append(f"Gemini: {result['error']}")

    # Attempt 2: youtube-transcript-api fallback
    if YTT_AVAILABLE:
        result = get_transcript_via_api(url_or_id, max_chars=max_chars)
        if "error" not in result:
            return result
        errors.append(f"Transcript API: {result['error']}")

    # All methods failed
    return {"error": f"All extraction methods failed. {' | '.join(errors)}"}


def summarize_for_classification(
    transcript_text: str,
    api_key: Optional[str] = None,
) -> str:
    """Summarize transcript text into a content description for classification.

    Args:
        transcript_text: The transcript or content text.
        api_key: Gemini API key. Falls back to env var.

    Returns:
        2-3 sentence content description for policy classification.
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package required for summarization.")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key required for summarization.")

    client = genai.Client(api_key=key)

    truncated = transcript_text[:100000]

    prompt = f"""You are a content policy auditor. Analyze the following video transcript and complete a structured audit.

For EACH of the 14 categories below, respond with:
- "PRESENT" if signals are detected, followed by specific evidence from the transcript
- "ABSENT" if no signals detected

If the transcript is not in English, understand it in the original language first.

Respond ONLY with valid JSON. No markdown fences.

{{
  "video_topic": "One sentence describing what the video is about",
  "audit": {{
    "inappropriate_language": {{"status": "PRESENT or ABSENT", "evidence": "List specific words/phrases"}},
    "violence": {{"status": "PRESENT or ABSENT", "evidence": "Describe violent references"}},
    "adult_content": {{"status": "PRESENT or ABSENT", "evidence": "Describe sexual/adult references"}},
    "shocking_content": {{"status": "PRESENT or ABSENT", "evidence": "Describe graphic/disturbing references"}},
    "harmful_acts": {{"status": "PRESENT or ABSENT", "evidence": "Describe dangerous acts, misinformation, unverified claims"}},
    "hateful_derogatory": {{"status": "PRESENT or ABSENT", "evidence": "Describe hate speech, slurs, discrimination"}},
    "recreational_drugs": {{"status": "PRESENT or ABSENT", "evidence": "Describe drug use or references"}},
    "firearms": {{"status": "PRESENT or ABSENT", "evidence": "Describe firearms shown or referenced"}},
    "controversial_issues": {{"status": "PRESENT or ABSENT", "evidence": "Describe political/social/religious controversy"}},
    "sensitive_events": {{"status": "PRESENT or ABSENT", "evidence": "Describe war, terrorism, disasters, death references"}},
    "enabling_dishonest_behavior": {{"status": "PRESENT or ABSENT", "evidence": "Describe scams, fraud, deception"}},
    "inappropriate_for_kids": {{"status": "PRESENT or ABSENT", "evidence": "Describe content inappropriate for minors"}},
    "incendiary_demeaning": {{"status": "PRESENT or ABSENT", "evidence": "Describe public shaming, bullying, humiliation"}},
    "tobacco": {{"status": "PRESENT or ABSENT", "evidence": "Describe tobacco/smoking/vaping references"}}
  }}
}}

TRANSCRIPT:
{truncated}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    # Parse and assemble description from audit
    try:
        import json as json_mod
        audit_data = json_mod.loads(response.text)
        parts = [audit_data.get("video_topic", "Video content")]
        audit = audit_data.get("audit", {})
        findings = []
        for key_name, cat_data in audit.items():
            if isinstance(cat_data, dict) and cat_data.get("status", "").upper() == "PRESENT":
                evidence = cat_data.get("evidence", "")
                display = key_name.replace("_", " ").title()
                findings.append(f"{display}: {evidence}")
        if findings:
            parts.append("Policy-relevant signals: " + "; ".join(findings))
        else:
            parts.append("No policy-relevant signals detected.")
        return " ".join(parts)
    except Exception:
        return response.text.strip()


def summarize_youtube_direct(
    url_or_id: str,
    api_key: Optional[str] = None,
) -> str:
    """One-shot: Audit a YouTube video against all 14 policy categories.

    Instead of asking Gemini for a free-form description (which gets sanitized),
    this asks Gemini to perform a structured analytical audit — answering
    YES/NO per category with evidence. Safety filters are less aggressive
    on analytical outputs than on descriptive content generation.

    Args:
        url_or_id: YouTube URL or video ID.
        api_key: Gemini API key.

    Returns:
        Content description assembled from audit findings.
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package required.")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY required.")

    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url_or_id}")

    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=yt_url,
                        mime_type="video/*",
                    ),
                    types.Part.from_text(
                        text="""You are a content policy auditor. Analyze this video (audio, visuals, speech, text overlays) and complete the following structured audit.

For EACH of the 14 categories below, respond with:
- "PRESENT" if signals are detected, followed by specific evidence
- "ABSENT" if no signals detected

Respond ONLY with valid JSON. No markdown fences. No commentary.

{
  "video_topic": "One sentence describing what the video is about",
  "audit": {
    "inappropriate_language": {"status": "PRESENT or ABSENT", "evidence": "List specific words/phrases heard or shown"},
    "violence": {"status": "PRESENT or ABSENT", "evidence": "Describe violent scenes, fighting, weapons, injuries shown"},
    "adult_content": {"status": "PRESENT or ABSENT", "evidence": "Describe sexual content, nudity, suggestive material"},
    "shocking_content": {"status": "PRESENT or ABSENT", "evidence": "Describe graphic, disturbing, or gory content"},
    "harmful_acts": {"status": "PRESENT or ABSENT", "evidence": "Describe dangerous stunts, self-harm, harmful challenges, misinformation, or unverified medical claims"},
    "hateful_derogatory": {"status": "PRESENT or ABSENT", "evidence": "Describe hate speech, slurs, discrimination against any group"},
    "recreational_drugs": {"status": "PRESENT or ABSENT", "evidence": "Describe drug use, drug references, substance depiction, drug culture glorification"},
    "firearms": {"status": "PRESENT or ABSENT", "evidence": "Describe firearms shown, gun violence, weapons modification, or sales"},
    "controversial_issues": {"status": "PRESENT or ABSENT", "evidence": "Describe political, social, religious controversy discussed or depicted"},
    "sensitive_events": {"status": "PRESENT or ABSENT", "evidence": "Describe references to war, terrorism, disasters, mass casualties, or death"},
    "enabling_dishonest_behavior": {"status": "PRESENT or ABSENT", "evidence": "Describe scams, fraud, hacking, theft instructions, or deceptive practices"},
    "inappropriate_for_kids": {"status": "PRESENT or ABSENT", "evidence": "Describe content that targets or involves children inappropriately"},
    "incendiary_demeaning": {"status": "PRESENT or ABSENT", "evidence": "Describe public shaming, bullying, humiliation, or inflammatory provocation"},
    "tobacco": {"status": "PRESENT or ABSENT", "evidence": "Describe tobacco use, smoking, vaping, or tobacco product promotion"}
  }
}"""
                    ),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    # Parse the structured audit and assemble a content description
    try:
        import json
        audit_data = json.loads(response.text)

        # Build content description from audit findings
        parts = []
        topic = audit_data.get("video_topic", "Video content")
        parts.append(topic)

        audit = audit_data.get("audit", {})
        category_names = {
            "inappropriate_language": "Inappropriate Language",
            "violence": "Violence",
            "adult_content": "Adult Content",
            "shocking_content": "Shocking Content",
            "harmful_acts": "Harmful Acts",
            "hateful_derogatory": "Hateful & Derogatory Content",
            "recreational_drugs": "Recreational Drugs",
            "firearms": "Firearms",
            "controversial_issues": "Controversial Issues",
            "sensitive_events": "Sensitive Events",
            "enabling_dishonest_behavior": "Enabling Dishonest Behavior",
            "inappropriate_for_kids": "Inappropriate Content for Kids",
            "incendiary_demeaning": "Incendiary & Demeaning",
            "tobacco": "Tobacco-Related Content",
        }

        present_categories = []
        for key_name, display_name in category_names.items():
            cat_data = audit.get(key_name, {})
            if cat_data.get("status", "").upper() == "PRESENT":
                evidence = cat_data.get("evidence", "")
                present_categories.append(f"{display_name}: {evidence}")

        if present_categories:
            parts.append("Policy-relevant signals detected: " + "; ".join(present_categories))
        else:
            parts.append("No policy-relevant signals detected in this video.")

        return " ".join(parts)

    except (json.JSONDecodeError, KeyError, TypeError):
        # If JSON parsing fails, return raw response as fallback
        return response.text.strip()


# === CLI Test ===
if __name__ == "__main__":
    print("=" * 70)
    print("YouTube Video Content Extractor — Test")
    print("Methods: Gemini Native > Transcript API (fallback)")
    print("=" * 70)

    test_url = input("\nPaste a YouTube URL (or press Enter for default): ").strip()
    if not test_url:
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print(f"\nVideo ID: {extract_video_id(test_url)}")

    # Test smart extraction
    print("\n--- Smart Extraction (auto-fallback) ---")
    result = get_transcript(test_url)
    if "error" in result:
        print(f"\u274C {result['error']}")
    else:
        print(f"Source: {result['source']}")
        print(f"Language: {result.get('original_language')}")
        print(f"Characters: {result['char_count']}")
        print(f"\nFirst 300 chars:\n  {result['full_text'][:300]}...")

    # Test one-shot summarization
    if os.getenv("GEMINI_API_KEY") and GENAI_AVAILABLE:
        print("\n--- One-Shot Summarization (direct) ---")
        try:
            summary = summarize_youtube_direct(test_url)
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"\u274C {e}")
