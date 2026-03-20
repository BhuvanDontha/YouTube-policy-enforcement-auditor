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
                                "Extract the complete transcript of this video. "
                                "Return ONLY the spoken text, in the original language. "
                                "Do not add timestamps, speaker labels, or commentary. "
                                "If the video has no speech, describe the visual content instead."
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

    prompt = f"""You are a content analyst. Given the following video transcript, write a 2-3 sentence summary describing what the video contains. Focus on elements relevant to 14 advertiser-friendly content policy: Inappropriate language, Violence, Adult content, Shocking content, Harmful acts and unreliable content, Hateful & derogatory content, Recreational drugs and drug-related content, Firearms-related content, Controversial issues, Sensitive events, Enabling dishonest behavior, Inappropriate content for kids and families, Incendiary and demeaning, Tobacco-related content.

If the content is benign, simply describe what it is about. Do NOT classify the content. Just describe it factually.

If the transcript is not in English, first understand it in its original language, then write the summary in English.

TRANSCRIPT:
{truncated}

SUMMARY (2-3 sentences, in English):"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return response.text.strip()


def summarize_youtube_direct(
    url_or_id: str,
    api_key: Optional[str] = None,
) -> str:
    """One-shot: Summarize a YouTube video for classification in a single Gemini call.

    Skips transcript extraction entirely — Gemini watches the video and summarizes.
    Most efficient method: 1 API call instead of 2.

    Args:
        url_or_id: YouTube URL or video ID.
        api_key: Gemini API key.

    Returns:
        2-3 sentence content description for policy classification.
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
                        text=(
                            "You are a content analyst. Write a 2-3 sentence summary "
                            "of this video focusing on elements relevant to advertiser-friendly "
                            "content policy: violence, language, drug references, sensitive topics, "
                            "adult content, controversial issues, harmful acts, etc. "
                            "If the content is benign, simply describe what it is about. "
                            "Do NOT classify the content. Just describe it factually. "
                            "Always respond in English regardless of the video's language."
                        )
                    ),
                ]
            )
        ],
        config=types.GenerateContentConfig(temperature=0.2),
    )

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
