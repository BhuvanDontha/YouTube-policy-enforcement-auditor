"""
YouTube Transcript Extractor
Pulls captions from a YouTube video URL, summarizes with Gemini for classification.

Author: Bhuvan Dontha
"""

import re
import os
from typing import Optional, Dict

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    raise ImportError(
        "youtube-transcript-api package required. Install: pip install youtube-transcript-api"
    )

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None

    url = url.strip()

    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})',
        r'(?:watch\?.*v=)([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_transcript(url_or_id: str, max_chars: int = 100000) -> Dict:
    """Fetch transcript from a YouTube video.

    Args:
        url_or_id: YouTube URL or video ID.
        max_chars: Max characters to return (truncates if longer).

    Returns:
        Dict with keys: full_text, source, char_count, truncated, video_id, duration_minutes.
        On failure: Dict with key 'error'.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return {"error": f"Could not extract video ID from: {url_or_id}"}

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=['en'])

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
            "source": "auto-captions" if transcript else "manual",
            "char_count": len(full_text),
            "truncated": truncated,
            "video_id": video_id,
            "duration_minutes": round(total_duration / 60, 1),
        }

    except Exception as e:
        return {"error": str(e)}


def summarize_for_classification(
    transcript_text: str,
    api_key: Optional[str] = None,
) -> str:
    """Summarize a transcript into a content description for classification.

    Args:
        transcript_text: The transcript text (can be long).
        api_key: Gemini API key. If None, reads from env.

    Returns:
        A 2-3 sentence content description.
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package required for summarization.")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key required for summarization.")

    client = genai.Client(api_key=key)

    truncated = transcript_text[:100000]

    prompt = f"""You are a content analyst. Given the following video transcript, write a 2-3 sentence summary describing what the video contains. Focus on elements relevant to advertiser-friendly content policy: violence, language, drug references, sensitive topics, adult content, controversial issues, harmful acts, etc.

If the content is benign, simply describe what it is about. Do NOT classify the content. Just describe it factually.

TRANSCRIPT:
{truncated}

SUMMARY (2-3 sentences):"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return response.text.strip()


# === CLI Test ===
if __name__ == "__main__":
    print("=" * 70)
    print("YouTube Transcript Extractor — Test")
    print("=" * 70)

    test_url = input("\nPaste a YouTube URL (or press Enter for default): ").strip()
    if not test_url:
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    result = get_transcript(test_url)
    if "error" in result:
        print(f"\u274C {result['error']}")
    else:
        print(f"\nVideo ID: {result['video_id']}")
        print(f"Duration: {result['duration_minutes']} min")
        print(f"Characters: {result['char_count']}")
        print(f"Truncated: {result['truncated']}")
        print(f"\nFirst 300 chars:\n  {result['full_text'][:300]}...")

        if os.getenv("GEMINI_API_KEY") and GENAI_AVAILABLE:
            print(f"\nSummarizing...")
            summary = summarize_for_classification(result["full_text"])
            print(f"Summary: {summary}")
