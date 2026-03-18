"""
YouTube Transcript Extractor
Pulls captions from a YouTube video URL, summarizes with Gemini for classification.

Author: Bhuvan Dontha
"""

import re
import os
from typing import Optional, Dict
from langdetect import detect
from deep_translator import GoogleTranslator

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
    """Extract YouTube video ID from various URL formats robustly."""
    if not url:
        return None

    url = url.strip()

    # Direct video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    # Handle full URLs with parameters
    try:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)

        # Standard watch URL
        if parsed.hostname in ["www.youtube.com", "youtube.com"]:
            query = parse_qs(parsed.query)
            if "v" in query:
                return query["v"][0]

        # Short URL
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")

    except Exception:
        pass

    # Fallback regex
    match = re.search(r'([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


def translate_large_text(text: str, chunk_size: int = 1000) -> str:
    """Robust translation with chunking + error handling."""

    translated_chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]

        # Skip empty / whitespace chunks
        if not chunk.strip():
            continue

        try:
            translated = GoogleTranslator(
                source='auto',
                target='en'
            ).translate(chunk)

            translated_chunks.append(translated)

        except Exception:
            # Fallback: keep original chunk (DO NOT break pipeline)
            translated_chunks.append(chunk)

    return " ".join(translated_chunks)


def get_transcript(url_or_id: str, max_chars: int = 100000) -> Dict:
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return {"error": f"Could not extract video ID from: {url_or_id}"}

    try:
        ytt_api = YouTubeTranscriptApi()

        # -------------------------
        # Step 1: List available transcripts
        # -------------------------
        available = ytt_api.list(video_id)

        if not available:
            return {"error": "No transcripts available for this video."}

        # -------------------------
        # Step 2: Pick best language
        # Priority:
        #   1. English if available
        #   2. Otherwise first available language
        # -------------------------
        languages = [t.language_code for t in available]

        if "en" in languages:
            selected_lang = "en"
        else:
            selected_lang = languages[0]

        # -------------------------
        # Step 3: Fetch transcript using selected language
        # -------------------------
        transcript = ytt_api.fetch(video_id, languages=[selected_lang])

        segments = []
        total_duration = 0.0

        for snippet in transcript:
            segments.append(snippet.text)
            total_duration = max(
                total_duration,
                snippet.start + snippet.duration
            )

        full_text = " ".join(segments)

        # -------------------------
        # Step 4: Language Detection
        # -------------------------
        try:
            language = detect(full_text)
        except:
            language = selected_lang or "unknown"

        translated = False

        # -------------------------
        # Step 5: Translate if needed (SAFE + CONTROLLED)
        # -------------------------
        if language != "en":

            text_for_translation = full_text[:4000]

            try:
                translated_text = translate_large_text(text_for_translation)

                # Only update if translation actually changed content
                if translated_text and translated_text != text_for_translation:
                    full_text = translated_text
                    translated = True

            except Exception:
                # Do NOT fail pipeline if translation fails
                translated = False

        truncated = len(full_text) > max_chars
        if truncated:
            full_text = full_text[:max_chars]

        return {
            "full_text": full_text,
            "video_id": video_id,
            "char_count": len(full_text),
            "truncated": truncated,
            "duration_minutes": round(total_duration / 60, 1),
            "original_language": language,
            "translated_to": "en" if translated else None,
            "source": f"youtube_detected_{language}",
        }

    except Exception as e:
        return {"error": f"Could not fetch transcript: {str(e)}"}


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
        print(f"Language: {result.get('original_language')}")
        print(f"Translated: {result.get('translated_to')}")
        print(f"Source: {result.get('source')}")

        if os.getenv("GEMINI_API_KEY") and GENAI_AVAILABLE:
            print(f"\nSummarizing...")
            summary = summarize_for_classification(result["full_text"])
            print(f"Summary: {summary}")
