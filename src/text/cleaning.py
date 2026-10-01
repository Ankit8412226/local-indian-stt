"""Text cleaning and normalization utility for Indian ASR training.

Preserves Hinglish / Hindi-English code-switching without destroying English words.
Normalizes Devanagari Unicode (NFKC) and cleans whitespace and punctuation artifacts.
"""
from __future__ import annotations

import re
import unicodedata
from src.text.normalizer import normalize_numbers_for_display


def clean_transcript(text: str | None, language: str = "hi") -> tuple[str, bool, str | None]:
    """Clean transcript text.

    Returns:
        tuple of (cleaned_text, is_valid, quarantine_reason)

    Rules:
      - Empty or whitespace-only -> Invalid (quarantined)
      - Devanagari NFKC Unicode normalization applied
      - Digit words / numbers preserved or normalized
      - Hinglish (Hindi + English) words preserved intact
      - Excessive punctuation / non-speech noise tags sanitized
    """
    if not text or not text.strip():
        return "", False, "empty_transcript"

    raw = text.strip()

    # 1. Unicode normalization (NFKC)
    cleaned = unicodedata.normalize("NFKC", raw)

    # 2. Normalize Indic digits (e.g. Devanagari numerals to standard)
    cleaned = normalize_numbers_for_display(cleaned)

    # 3. Strip unwanted control chars or zero-width joiners if malformed
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", cleaned)  # Strip noise tags like [noise], (laughter)

    # 4. Strip excessive punctuation while preserving Devanagari, Latin, space
    cleaned = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0B00-\u0B7F\u0C00-\u0C7F\u0D00-\u0D7F]", "", cleaned)

    # 5. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 2:
        return "", False, "transcript_too_short"

    return cleaned, True, None
