"""Text normalizer for Indian language transcripts.

Rules:
- Preserve natural Hinglish as-is (e.g., "Sir plot ka payment tomorrow tak kar dena.")
- Do NOT translate English words to Hindi.
- Do NOT transliterate unless explicitly requested.
- Preserve: names, places, numbers, currency, addresses, dates.
- Normalization is applied AFTER transcription for WER/CER comparison only.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_for_wer(text: str, language: str = "hi") -> str:
    """Normalize transcript text for WER/CER computation.

    This normalization is ONLY for metric computation.
    The original transcript is always preserved.

    Steps:
      1. Unicode NFC normalization
      2. Lowercase (for Latin script)
      3. Strip punctuation except Devanagari-relevant characters
      4. Collapse whitespace
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase latin characters only (Devanagari has no case)
    text = text.lower()

    # Remove punctuation except Devanagari nukta, anusvara, visarga, chandrabindu
    # Keep: Devanagari range U+0900-U+097F, Latin letters, digits, spaces
    text = re.sub(r"[^\u0900-\u097f\u0a00-\u0a7f\u0b00-\u0b7f\u0c00-\u0c7f\u0d00-\u0d7f"
                  r"a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_numbers_for_display(text: str) -> str:
    """Convert Devanagari digit characters to ASCII digits in transcripts.

    Only used for display/export. Does not affect WER computation.
    """
    devanagari_digits = {
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    }
    for dev, ascii_d in devanagari_digits.items():
        text = text.replace(dev, ascii_d)
    return text
