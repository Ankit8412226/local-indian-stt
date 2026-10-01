"""Code-switching detection utilities for Hindi-English (Hinglish) transcripts.

Preserves the natural Hinglish structure.
Does NOT translate or transliterate.
"""
from __future__ import annotations

import re

# Common Hindi function words (Romanized) that signal Hinglish
_HINDI_ROMAN_WORDS = frozenset([
    "ka", "ki", "ke", "ko", "ne", "se", "mein", "par", "tak", "aur",
    "hai", "hain", "tha", "thi", "the", "ho", "hoga", "hogi", "honge",
    "kar", "karo", "karna", "karte", "karta", "karti", "raha", "rahi",
    "nahi", "nhi", "bhi", "sirf", "lekin", "aur", "ya", "kya", "yeh",
    "woh", "iske", "uske", "apna", "apni", "apne", "ek", "do", "teen",
    "abhi", "phir", "fir", "sab", "matlab", "theek", "thik", "haan", "han",
    "nahin", "bahut", "bohot", "paise", "paisa", "rupay", "rupee",
    "sir", "ji", "sahab", "bhai", "didi", "acha", "accha",
])


def detect_code_switching(text: str) -> bool:
    """Return True if the text shows Hindi-English code-switching (Hinglish).

    Strategy:
    1. Check if both Devanagari and Latin characters are present.
    2. Also check if Romanized Hindi function words appear in an otherwise Latin transcript.
    """
    if not text:
        return False

    import unicodedata
    has_devanagari = any("\u0900" <= c <= "\u097f" for c in text)
    has_latin = any(c.isalpha() and c.isascii() for c in text)

    if has_devanagari and has_latin:
        return True

    # Check Romanized Hinglish (all Latin but mixed vocabulary)
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    hindi_hits = words & _HINDI_ROMAN_WORDS
    if len(hindi_hits) >= 2 and len(words) > 3:
        return True

    return False


def tag_code_switch_spans(text: str) -> list[dict]:
    """Tag spans of Devanagari vs Latin in a transcript.

    Returns a list of dicts: [{"text": ..., "script": "devanagari"|"latin"|"other"}]
    Useful for fine-grained analysis and future WER category breakdowns.
    """
    if not text:
        return []

    spans = []
    current_script = None
    current_span = []

    def _script_of(c: str) -> str:
        if "\u0900" <= c <= "\u097f":
            return "devanagari"
        if c.isalpha() and c.isascii():
            return "latin"
        return "other"

    for char in text:
        s = _script_of(char)
        if s != current_script:
            if current_span:
                spans.append({"text": "".join(current_span), "script": current_script})
            current_span = [char]
            current_script = s
        else:
            current_span.append(char)

    if current_span:
        spans.append({"text": "".join(current_span), "script": current_script})

    return spans
