"""Language and script detection for Indian speech transcripts."""
from __future__ import annotations

import re
import unicodedata


# Unicode script ranges
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_script_mix(text: str) -> dict[str, float]:
    """Estimate proportion of Devanagari vs. Latin characters in text.

    Returns:
        dict with keys "devanagari_ratio", "latin_ratio", "other_ratio"
    """
    if not text:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "other_ratio": 0.0}

    chars = [c for c in text if not unicodedata.category(c).startswith("Z")]
    total = len(chars)
    if total == 0:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "other_ratio": 0.0}

    dev = sum(1 for c in chars if _DEVANAGARI.match(c))
    lat = sum(1 for c in chars if _LATIN.match(c))
    other = total - dev - lat
    return {
        "devanagari_ratio": round(dev / total, 4),
        "latin_ratio": round(lat / total, 4),
        "other_ratio": round(other / total, 4),
    }


def classify_transcript_language(text: str) -> str:
    """Classify transcript language type.

    Returns one of:
      "hindi"         — mostly Devanagari
      "english"       — mostly Latin
      "hinglish"      — mixed Devanagari + Latin
      "indian_english"— all Latin but expected to be Indian English
      "unknown"
    """
    if not text or not text.strip():
        return "unknown"

    mix = detect_script_mix(text)
    dev = mix["devanagari_ratio"]
    lat = mix["latin_ratio"]

    if dev >= 0.8:
        return "hindi"
    if lat >= 0.9:
        return "english"
    if dev > 0.1 and lat > 0.1:
        return "hinglish"
    if lat >= 0.6:
        return "indian_english"
    return "unknown"
