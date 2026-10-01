"""WER, CER, and RTF metrics computation."""
from __future__ import annotations

import math


def word_error_rate(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis."""
    if isinstance(reference, list) and isinstance(hypothesis, list):
        if not reference:
            return 0.0
        wers = [word_error_rate(r, h) for r, h in zip(reference, hypothesis)]
        return round(float(np.mean(wers)), 6) if 'np' in globals() else round(sum(wers) / len(wers), 6)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d = _edit_distance(ref_words, hyp_words)
    return round(d / len(ref_words), 6)


def char_error_rate(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """Compute Character Error Rate (CER)."""
    if isinstance(reference, list) and isinstance(hypothesis, list):
        if not reference:
            return 0.0
        cers = [char_error_rate(r, h) for r, h in zip(reference, hypothesis)]
        return round(sum(cers) / len(cers), 6)

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    d = _edit_distance(ref_chars, hyp_chars)
    return round(d / len(ref_chars), 6)


def real_time_factor(inference_time: float, audio_duration: float) -> float:
    """RTF = inference_time / audio_duration. Lower is faster."""
    if audio_duration <= 0:
        return 0.0
    return round(inference_time / audio_duration, 6)


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
