"""Text normalisation shared by training and inference.

Keeping a single implementation guarantees that the text the model sees at
prediction time is processed exactly like the text it was trained on.
"""

from __future__ import annotations

import re

# Compiled once at import time - these run on every request.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_BRACKET_RE = re.compile(r"\[.*?\]")
_DIGIT_WORD_RE = re.compile(r"\w*\d\w*")
_NON_WORD_RE = re.compile(r"[\W_]+")
_WS_RE = re.compile(r"\s+")

# The ISOT/Kaggle "True.csv" articles almost all start with a Reuters dateline
# such as "WASHINGTON (Reuters) - ". Leaving it in lets a model "cheat" by
# learning the word "reuters" instead of anything about the content, which
# inflates offline accuracy and generalises poorly to other sources.
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z .,'/-]{0,60}\(\s*reuters\s*\)\s*[-–—]\s*", re.I)
_REUTERS_RE = re.compile(r"\breuters\b", re.I)


def strip_source_leakage(text: str) -> str:
    """Remove publisher datelines/tokens that leak the label in the ISOT dataset."""
    text = _DATELINE_RE.sub("", text)
    return _REUTERS_RE.sub(" ", text)


def clean_text(text: str) -> str:
    """Normalise raw article text for TF-IDF vectorisation.

    Mirrors the ``wordopt`` function from the original notebook but applies the
    steps in an order that actually works (URLs/HTML are removed *before*
    punctuation is stripped, otherwise they are already mangled).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = _DIGIT_WORD_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def prepare(text: str, *, strip_leakage: bool = True) -> str:
    """Full preprocessing pipeline used by both training and inference."""
    if strip_leakage:
        text = strip_source_leakage(text or "")
    return clean_text(text)


def word_count(text: str) -> int:
    """Number of tokens left after cleaning (used for input validation)."""
    cleaned = clean_text(text)
    return len(cleaned.split()) if cleaned else 0
