"""
app/chatbot/voice.py
─────────────────────
Speech-to-Text Voice Session model and conservative voice transcript sanitizer.

Responsibility
──────────────
1. VoiceSession dataclass to model speech state machine independently from UI.
2. Conservative transcript sanitizer that strips ONLY leading sentence-start
   hesitation fillers ('uh', 'um', 'well', 'you know') while preserving
   all in-sentence words ('like', 'so') and medical terms ('COPD', 'acetaminophen').
3. Normalises excessive punctuation and whitespace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VOICE_CONFIDENCE_THRESHOLD: float = 0.60

# Leading hesitation fillers at the start of a sentence (case-insensitive)
_LEADING_FILLERS_RE = re.compile(
    r"^(?:(?:uh+|um+|er+|ah+|well|you\s+know|so)\b[\s,.\-]*)+",
    re.IGNORECASE,
)

# Excess punctuation and whitespace cleanup
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_PUNCTUATION_RE = re.compile(r"([,.?!])\1+")


@dataclass
class VoiceSession:
    """Decoupled model representing the Voice Input state machine."""
    state: str = "Idle"  # Idle, Listening, Processing, Recognised, PermissionDenied, NoSpeechDetected, BrowserUnsupported, RecognitionFailed, Timeout
    raw_transcript: str = ""
    clean_transcript: str = ""
    confidence: float | None = None
    language: str = "en-US"
    browser_supported: bool = True
    error: str | None = None

    @property
    def is_low_confidence(self) -> bool:
        if self.confidence is None:
            return False
        return self.confidence < VOICE_CONFIDENCE_THRESHOLD


def clean_voice_transcript(raw_text: str) -> str:
    """Sanitise voice transcript conservatively.

    Strips ONLY leading hesitation fillers (e.g. 'Uh... um... what is COPD?') at the start of
    the transcript, normalises punctuation and spacing, and leaves all medical terminology
    and in-sentence words ('like', 'so') completely intact.

    Returns empty string for filler-only input ('um... uh... er...').
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # 1. Strip leading hesitation fillers at sentence start
    text = _LEADING_FILLERS_RE.sub("", text).strip()

    if not text:
        return ""

    # 2. Fix repeated punctuation (e.g. 'what   are,,, the symptoms??' -> 'what are, the symptoms?')
    text = _MULTI_PUNCTUATION_RE.sub(r"\1", text)

    # 3. Fix multiple spaces
    text = _MULTI_WHITESPACE_RE.sub(" ", text).strip()

    # 4. Capitalise first letter if sentence start
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text
