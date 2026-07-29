"""
tests/test_voice.py — Unit tests for VoiceSession and clean_voice_transcript.
"""

from app.chatbot.voice import VoiceSession, clean_voice_transcript


def test_clean_voice_transcript_empty():
    assert clean_voice_transcript("") == ""
    assert clean_voice_transcript("   ") == ""


def test_clean_voice_transcript_filler_only():
    assert clean_voice_transcript("um... uh... er...") == ""
    assert clean_voice_transcript("uh um well you know") == ""


def test_clean_voice_transcript_leading_fillers():
    assert clean_voice_transcript("Uh... um... what are the symptoms of asthma?") == "What are the symptoms of asthma?"
    assert clean_voice_transcript("Well... you know... how is diabetes treated?") == "How is diabetes treated?"


def test_clean_voice_transcript_preserve_in_sentence_words():
    # In-sentence "like" and "so" MUST NOT be stripped
    assert clean_voice_transcript("What foods are like asthma triggers?") == "What foods are like asthma triggers?"
    assert clean_voice_transcript("So what causes diabetes?") == "What causes diabetes?"  # leading 'so' stripped, sentence capitalized


def test_clean_voice_transcript_punctuation_and_whitespace():
    assert clean_voice_transcript("Uh... what   are,,, the symptoms??") == "What are, the symptoms?"


def test_clean_voice_transcript_medical_terms_intact():
    assert clean_voice_transcript("What is COPD?") == "What is COPD?"
    assert clean_voice_transcript("What is acetaminophen?") == "What is acetaminophen?"
    assert clean_voice_transcript("Should I use my inhaler?") == "Should I use my inhaler?"


def test_voice_session_confidence():
    vs1 = VoiceSession(confidence=0.94)
    assert vs1.is_low_confidence is False

    vs2 = VoiceSession(confidence=0.45)
    assert vs2.is_low_confidence is True

    vs3 = VoiceSession(confidence=None)
    assert vs3.is_low_confidence is False
