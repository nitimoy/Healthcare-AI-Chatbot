"""
tests/run_tests.py — Direct test runner for EvidenceAssessmentEngine.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_evaluator import (
    test_rule1_no_chunks,
    test_rule2_empty_content,
    test_rule3_cohesion_pass,
    test_rule3_cohesion_fail,
    test_rule4_metadata_alignment_fail,
    test_rule5_diversity_deduplication,
    test_mixed_topic_cohesion_fail,
    test_guardrail_poisoning,
    test_guardrail_multi_condition_diet,
    test_guardrail_medication_modification_stop,
    test_guardrail_disease_prediction,
    test_guardrail_pediatric_breathing_emergency,
    test_guardrail_unresponsive_emergency,
    test_guardrail_anaphylaxis_emergency,
    test_guardrail_heavy_bleeding_emergency,
    test_guardrail_treatment_substitution_refusal,
    test_guardrail_drug_interaction_refusal,
    test_guardrail_negative_regression_general_questions,
    test_conversational_greetings,
    test_conversational_gratitude,
)

def run():
    print("🧪 Running Final Modular Safety & Regression Test Suite...")
    test_rule1_no_chunks()
    print("  ✅ Rule 1 (No Chunks) PASSED")
    test_rule2_empty_content()
    print("  ✅ Rule 2 (Content Availability) PASSED")
    test_rule3_cohesion_pass()
    print("  ✅ Rule 3 (Evidence Cohesion Pass) PASSED")
    test_rule3_cohesion_fail()
    print("  ✅ Rule 3 (Evidence Cohesion Fail) PASSED")
    test_rule4_metadata_alignment_fail()
    print("  ✅ Rule 4 (Metadata Alignment Fail) PASSED")
    test_rule5_diversity_deduplication()
    print("  ✅ Rule 5 (Diversity & Quality) PASSED")
    test_mixed_topic_cohesion_fail()
    print("  ✅ Mixed-Topic Cohesion Fail PASSED")
    test_guardrail_poisoning()
    print("  ✅ Poisoning Immediate Emergency PASSED")
    test_guardrail_multi_condition_diet()
    print("  ✅ Multi-Condition Diet Refusal PASSED")
    test_guardrail_medication_modification_stop()
    print("  ✅ Medication Modification Stop/Dose Refusal PASSED")
    test_guardrail_disease_prediction()
    print("  ✅ Disease Risk Prediction Refusal PASSED")
    test_guardrail_pediatric_breathing_emergency()
    print("  ✅ Pediatric Respiratory Distress Emergency PASSED")
    test_guardrail_unresponsive_emergency()
    print("  ✅ Unresponsive / Unconscious Emergency PASSED")
    test_guardrail_anaphylaxis_emergency()
    print("  ✅ Anaphylaxis & Severe Allergy Emergency PASSED")
    test_guardrail_heavy_bleeding_emergency()
    print("  ✅ Heavy Bleeding Emergency PASSED")
    test_guardrail_treatment_substitution_refusal()
    print("  ✅ Treatment Substitution Refusal PASSED")
    test_guardrail_drug_interaction_refusal()
    print("  ✅ Personal Drug Interaction Refusal PASSED")
    test_guardrail_negative_regression_general_questions()
    print("  ✅ Negative Regression (General Educational Questions Allowed) PASSED")
    test_conversational_greetings()
    print("  ✅ Conversational Greetings & Capabilities PASSED")
    test_conversational_gratitude()
    print("  ✅ Conversational Gratitude & Closing PASSED")


    print("\n🎤 Running Voice Sanitizer & Session Model Test Suite...")
    from tests.test_voice import (
        test_clean_voice_transcript_empty,
        test_clean_voice_transcript_filler_only,
        test_clean_voice_transcript_leading_fillers,
        test_clean_voice_transcript_preserve_in_sentence_words,
        test_clean_voice_transcript_punctuation_and_whitespace,
        test_clean_voice_transcript_medical_terms_intact,
        test_voice_session_confidence,
    )
    test_clean_voice_transcript_empty()
    test_clean_voice_transcript_filler_only()
    test_clean_voice_transcript_leading_fillers()
    test_clean_voice_transcript_preserve_in_sentence_words()
    test_clean_voice_transcript_punctuation_and_whitespace()
    test_clean_voice_transcript_medical_terms_intact()
    test_voice_session_confidence()
    print("  ✅ Voice Sanitizer & Medical Term Preservation PASSED")

    print("\n🎉 ALL 25 UNIT, SAFETY & VOICE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run()
