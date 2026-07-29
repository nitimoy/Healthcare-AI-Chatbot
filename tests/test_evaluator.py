"""
tests/test_evaluator.py
────────────────────────
Unit tests for EvidenceAssessmentEngine.
"""

from app.chatbot.evaluator import EvidenceAssessmentEngine
from app.chatbot.retriever import RetrievedChunk


def make_chunk(
    chunk_id: str = "doc1_001",
    document_id: str = "doc1",
    title: str = "Diabetes",
    content: str = "Diabetes is a chronic metabolic disease characterized by high blood glucose levels.",
    groups: list[str] = None,
    mesh: list[str] = None,
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        url=f"https://medlineplus.gov/{title.lower().replace(' ', '')}.html",
        source="MedlinePlus",
        groups=groups or ["Endocrine System", "Diabetes Mellitus"],
        mesh=mesh or ["Diabetes Mellitus"],
        primary_institute="NIDDK",
        chunk_index=1,
        total_chunks=1,
        distance=2.0 * (1.0 - score),
    )


def test_rule1_no_chunks():
    engine = EvidenceAssessmentEngine()
    is_valid, reason = engine.evaluate("What is diabetes?", [])
    assert not is_valid
    assert "No chunks" in reason


def test_rule2_empty_content():
    engine = EvidenceAssessmentEngine()
    chunk = make_chunk(content="   ")
    is_valid, reason = engine.evaluate("What is diabetes?", [chunk])
    assert not is_valid
    assert "content volume" in reason.lower()


def test_rule3_cohesion_pass():
    engine = EvidenceAssessmentEngine()
    chunks = [
        make_chunk(chunk_id="c1", document_id="doc1", title="Diabetes"),
        make_chunk(chunk_id="c2", document_id="doc1", title="Diabetes"),
        make_chunk(chunk_id="c3", document_id="doc2", title="Diabetes Type 2"),
    ]
    is_valid, reason = engine.evaluate("What is diabetes?", chunks)
    assert is_valid
    assert "passed" in reason.lower()


def test_rule3_cohesion_fail():
    engine = EvidenceAssessmentEngine()
    # 5 completely unrelated chunks with no group/doc overlap
    chunks = [
        make_chunk(chunk_id="c1", document_id="doc1", title="Diabetes", groups=["Endocrine"]),
        make_chunk(chunk_id="c2", document_id="doc2", title="Hypertension", groups=["Cardiovascular"]),
        make_chunk(chunk_id="c3", document_id="doc3", title="Burns", groups=["Skin"]),
        make_chunk(chunk_id="c4", document_id="doc4", title="Lung Cancer", groups=["Oncology"]),
        make_chunk(chunk_id="c5", document_id="doc5", title="Menopause", groups=["Reproductive"]),
    ]
    is_valid, reason = engine.evaluate("How to repair bicycle tire?", chunks)
    assert not is_valid
    assert "cohesion" in reason.lower()


def test_rule4_metadata_alignment_fail():
    engine = EvidenceAssessmentEngine()
    chunks = [
        make_chunk(chunk_id="c1", title="Diabetes Mellitus", groups=["Endocrine System"], mesh=["Diabetes"], score=0.72),
    ]
    # Query with non-stopword keywords having zero overlap and low score (0.72 < 0.78)
    is_valid, reason = engine.evaluate("bicycle tire repair transmission chain", chunks)
    assert not is_valid
    assert "metadata alignment" in reason.lower()


def test_rule5_diversity_deduplication():
    engine = EvidenceAssessmentEngine()
    chunks = [
        make_chunk(chunk_id="c1", title="Diabetes"),
        make_chunk(chunk_id="c1", title="Diabetes"),  # Duplicate chunk_id
    ]
    is_valid, reason = engine.evaluate("What is diabetes?", chunks)
    assert is_valid  # Still valid, but logs warning


def test_mixed_topic_cohesion_fail():
    """Verify that broad queries returning scattered medical topics fail Evidence Cohesion."""
    engine = EvidenceAssessmentEngine()
    chunks = [
        make_chunk(chunk_id="c1", document_id="doc1", title="Burns", groups=["Injuries"]),
        make_chunk(chunk_id="c2", document_id="doc2", title="Diabetes Mellitus", groups=["Endocrine"]),
        make_chunk(chunk_id="c3", document_id="doc3", title="Pregnancy", groups=["Reproductive"]),
        make_chunk(chunk_id="c4", document_id="doc4", title="High Blood Pressure", groups=["Cardiovascular"]),
    ]
    is_valid, reason = engine.evaluate("What foods should I avoid?", chunks)
    assert not is_valid
    assert "cohesion" in reason.lower()


def test_guardrail_poisoning():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("My child ingested bleach poison")
    assert resp is not None
    assert "emergency" in resp.lower()


def test_guardrail_multi_condition_diet():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("Is keto diet safe for someone with diabetes and kidney disease?")
    assert resp is not None
    assert "cannot recommend specific diets" in resp.lower()


def test_guardrail_medication_modification_stop():
    from app.chatbot.guardrails import check_and_respond
    resp1 = check_and_respond("Can I stop taking my blood pressure medicine?")
    assert resp1 is not None
    assert "cannot advise" in resp1.lower()

    resp2 = check_and_respond("Should I stop chemotherapy?")
    assert resp2 is not None
    assert "cannot advise" in resp2.lower()

    resp3 = check_and_respond("Should I increase my insulin dose?")
    assert resp3 is not None
    assert "cannot advise" in resp3.lower()


def test_guardrail_disease_prediction():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("Will I develop cancer?")
    assert resp is not None
    assert "cannot predict" in resp.lower()


def test_guardrail_pediatric_breathing_emergency():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("My baby is struggling to breathe.")
    assert resp is not None
    assert "MEDICAL EMERGENCY DETECTED" in resp


def test_guardrail_unresponsive_emergency():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("My child is unresponsive and won't wake up.")
    assert resp is not None
    assert "MEDICAL EMERGENCY DETECTED" in resp


def test_guardrail_anaphylaxis_emergency():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("My throat is closing from a severe allergic reaction.")
    assert resp is not None
    assert "MEDICAL EMERGENCY DETECTED" in resp


def test_guardrail_heavy_bleeding_emergency():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("Heavy bleeding after an accident, blood is spurting.")
    assert resp is not None
    assert "MEDICAL EMERGENCY DETECTED" in resp


def test_guardrail_treatment_substitution_refusal():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("Can I give my child herbal medicine instead of an inhaler?")
    assert resp is not None
    assert "general health education only" in resp.lower() or "not able to answer" in resp.lower()


def test_guardrail_drug_interaction_refusal():
    from app.chatbot.guardrails import check_and_respond
    resp = check_and_respond("Can I take ibuprofen with blood pressure medicine?")
    assert resp is not None
    assert "general health education only" in resp.lower() or "not able to answer" in resp.lower()


def test_guardrail_negative_regression_general_questions():
    """Verify that general educational questions pass guardrails cleanly without false positives."""
    from app.chatbot.guardrails import check_and_respond
    assert check_and_respond("What is chemotherapy?") is None
    assert check_and_respond("What is insulin?") is None
    assert check_and_respond("What causes cancer?") is None
    assert check_and_respond("What is the ketogenic diet?") is None
    assert check_and_respond("What medicines are commonly used for asthma?") is None
    assert check_and_respond("Is paracetamol safer than ibuprofen for most adults?") is None
    assert check_and_respond("Why can NSAIDs interact with blood pressure medicines?") is None

