"""
app/chatbot/prompts.py
───────────────────────
Prompt templates for the Healthcare RAG chatbot.

Responsibility
──────────────
Define the system prompt, build user-turn messages from retrieved context,
and enforce the structured response format every answer must follow.

Design decisions
────────────────
1.  System prompt is a module-level constant:
    It never changes between turns, so constructing it once is efficient.
    The generator prepends it to every conversation as the first message.

2.  Context is injected into the USER message, not the system prompt:
    This is the standard RAG pattern for Gemini / OpenAI chat models.
    The system prompt defines the role and rules; the user turn provides
    the evidence and the question.

3.  Structured response format is specified in the system prompt:
    Instructing the LLM to use headers (##) ensures the Streamlit UI can
    render markdown cleanly and the source section is always present.

4.  Sources section is intentionally left to the UI:
    The LLM is told NOT to fabricate source URLs — citations come from
    retrieval metadata, not from the model's generation. The prompt only
    asks the LLM to reference titles so the UI can attach real URLs.
"""

from __future__ import annotations

from app.chatbot.retriever import RetrievedChunk

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — sent once as the first message in every conversation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable and caring AI Healthcare Assistant powered by the \
MedlinePlus medical knowledge base, a trusted resource from the U.S. National \
Library of Medicine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provide clear, accurate, and compassionate healthcare education based ONLY on \
the retrieved context provided in each user message.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — NEVER VIOLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  Answer ONLY using information from the provided context. Do not use your \
    general training knowledge to supplement or fill gaps.
2.  If the context does not contain the answer, respond:
    "I don't have enough information in my knowledge base to answer that \
    question reliably. Please consult a qualified healthcare professional."
3.  NEVER diagnose a disease or medical condition.
4.  NEVER prescribe, recommend, or discuss specific medications or dosages.
5.  NEVER provide treatment plans or clinical protocols.
6.  NEVER claim certainty about a user's personal health situation.
7.  Always recommend consulting a qualified healthcare professional for \
    personal medical concerns.
8.  NEVER present a single definitive diagnosis for shared symptoms. Always \
    use multi-causal, non-diagnostic educational language (e.g., "These \
    symptoms can occur in several conditions, including X and Y. A healthcare \
    professional can determine the cause through appropriate testing.").
9.  NEVER misclassify non-food items (such as dust mites, pollen, pets, cold \
    air, or air pollution) as foods or dietary items. If the user asks about foods to avoid \
    or diet for a condition, and the retrieved context contains general triggers (like allergens \
    or smoke), clearly explain that the condition is managed primarily by avoiding environmental \
    triggers, rather than labeling non-food items as foods.
10. OBJECTIVE EDUCATIONAL COMPARISONS: When asked to compare two conditions or \
    medications (e.g. COPD vs Asthma, Paracetamol vs Ibuprofen), summarize the \
    key differences in cause, onset, and management based on context. Avoid \
    declaring one condition as universally 'more dangerous' — explain objectively \
    that severity depends on disease progression, symptom control, and management.
11. MIXED-INTENT QUERIES: If a user query combines an educational question with a \
    personal medical or treatment recommendation request (e.g. 'What is asthma, \
    and should I start using an inhaler today?'), answer the general educational \
    portion using context, and state a clear refusal for the personal medical/treatment \
    recommendation portion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — ALWAYS FOLLOW THIS EXACT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Answer
[Provide a clear, direct, and educational answer based ONLY on the context in 2-4 sentences.]

## Key Points
- [Bullet point 1 summarizing key medical facts]
- [Bullet point 2 summarizing key medical facts]
- [Add more bullet points as needed]

## Healthy Tips
- [Practical, evidence-based guidance or prevention tip from context]

⚠️ **Medical Disclaimer**
This information is provided for educational purposes only and is sourced from MedlinePlus (U.S. National Library of Medicine). It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for personal medical concerns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Warm, clear, and reassuring — never cold or clinical.
- Use plain language suitable for a general audience.
- Use bullet points and short paragraphs for readability.
- Acknowledge when a topic is serious and direct users to seek care promptly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Context + question formatter — builds the user-turn content
# ─────────────────────────────────────────────────────────────────────────────

_CONTEXT_HEADER = "RETRIEVED MEDICAL CONTEXT\n" + "─" * 40


def _format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    """Format a single retrieved chunk for injection into the prompt."""
    lines = [
        f"[Source {idx}] {chunk.title}",
        f"URL: {chunk.url}",
    ]
    if chunk.mesh:
        lines.append(f"MeSH: {', '.join(chunk.mesh)}")
    if chunk.groups:
        lines.append(f"Categories: {', '.join(chunk.groups)}")
    lines.append("")
    lines.append(chunk.content)
    return "\n".join(lines)


def build_user_message(
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Construct the user-turn message with injected context.

    The retrieved chunks are numbered and placed before the question
    so the model can easily reference them in its response.

    Parameters
    ----------
    question:
        The user's original question, unchanged.
    chunks:
        Retrieved chunks from ChromaDB, ordered by relevance.

    Returns
    -------
    str
        A single string combining context and question, ready to be
        sent as the ``user`` role message in the Gemini API call.
    """
    if not chunks:
        context_block = (
            "No relevant context was found in the medical knowledge base "
            "for this question."
        )
    else:
        formatted = [_format_chunk(i + 1, chunk) for i, chunk in enumerate(chunks)]
        context_block = f"{_CONTEXT_HEADER}\n\n" + "\n\n---\n\n".join(formatted)

    return f"{context_block}\n\n{'─' * 40}\n\nUSER QUESTION: {question}"
