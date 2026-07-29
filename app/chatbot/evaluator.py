"""
app/chatbot/evaluator.py
─────────────────────────
EvidenceAssessmentEngine — Deterministic Evidence Assessment Layer.

Responsibility
──────────────
Performs lightweight, deterministic validation on retrieved candidate chunks
before calling the LLM. Eliminates redundant LLM calls, costs, and latency,
ensuring that only topic-aligned, cohesive medical evidence is sent to generation.

If validation fails (e.g. out-of-domain queries, fragmented random noise),
the system returns a safe, trustworthy fallback message without making an LLM call.

Design decisions
────────────────
1.  100% Deterministic (Zero LLM calls):
    Uses metadata inspection and heuristic checks with negligible computational overhead.

2.  5 Evidence Rules:
    - Rule 1: Retrieval Presence (len(chunks) > 0)
    - Rule 2: Content Availability (chunks contain non-empty, substantive text)
    - Rule 3: Evidence Cohesion (majority of chunks belong to a coherent primary topic or shared categories)
    - Rule 4: Weighted Metadata Alignment (normalized query token overlap against title/MeSH/groups)
    - Rule 5: Retrieval Diversity (unique chunk_ids without redundant duplicate copies)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.chatbot.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Common non-informative stop words to ignore during metadata token matching
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
    "through", "about", "against", "between", "into", "throughout", "during", "before",
    "after", "above", "below", "to", "from", "up", "upon", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "should", "now", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "do", "does", "did", "have", "has", "had"
}


def _tokenize(text: str) -> set[str]:
    """Extract normalized alphanumeric tokens from text, excluding stop words."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


class EvidenceAssessmentEngine:
    """Deterministic Evidence Assessment Layer for Healthcare RAG.

    Validates candidate chunks before LLM generation using 5 deterministic rules.
    """

    def evaluate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[bool, str]:
        """Perform deterministic evidence assessment on retrieved chunks.

        Parameters
        ----------
        question:
            The user's input question (or search query).
        chunks:
            Retrieved candidate chunks from ChromaDB.

        Returns
        -------
        tuple[bool, str]
            (is_valid, reason) tuple.
            is_valid is True if evidence passes all deterministic rules.
            reason provides diagnostic detail for logging.
        """
        # ── Rule 1: Retrieval Presence ────────────────────────────────────────
        if not chunks:
            logger.info("Evidence Assessment FAIL: Rule 1 (No chunks retrieved)")
            return False, "No chunks retrieved from vector store"

        # ── Rule 2: Content Availability ──────────────────────────────────────
        has_content = any(
            c.content and len(c.content.strip()) > 20 for c in chunks
        )
        if not has_content:
            logger.info("Evidence Assessment FAIL: Rule 2 (Content volume sparse/empty)")
            return False, "Retrieved content volume is empty or insufficient"

        # ── Rule 3: Evidence Cohesion ─────────────────────────────────────────
        # Ensure retrieved chunks predominantly originate from a coherent primary medical topic or shared categories
        primary_doc = chunks[0].document_id
        primary_title = chunks[0].title
        primary_groups = set(chunks[0].groups)

        coherent_count = 0
        for c in chunks:
            if (
                c.document_id == primary_doc
                or c.title == primary_title
                or (set(c.groups) & primary_groups)
            ):
                coherent_count += 1

        cohesion_ratio = coherent_count / len(chunks)
        # If less than 40% of retrieved chunks share the primary topic or group, evidence is fragmented
        if cohesion_ratio < 0.4:
            logger.info(
                "Evidence Assessment FAIL: Rule 3 (Evidence Cohesion ratio %.2f < 0.40)",
                cohesion_ratio,
            )
            return False, f"Retrieved evidence lacks topic cohesion (ratio: {cohesion_ratio:.2f})"

        # ── Rule 4: Weighted Metadata Alignment ─────────────────────────────
        # Compare query tokens against titles, MeSH terms, and category groups.
        # Multiple metadata matches increase confidence, while lack of overlap combined
        # with low similarity (<0.78) indicates weak evidence alignment.
        query_tokens = _tokenize(question)
        if query_tokens and chunks:
            metadata_text = []
            for c in chunks:
                metadata_text.append(c.title)
                metadata_text.extend(c.mesh)
                metadata_text.extend(c.groups)

            all_metadata_tokens = _tokenize(" ".join(metadata_text))
            overlap = query_tokens & all_metadata_tokens

            top_score = chunks[0].similarity_score
            if not overlap and top_score < 0.78:
                logger.info(
                    "Evidence Assessment FAIL: Rule 4 (Zero metadata alignment and low score %.2f < 0.78)",
                    top_score,
                )
                return False, f"Weak metadata alignment (top score: {top_score:.2f})"

        # ── Rule 5: Retrieval Diversity & Quality ────────────────────────────
        unique_ids = {c.chunk_id for c in chunks}
        if len(unique_ids) < len(chunks):
            logger.warning(
                "Evidence Assessment WARNING: Rule 5 (Duplicate chunk IDs detected: %s unique / %s total)",
                len(unique_ids),
                len(chunks),
            )

        logger.info(
            "Evidence Assessment PASS: Verified %s chunks | primary_topic=%r | cohesion=%.0f%%",
            len(chunks),
            primary_title,
            cohesion_ratio * 100,
        )
        return True, "Evidence assessment passed successfully"
