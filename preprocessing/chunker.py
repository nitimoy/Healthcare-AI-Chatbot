"""
preprocessing/chunker.py
────────────────────────
Token-aware sliding-window chunker backed by semantic-text-splitter.

Why semantic-text-splitter instead of the old hand-rolled build_chunks?
  ┌────────────────────────────────────────────────────────────────────┐
  │ Old code                   │ This module                          │
  ├────────────────────────────┼──────────────────────────────────────┤
  │ Word count (split())       │ Exact tokenizer token count          │
  │ Overlap = sentence slice   │ Overlap = sliding token window       │
  │ Bug: overlap never shrinks │ Correct: always ≤ CHUNK_OVERLAP toks │
  │ ~120 lines of custom code  │ ~40 lines + well-tested Rust library  │
  └────────────────────────────┴──────────────────────────────────────┘

semantic-text-splitter (v0.32+):
  • Rust-backed — fast enough to process thousands of documents.
  • API: TextSplitter.from_huggingface_tokenizer(tokenizer, capacity, overlap)
  • Sentence-boundary respecting — does not cut mid-sentence.
  • Actively maintained (2024-present).

Default model: bert-base-uncased (512-token context window).
  → CHUNK_SIZE 384 tokens leaves ~128 tokens for the query at retrieval time.
  → CHUNK_OVERLAP 64 tokens ≈ 2-3 sentences of context continuity.

To switch models, change TOKENIZER_NAME and optionally CHUNK_SIZE /
CHUNK_OVERLAP at the top of this file or pass them via CLI flags in
scripts/chunk_documents.py.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TOKENIZER_NAME = "bert-base-uncased"

# Maximum tokens per chunk (header + summary window combined).
# 384 = safe budget for 512-token context models (BERT, MiniLM, etc.).
# Increase to ~1024 for models with larger context windows.
CHUNK_SIZE: int = 384

# Token overlap between consecutive chunks.
# 64 tokens ≈ 2-3 sentences — enough for retrieval continuity.
CHUNK_OVERLAP: int = 64


# ─────────────────────────────────────────────
# Internal cache holder
# ─────────────────────────────────────────────


class _SplitterBundle(NamedTuple):
    splitter: TextSplitter
    tokenizer: Tokenizer


@lru_cache(maxsize=1)
def _get_bundle() -> _SplitterBundle:
    """Load the tokenizer once and build the splitter.

    lru_cache means this is called only once per process, even when
    chunk_summary() is called thousands of times.

    Truncation is disabled on the tokenizer before passing it to
    semantic-text-splitter, as required by the library — otherwise the
    tokenizer's truncation limit silently caps chunk sizes.
    """
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.no_truncation()
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer,
        CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    return _SplitterBundle(splitter=splitter, tokenizer=tokenizer)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────


def chunk_summary(summary: str, header_tokens: int = 0) -> list[str]:
    """Split *summary* into token-aware sliding-window chunks.

    The summary is chunked against a budget of ``CHUNK_SIZE - header_tokens``
    so that when the header is prepended to each chunk the combined content
    stays within ``CHUNK_SIZE`` tokens.

    Parameters
    ----------
    summary:
        The free-text clinical summary to chunk. May be an empty string
        (for topics with no summary), in which case ``[""]`` is returned
        so that callers always receive at least one chunk.
    header_tokens:
        Token count of the header string that will be prepended to every chunk.
        Deducted from CHUNK_SIZE so the final assembled chunk never exceeds the
        configured limit.

    Returns
    -------
    list[str]
        One or more chunk strings, or ``[""]`` if *summary* is empty.
    """
    if not summary or not summary.strip():
        return [""]

    bundle = _get_bundle()
    effective_capacity = max(64, CHUNK_SIZE - header_tokens)

    if effective_capacity == CHUNK_SIZE:
        # No header adjustment needed — use cached splitter directly.
        splitter = bundle.splitter
    else:
        # Build an ephemeral splitter with the reduced budget.
        # This is cheap (Rust constructor) and avoids polluting the cache.
        splitter = TextSplitter.from_huggingface_tokenizer(
            bundle.tokenizer,
            effective_capacity,
            overlap=CHUNK_OVERLAP,
        )

    chunks = splitter.chunks(summary)
    return chunks if chunks else [""]


def count_tokens(text: str) -> int:
    """Return the exact token count for *text* using the configured tokenizer.

    Used by metadata.py to populate the ``token_count`` field.
    Truncation is disabled, so counts are accurate for any length of text.
    """
    tokenizer = _get_bundle().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False).ids)
