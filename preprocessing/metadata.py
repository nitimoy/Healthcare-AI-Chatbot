"""
preprocessing/metadata.py
─────────────────────────
Builds the standardised output metadata dict for every chunk.

Centralising metadata construction here means:
  • The output schema is defined in exactly one place.
  • Adding or removing a field requires one change, not hunting through scripts.
  • token_count / word_count / char_count are computed consistently.
"""

from __future__ import annotations

from typing import Any

from preprocessing.chunker import count_tokens
from preprocessing.parser import RawRecord


def build_chunk_record(
    *,
    record: RawRecord,
    content: str,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    """Construct a single output chunk dict.

    Parameters
    ----------
    record:
        The source RawRecord this chunk belongs to.
    content:
        The fully assembled chunk text (header + summary_window [+ footer]).
    chunk_index:
        1-based position of this chunk within its document.
    total_chunks:
        Total number of chunks produced for this document.

    Returns
    -------
    dict matching the output schema:

        chunk_id      : "{doc_id}_chunk_{index:03}"
        document_id   : str
        content       : str
        token_count   : int  (exact, using the configured tokenizer)
        word_count    : int
        char_count    : int
        metadata      : dict
    """
    doc_id = record["id"]
    chunk_id = f"{doc_id}_chunk_{chunk_index:03}"

    token_count = count_tokens(content)
    word_count = len(content.split())
    char_count = len(content)

    metadata: dict[str, Any] = {
        "title": record["title"],
        "source": "MedlinePlus",
        "url": record["url"],
        "groups": [g["name"] for g in record["groups"] if g.get("name")],
        "mesh": [m["name"] for m in record["mesh_headings"] if m.get("name")],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "primary_institute": record["primary_institute"].get("name", ""),
    }

    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "content": content,
        "token_count": token_count,
        "word_count": word_count,
        "char_count": char_count,
        "metadata": metadata,
    }
