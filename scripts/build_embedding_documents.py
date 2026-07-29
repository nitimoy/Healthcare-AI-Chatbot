#!/usr/bin/env python3
"""
scripts/build_embedding_documents.py
─────────────────────────────────────
Convert the canonical knowledge base to RAG-ready embedding documents.

Pipeline position:

    data/raw/medical_kb_raw.json
            ↓
    scripts/build_embedding_documents.py
            ↓
    data/processed/embedding_documents.json

This script remains in the pipeline for the embedding document format
(the concatenated natural-language string + embedding metadata), which is
consumed by chunk_documents.py.

Changes from the original:
  • Uses preprocessing.io for JSON I/O (orjson).
  • Uses preprocessing.splitter to build the header/footer/summary — the
    same logic used by the chunker, ensuring consistency between the two.
  • Removes duplicated join_lines / bullet_list helpers (now in splitter.py).
  • Removed the unused `typing.Any` / `LANGUAGE` / `DOCUMENT_TYPE` constants
    that were only used to annotate fields which no longer need them.

Usage:
    python scripts/build_embedding_documents.py
"""

from __future__ import annotations

import time
from pathlib import Path

from preprocessing.io import load_json, save_json
from preprocessing.splitter import split

INPUT_PATH = Path("data/raw/medical_kb_raw.json")
OUTPUT_PATH = Path("data/processed/embedding_documents.json")


def build_document(record: dict) -> dict:
    """Build a single embedding document from a RawRecord dict."""
    parts = split(record)

    # Assemble the full natural-language text (header + summary + footer).
    # This is the text that gets embedded — no chunking yet.
    sections = [s for s in [parts.header, parts.summary, parts.footer] if s]
    content = "\n\n".join(sections)

    metadata = {
        "id": record["id"],
        "title": record["title"],
        "document_type": "medical_topic",
        "language": "en",
        "source": "MedlinePlus",
        "url": record["url"],
        "groups": [g["name"] for g in record["groups"] if g.get("name")],
        "mesh": [m["name"] for m in record["mesh_headings"] if m.get("name")],
        "primary_institute": record["primary_institute"].get("name", ""),
    }

    return {
        "id": record["id"],
        "content": content,
        "metadata": metadata,
    }


def run() -> None:
    print(f"Loading  : {INPUT_PATH}")
    t0 = time.perf_counter()
    records = load_json(INPUT_PATH)
    print(f"Records  : {len(records):,}  ({time.perf_counter() - t0:.1f}s)")

    embedding_docs = [build_document(r) for r in records]

    save_json(OUTPUT_PATH, embedding_docs)
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"Saved    : {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    print("\n── Sample Document ────────────────────────────────────────")
    import orjson
    print(orjson.dumps(embedding_docs[0], option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    run()