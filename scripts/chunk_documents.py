#!/usr/bin/env python3
"""
scripts/chunk_documents.py
──────────────────────────
Entry point: load raw records → split → chunk → save.

Pipeline position:

    data/raw/medical_kb_raw.json
            ↓
    scripts/chunk_documents.py
            ↓
    data/processed/chunked_documents.json

This script is intentionally thin. All logic lives in the preprocessing/
package so it can be unit-tested and reused independently.

Usage:
    python scripts/chunk_documents.py
    python scripts/chunk_documents.py --input data/raw/medical_kb_raw.json
    python scripts/chunk_documents.py --chunk-size 512 --overlap 96
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from preprocessing import chunker as chunker_mod
from preprocessing.io import load_json, save_json
from preprocessing.metadata import build_chunk_record
from preprocessing.splitter import split

# ─────────────────────────────────────────────
# Defaults (override via CLI flags)
# ─────────────────────────────────────────────

DEFAULT_INPUT = Path("data/raw/medical_kb_raw.json")
DEFAULT_OUTPUT = Path("data/processed/chunked_documents.json")


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────


def run(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    overlap: int,
) -> None:
    # Allow CLI to override chunker config before the splitter is instantiated.
    chunker_mod.CHUNK_SIZE = chunk_size
    chunker_mod.CHUNK_OVERLAP = overlap

    # Import after config mutation so lru_cache picks up the new values.
    from preprocessing.chunker import chunk_summary, count_tokens

    print(f"Loading  : {input_path}")
    t0 = time.perf_counter()
    records = load_json(input_path)
    print(f"Records  : {len(records):,}  ({time.perf_counter() - t0:.1f}s)")

    print(f"\nTokenizer: {chunker_mod.TOKENIZER_NAME}")
    print(f"Chunk size: {chunk_size} tokens  |  Overlap: {overlap} tokens\n")

    chunks: list[dict] = []

    for record in records:
        parts = split(record)
        header_tokens = count_tokens(parts.header) if parts.header else 0
        footer_tokens = count_tokens(parts.footer) if parts.footer else 0

        # The last chunk carries the footer, so its summary window must be smaller.
        # All earlier chunks only carry the header.
        # We chunk once using the stricter of the two budgets to keep things simple
        # and deterministic — the small extra margin on non-last chunks is negligible.
        reserved = header_tokens + footer_tokens
        summary_chunks = chunk_summary(parts.summary, header_tokens=reserved)
        total = len(summary_chunks)

        for idx, summary_window in enumerate(summary_chunks, start=1):
            # Header prepended to every chunk for self-containment.
            # Footer appended only to the last chunk.
            content_parts = [parts.header, summary_window]
            if idx == total and parts.footer:
                content_parts.append(parts.footer)

            content = "\n\n".join(p for p in content_parts if p)

            chunks.append(
                build_chunk_record(
                    record=record,
                    content=content,
                    chunk_index=idx,
                    total_chunks=total,
                )
            )

    print(f"Saving   : {output_path}")
    t1 = time.perf_counter()
    save_json(output_path, chunks)
    elapsed = time.perf_counter() - t1

    # ── Report ──────────────────────────────────────────
    size_mb = output_path.stat().st_size / 1_048_576
    token_counts = [c["token_count"] for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    print()
    print("=" * 60)
    print(f"{'Input records':<30} {len(records):>10,}")
    print(f"{'Output chunks':<30} {len(chunks):>10,}")
    print(f"{'Avg chunks / document':<30} {len(chunks)/len(records):>10.2f}")
    print(f"{'Avg tokens / chunk':<30} {avg_tokens:>10.1f}")
    print(f"{'Output size':<30} {size_mb:>9.1f} MB")
    print(f"{'Write time':<30} {elapsed:>9.2f}s")
    print("=" * 60)

    # Sample output for spot-checking
    print("\nSample chunk (first):\n")
    import orjson
    print(orjson.dumps(chunks[0], option=orjson.OPT_INDENT_2).decode())


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk medical knowledge base into RAG-ready documents.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=chunker_mod.CHUNK_SIZE,
        help="Maximum tokens per chunk (default: %(default)s)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=chunker_mod.CHUNK_OVERLAP,
        help="Token overlap between consecutive chunks (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )