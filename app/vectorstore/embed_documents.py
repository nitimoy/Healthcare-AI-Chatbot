"""
app/vectorstore/embed_documents.py
────────────────────────────────────
One-time indexing script.

Responsibility
──────────────
Load the verified chunked_documents.json produced by the ingestion pipeline,
embed each chunk with BAAI/bge-m3, and persist everything into ChromaDB.

This script is idempotent:
  - If the collection already contains exactly as many documents as there are
    chunks, indexing is skipped entirely.
  - Chunks are upserted in batches so the script is safe to re-run if it was
    interrupted mid-way.

Run
───
From the project root:

    PYTHONPATH=. python app/vectorstore/embed_documents.py

Expected output (first run, ~2,010 chunks):

    [14:00:01] INFO  Loading chunks from data/processed/chunked_documents.json
    [14:00:01] INFO  Loaded 2,010 chunks
    [14:00:02] INFO  Initialising ChromaDB at data/chroma
    [14:00:02] INFO  Loading embedding model BAAI/bge-m3 (first run downloads ~1.2 GB)
    [14:00:15] INFO  Starting indexing — 2,010 chunks in 32 batches of 64
    [14:00:16] INFO  Batch  1/32 — indexed chunks 1-64
    ...
    [14:02:30] INFO  Batch 32/32 — indexed chunks 1985-2010
    ════════════════════════════════════════════
    Indexing complete
    Total chunks in file   : 2,010
    Already in collection  :     0
    Newly indexed          : 2,010
    Collection total       : 2,010
    ════════════════════════════════════════════

Design decisions
────────────────
1. SentenceTransformerEmbeddingFunction:
   ChromaDB's built-in wrapper that calls sentence-transformers automatically.
   This guarantees the same embedding function is used at index time and query
   time, which is required for correct cosine similarity search.

2. Batched upsert (not add):
   collection.upsert() is idempotent — safe to call even if some IDs already
   exist. This allows partial re-runs without creating duplicates.

3. Metadata type conversion:
   ChromaDB metadata values must be str | int | float | bool.
   List fields (mesh, groups) are joined with the pipe character '|'.
   The retriever splits them back on read.

4. Early exit on exact count match:
   If collection.count() == len(chunks), all chunks are already present.
   The script exits without loading the model, saving time on repeated runs.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# ── Resolve project root so imports work regardless of cwd ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config.settings import (
    CHROMA_PATH,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from preprocessing.io import load_json

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_chroma_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert a chunk's metadata dict to ChromaDB-compatible flat types.

    ChromaDB rejects list values — convert them to pipe-delimited strings
    so they can be split back into lists by the retriever.

    Parameters
    ----------
    meta:
        The ``metadata`` sub-dict from a chunk record.

    Returns
    -------
    dict with only str, int, float, or bool values.
    """
    return {
        "title": str(meta.get("title", "")),
        "source": str(meta.get("source", "MedlinePlus")),
        "url": str(meta.get("url", "")),
        # Lists → pipe-delimited strings
        "groups": "|".join(meta.get("groups") or []),
        "mesh": "|".join(meta.get("mesh") or []),
        "chunk_index": int(meta.get("chunk_index", 1)),
        "total_chunks": int(meta.get("total_chunks", 1)),
        "primary_institute": str(meta.get("primary_institute", "")),
    }


def _batched(items: list, size: int):
    """Yield successive sublists of *size* from *items*."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ─────────────────────────────────────────────────────────────────────────────
# Core indexing logic
# ─────────────────────────────────────────────────────────────────────────────


def build_index(
    chunks_path: str = CHUNKS_PATH,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> None:
    """Embed and index all chunks into ChromaDB.

    Parameters
    ----------
    chunks_path:
        Path (relative to cwd or absolute) to chunked_documents.json.
    chroma_path:
        Directory where ChromaDB persists its data.
    collection_name:
        Name of the ChromaDB collection to create or reuse.
    embedding_model:
        HuggingFace model ID for sentence-transformers.
    batch_size:
        Number of chunks per ChromaDB upsert call.
    """
    # ── 1. Load chunks ────────────────────────────────────────────────────────
    abs_chunks = PROJECT_ROOT / chunks_path
    logger.info("Loading chunks from %s", abs_chunks)

    if not abs_chunks.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {abs_chunks}\n"
            "Run the ingestion pipeline first:\n"
            "  PYTHONPATH=. python scripts/chunk_documents.py"
        )

    chunks: list[dict[str, Any]] = load_json(str(abs_chunks))
    total_chunks = len(chunks)
    logger.info("Loaded %s chunks", f"{total_chunks:,}")

    # ── 2. Initialise ChromaDB ────────────────────────────────────────────────
    abs_chroma = PROJECT_ROOT / chroma_path
    abs_chroma.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising ChromaDB at %s", abs_chroma)

    client = chromadb.PersistentClient(path=str(abs_chroma))

    # ── 3. Load embedding function ────────────────────────────────────────────
    logger.info(
        "Loading embedding model %s (first run downloads ~1.2 GB)", embedding_model
    )
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        # BGE-M3 produces better results with these settings
        normalize_embeddings=True,
    )

    # ── 4. Get or create collection ───────────────────────────────────────────
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for retrieval
    )

    existing_count = collection.count()
    logger.info(
        "Collection '%s' — existing documents: %s", collection_name, f"{existing_count:,}"
    )

    # ── 5. Deduplication guard ────────────────────────────────────────────────
    if existing_count == total_chunks:
        _print_summary(total_chunks, existing_count, 0, collection.count())
        logger.info("Collection is already fully indexed — nothing to do.")
        return

    # ── 6. Determine which IDs are already indexed ────────────────────────────
    # Fetch all existing IDs so we can skip them during upsert.
    # For large collections this is more accurate than count-based dedup.
    newly_indexed = 0
    skipped = 0

    if existing_count > 0:
        logger.info("Fetching existing IDs to detect partial index...")
        existing = collection.get(include=[])  # IDs only — no embeddings/documents
        existing_ids: set[str] = set(existing["ids"])
        chunks_to_index = [c for c in chunks if c["chunk_id"] not in existing_ids]
        skipped = len(chunks) - len(chunks_to_index)
        logger.info(
            "%s chunks already indexed, %s to add",
            f"{skipped:,}",
            f"{len(chunks_to_index):,}",
        )
    else:
        chunks_to_index = chunks

    # ── 7. Batch upsert ───────────────────────────────────────────────────────
    batches = list(_batched(chunks_to_index, batch_size))
    total_batches = len(batches)

    if total_batches == 0:
        logger.info("No new chunks to index.")
    else:
        logger.info(
            "Starting indexing — %s chunks in %s batches of %s",
            f"{len(chunks_to_index):,}",
            total_batches,
            batch_size,
        )

    t_start = time.perf_counter()

    for batch_num, batch in enumerate(batches, start=1):
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = [_to_chroma_metadata(c["metadata"]) for c in batch]

        # Upsert = insert-or-update → idempotent
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        newly_indexed += len(batch)
        elapsed = time.perf_counter() - t_start
        rate = newly_indexed / elapsed if elapsed > 0 else 0

        logger.info(
            "Batch %s/%s — %s new chunks indexed  (%.1f chunks/s)",
            f"{batch_num:>{len(str(total_batches))}}",
            total_batches,
            f"{newly_indexed:,}",
            rate,
        )

    # ── 8. Final verification ─────────────────────────────────────────────────
    final_count = collection.count()
    _print_summary(total_chunks, skipped, newly_indexed, final_count)

    if final_count != total_chunks:
        logger.warning(
            "Expected %s documents in collection but found %s — "
            "re-run the script to complete indexing.",
            total_chunks,
            final_count,
        )
    else:
        logger.info("Indexing verified: collection count matches chunk file.")


def _print_summary(
    total: int, existing: int, indexed: int, collection_count: int
) -> None:
    """Print a formatted indexing summary to stdout."""
    sep = "═" * 44
    print(f"\n{sep}")
    print("Indexing summary")
    print(sep)
    print(f"  Total chunks in file   : {total:>6,}")
    print(f"  Already in collection  : {existing:>6,}")
    print(f"  Newly indexed          : {indexed:>6,}")
    print(f"  Collection total       : {collection_count:>6,}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        build_index()
    except FileNotFoundError as exc:
        logger.error("Setup error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Re-run to resume — upsert is idempotent.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error during indexing: %s", exc)
        sys.exit(1)
