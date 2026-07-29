"""
app/chatbot/retriever.py
────────────────────────
ChromaDB retrieval layer for the Healthcare RAG pipeline.

Responsibility
──────────────
Accept a plain-English user query, embed it with the same BAAI/bge-m3
model used at index time, query ChromaDB for the Top-K most similar
chunks, and return typed RetrievedChunk objects with all metadata.

Usage
─────
    from app.chatbot.retriever import Retriever

    retriever = Retriever()                    # connects once, reuses
    results   = retriever.retrieve("What are the symptoms of diabetes?")

    for chunk in results:
        print(chunk.title, chunk.distance)
        print(chunk.content[:200])

Design decisions
────────────────
1.  Same embedding function as indexing:
    SentenceTransformerEmbeddingFunction(BAAI/bge-m3) must be identical
    to the function used in embed_documents.py. ChromaDB uses this to
    embed the query before running the HNSW similarity search.

2.  Singleton pattern (module-level instance):
    Loading the BGE-M3 model takes ~2-3 seconds. The Retriever class
    initialises lazily (on first retrieve() call) and caches the
    collection in an instance variable so subsequent calls are fast.

3.  Pipe-delimited metadata → lists:
    mesh and groups were stored as pipe-joined strings to satisfy
    ChromaDB's flat-type requirement. This module splits them back.

4.  RetrievedChunk dataclass:
    Typed return value prevents the generator from silently accessing
    wrong dict keys. Fields mirror the metadata stored by embed_documents.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Thread safety fixes for macOS / PyTorch / HuggingFace Tokenizers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config.settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievedChunk:
    """A single document chunk returned from ChromaDB.

    Attributes
    ----------
    content:
        Full assembled chunk text (header + summary window [+ footer]).
    chunk_id:
        Unique identifier, e.g. ``"6308_chunk_001"``.
    document_id:
        Parent document ID, e.g. ``"6308"``.
    title:
        Human-readable medical topic title, e.g. ``"Diabetes"``.
    url:
        Source MedlinePlus URL for citation.
    source:
        Always ``"MedlinePlus"`` for this knowledge base.
    groups:
        Medical category list, e.g. ``["Diabetes Mellitus", "Diagnostic Tests"]``.
    mesh:
        MeSH heading list, e.g. ``["Glycated Hemoglobin"]``. May be empty.
    primary_institute:
        NIH institute responsible for this topic.
    chunk_index:
        1-based position within the parent document.
    total_chunks:
        Total chunks in the parent document.
    distance:
        Cosine distance from the query (lower = more similar).
        Range: 0.0 (identical) – 2.0 (opposite).
    """

    content: str
    chunk_id: str
    document_id: str
    title: str
    url: str
    source: str
    groups: list[str]
    mesh: list[str]
    primary_institute: str
    chunk_index: int
    total_chunks: int
    distance: float

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to a 0-1 similarity score."""
        return max(0.0, 1.0 - self.distance / 2.0)

    @property
    def is_last_chunk(self) -> bool:
        """True if this chunk is the last in its parent document."""
        return self.chunk_index == self.total_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _split_pipe(value: str | None) -> list[str]:
    """Split a pipe-delimited string back into a list, filtering blanks."""
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _parse_chunk(
    doc_id: str,
    document: str,
    metadata: dict[str, Any],
    distance: float,
) -> RetrievedChunk:
    """Build a RetrievedChunk from raw ChromaDB result fields."""
    return RetrievedChunk(
        content=document,
        chunk_id=doc_id,
        document_id=str(metadata.get("document_id", "")),
        title=str(metadata.get("title", "")),
        url=str(metadata.get("url", "")),
        source=str(metadata.get("source", "MedlinePlus")),
        groups=_split_pipe(metadata.get("groups", "")),
        mesh=_split_pipe(metadata.get("mesh", "")),
        primary_institute=str(metadata.get("primary_institute", "")),
        chunk_index=int(metadata.get("chunk_index", 1)),
        total_chunks=int(metadata.get("total_chunks", 1)),
        distance=float(distance),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────


class Retriever:
    """ChromaDB-backed retriever for the medical knowledge base.

    The collection connection and embedding function are initialised once
    on the first ``retrieve()`` call (lazy initialisation), then cached
    for the lifetime of the instance.

    Parameters
    ----------
    chroma_path:
        Path to the ChromaDB persistence directory.
    collection_name:
        Name of the ChromaDB collection to query.
    embedding_model:
        HuggingFace sentence-transformers model ID.
        MUST match the model used during indexing.
    default_k:
        Default number of results to return when ``k`` is not supplied.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        default_k: int = TOP_K,
    ) -> None:
        self._chroma_path = chroma_path
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._default_k = default_k

        # Initialised lazily on first retrieve() call
        self._collection: chromadb.Collection | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_collection(self) -> chromadb.Collection:
        """Return the cached collection, connecting if not yet initialised."""
        if self._collection is not None:
            return self._collection

        logger.info(
            "Connecting to ChromaDB at '%s', collection '%s'",
            self._chroma_path,
            self._collection_name,
        )

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model,
            normalize_embeddings=True,
        )

        client = chromadb.PersistentClient(path=self._chroma_path)

        try:
            self._collection = client.get_collection(
                name=self._collection_name,
                embedding_function=embedding_fn,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{self._collection_name}' not found in ChromaDB at "
                f"'{self._chroma_path}'.\n"
                "Run the indexing script first:\n"
                "  PYTHONPATH=. python app/vectorstore/embed_documents.py"
            ) from exc

        doc_count = self._collection.count()
        logger.info("Connected — collection contains %s documents", f"{doc_count:,}")
        return self._collection

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Embed *query* and return the Top-K most relevant chunks.

        Parameters
        ----------
        query:
            The user's natural-language question.
        k:
            Number of results to return.
            Defaults to ``self._default_k`` (from ``settings.TOP_K``).

        Returns
        -------
        list[RetrievedChunk]
            Results ordered by ascending cosine distance (most relevant first).
            Empty list if the collection is empty or query is blank.
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query — returning empty list.")
            return []

        n = k if k is not None else self._default_k
        collection = self._get_collection()

        logger.debug("Querying ChromaDB: k=%s, query=%r", n, query[:80])

        results = collection.query(
            query_texts=[query],
            n_results=min(n, collection.count()),  # never ask for more than exists
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB wraps results in an outer list (one per query_texts entry)
        ids: list[str] = results["ids"][0]
        documents: list[str] = results["documents"][0]
        metadatas: list[dict] = results["metadatas"][0]
        distances: list[float] = results["distances"][0]

        chunks = [
            _parse_chunk(doc_id, document, metadata, distance)
            for doc_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

        # Filter out low-similarity noise chunks (e.g., Lung Cancer/Menopause when querying Burns)
        if chunks:
            top_doc = chunks[0].document_id
            top_title = chunks[0].title
            filtered = [
                c for c in chunks
                if c.similarity_score >= 0.78 or c.document_id == top_doc or c.title == top_title
            ]
            chunks = filtered if filtered else [chunks[0]]

        logger.debug(
            "Retrieved %s chunks — best distance: %.4f, worst: %.4f",
            len(chunks),
            distances[0] if distances else 0,
            distances[-1] if distances else 0,
        )

        return chunks

    @property
    def collection_size(self) -> int:
        """Number of documents currently in the collection."""
        return self._get_collection().count()
