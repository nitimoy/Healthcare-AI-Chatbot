"""
app/config/settings.py
───────────────────────
Single source of truth for all runtime configuration.

ALL values are read from environment variables, which are populated by
loading the .env file at module import time via python-dotenv.

To change any setting — including the LLM model, API endpoint, or
embedding model — edit .env only. No code changes required.

Priority:
  1. Environment variable already set in the shell (highest)
  2. Value in .env file
  3. Default fallback defined here (lowest)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ─────────────────────────────────────────────────────────────────────────────
# LLM — NVIDIA NIM (OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────

#: NVIDIA API key — read from NVIDIA_API_KEY in .env
NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")

#: Base URL for the NVIDIA NIM / OpenAI-compatible API
LLM_BASE_URL: str = os.environ.get(
    "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
)

#: Model identifier — change in .env to switch models, no code changes needed
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta/llama-3.1-8b-instruct")

# ─────────────────────────────────────────────────────────────────────────────
# Generation parameters
# ─────────────────────────────────────────────────────────────────────────────

#: Sampling temperature — lower = more factual, less creative
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.3"))

#: Maximum tokens the LLM may generate per response
MAX_OUTPUT_TOKENS: int = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

#: Number of chunks returned from ChromaDB per query
TOP_K: int = int(os.environ.get("TOP_K", "5"))

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────────────────────

#: sentence-transformers model used for both indexing and querying
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

#: Persistent storage path for ChromaDB (relative to project root)
CHROMA_PATH: str = os.environ.get("CHROMA_PATH", "data/chroma")

#: Collection name inside ChromaDB
COLLECTION_NAME: str = os.environ.get(
    "COLLECTION_NAME", "medical_knowledge_base"
)

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────

#: Output of the ingestion pipeline — input to the indexing script
CHUNKS_PATH: str = os.environ.get(
    "CHUNKS_PATH", "data/processed/chunked_documents.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Indexing
# ─────────────────────────────────────────────────────────────────────────────

#: Number of chunks per ChromaDB upsert batch
EMBED_BATCH_SIZE: int = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
