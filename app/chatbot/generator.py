"""
app/chatbot/generator.py
─────────────────────────
Orchestrates the complete RAG pipeline turn.

LLM backend: NVIDIA NIM API (OpenAI-compatible endpoint).
All connection parameters are read from .env via settings.py — no secrets
or model names are hardcoded here.

Pipeline per turn:
  1. Guardrails check — keyword regex (zero cost, no API call)
  2. ChromaDB retrieval — BGE-M3 embedding + HNSW cosine search
  3. Prompt assembly — system prompt + history + context + question
  4. NVIDIA NIM streaming — token-by-token via OpenAI SDK
  5. Memory update — stores exchange after stream is exhausted

Usage (Streamlit):
    generator = MedicalChatGenerator()
    chunks, token_stream = generator.stream_response("What is diabetes?")
    full_text = st.write_stream(token_stream)   # streams to UI
    # chunks → source citations
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from pathlib import Path
import sys

# Ensure project root is on sys.path so .env is found
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI

from app.chatbot.evaluator import EvidenceAssessmentEngine
from app.chatbot.guardrails import check_and_respond
from app.chatbot.memory import ConversationMemory
from app.chatbot.prompts import SYSTEM_PROMPT, build_user_message
from app.chatbot.retriever import RetrievedChunk, Retriever
from app.config.settings import (
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_OUTPUT_TOKENS,
    NVIDIA_API_KEY,
    TEMPERATURE,
    TOP_K,
)

logger = logging.getLogger(__name__)

_FOLLOWUP_INTENTS = re.compile(
    r"\b(it|its|this|these|that|those|they|them|their|same|such|"
    r"medicine|medications|drug|drugs|pill|pills|food|foods|diet|eat|eating|avoid|prevent|prevention|"
    r"contagious|spread|infectious|cure|exercise|workout|vaccine|vaccines|remedy|remedies|cause|causes|"
    r"dangerous|worse|better|more|different|difference|versus|vs|compare|comparison)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

TokenStream = Generator[str, None, None]
GeneratorResult = tuple[list[RetrievedChunk], TokenStream]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────


class MedicalChatGenerator:
    """Orchestrates guardrails → retrieval → NVIDIA NIM generation.

    All LLM settings (model, base URL, temperature, max tokens) come from
    environment variables loaded by settings.py — switch models by editing
    .env only.

    Parameters
    ----------
    retriever:
        Optional pre-built Retriever. Created fresh if None.
    memory:
        Optional ConversationMemory shared with the Streamlit session.
        Created fresh if None.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        # ── Validate API key ──────────────────────────────────────────────────
        if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your-nvidia-api-key-here":
            raise EnvironmentError(
                "NVIDIA_API_KEY is not configured.\n"
                "1. Get a key at: https://build.nvidia.com\n"
                "2. Add it to your .env file:\n"
                "   NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx\n"
                "3. Restart the app."
            )

        # ── OpenAI-compatible client pointing to NVIDIA NIM ───────────────────
        self._client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=30.0,
        )

        self._retriever = retriever or Retriever()
        self._memory = memory or ConversationMemory()
        self._evaluator = EvidenceAssessmentEngine()

        logger.info(
            "MedicalChatGenerator ready | model=%s | endpoint=%s | top_k=%s",
            LLM_MODEL,
            LLM_BASE_URL,
            TOP_K,
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def memory(self) -> ConversationMemory:
        """The shared conversation memory instance."""
        return self._memory

    # ── Private helpers ───────────────────────────────────────────────────────

    def _static_stream(self, text: str) -> TokenStream:
        """Wrap a static string as a single-token stream (for guardrail bypass)."""
        yield text

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build the full messages list in OpenAI format.

        Structure:
          [system] → [recent_user, recent_assistant] → [current_user]

        The system prompt defines role and rules.
        Only the most recent exchange provides conversational context to prevent
        stale topic contamination from older turns.
        """
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject only the MOST RECENT exchange (last 2 turns) to prevent old topic contamination
        history = self._memory.get_history()
        recent_turns = history[-2:] if len(history) >= 2 else history

        for turn in recent_turns:
            role = turn["role"] if turn["role"] != "model" else "assistant"
            content = turn["parts"][0]["text"]
            # Shorten past assistant responses in context history to prevent topic pollution
            if role == "assistant" and len(content) > 150:
                content = content[:150] + "..."
            messages.append({"role": role, "content": content})

        # Append current user message (with context injected)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _nvidia_stream(self, messages: list[dict]) -> TokenStream:
        """Open a streaming request to the NVIDIA NIM endpoint and yield tokens."""
        stream = self._client.chat.completions.create(
            model=LLM_MODEL,           # read from .env — no hardcoding
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def _contextualize_query(self, question: str) -> str:
        """Rewrite ambiguous follow-up questions into standalone search queries using recent conversation context."""
        if self._memory.is_empty:
            return question

        # If the question does not contain follow-up pronouns or context-dependent intents, treat as standalone
        if not _FOLLOWUP_INTENTS.search(question):
            return question

        # Focus strictly on the MOST RECENT 1-2 exchanges to resolve follow-up pronouns (preventing stale topic drift)
        recent_exchanges = list(self._memory._exchanges)[-2:]
        history_summary = []
        for user_text, assistant_text in recent_exchanges:
            history_summary.append(f"User: {user_text}\nAssistant: {assistant_text[:150]}")
        
        history_str = "\n".join(history_summary)

        prompt = (
            "Given the recent conversation context and a follow-up user question, "
            "rephrase the follow-up question to be a self-contained, standalone search query.\n\n"
            "RULES FOR REPHRASING:\n"
            "1. ALWAYS attach the primary medical subjects/conditions discussed in recent conversation context.\n"
            "2. If the user asks a comparison question (e.g. 'Which one is more dangerous?', 'How do they compare?'), identify ALL entities/conditions discussed in the recent context (e.g. COPD and Asthma) and preserve ALL of them in the standalone search query.\n\n"
            "Examples:\n"
            "- Context: COPD vs Asthma | Question: 'Which one is more dangerous?' -> Search Query: 'Is COPD or asthma more dangerous?'\n"
            "- Context: Asthma | Question: 'What foods should I avoid?' -> Search Query: 'What foods should people with asthma avoid?'\n"
            "- Context: Diabetes | Question: 'Is it contagious?' -> Search Query: 'Is diabetes contagious?'\n\n"
            "Do NOT answer the question. Return ONLY the plain text search query:\n\n"
            f"Recent Context:\n{history_str}\n\n"
            f"Follow-up Question: {question}\n\n"
            "Standalone Search Query:"
        )

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                logger.info("Rewrote follow-up query for retrieval: %r -> %r", question, rewritten)
                return rewritten
        except Exception as exc:
            logger.warning("Query rewriting failed: %s — using original question.", exc)
        
        return question

    # ── Public API ────────────────────────────────────────────────────────────

    def stream_response(
        self,
        question: str,
        k: int = TOP_K,
    ) -> GeneratorResult:
        """Run the full RAG pipeline and return a (chunks, token_stream) tuple.

        The token_stream is a lazy generator — the NVIDIA API is called only
        when the caller begins iterating it. Memory is updated automatically
        via a finally block after the stream is exhausted.

        Parameters
        ----------
        question:
            Raw user question.
        k:
            Number of ChromaDB results to retrieve.

        Returns
        -------
        tuple[list[RetrievedChunk], TokenStream]
            chunks       — retrieved evidence for source citations
            token_stream — generator yielding response text tokens
        """
        # ── 1. Guardrails ─────────────────────────────────────────────────────
        safety_response = check_and_respond(question)
        if safety_response:
            logger.info("Guardrail triggered — bypassing RAG pipeline (memory untouched).")
            return [], self._static_stream(safety_response)

        # ── 2. Contextualize Follow-up Query for Retrieval ────────────────────
        search_query = self._contextualize_query(question)

        # ── 3. Retrieval ──────────────────────────────────────────────────────
        logger.info("Retrieving top-%s chunks | query=%r", k, search_query[:80])
        chunks = self._retriever.retrieve(search_query, k=k)
        logger.info(
            "Retrieved %s chunks | best_score=%.3f",
            len(chunks),
            chunks[0].similarity_score if chunks else 0.0,
        )

        # ── 4. Deterministic Evidence Assessment ──────────────────────────────
        is_valid, reason = self._evaluator.evaluate(search_query, chunks)
        if not is_valid:
            logger.info("Evidence assessment failed (%s) — bypassing LLM (memory untouched).", reason)
            fallback_msg = (
                "I couldn't find reliable medical information in my knowledge base "
                "to answer that question. Please consult a qualified healthcare professional."
            )
            return [], self._static_stream(fallback_msg)

        # ── 5. Prompt assembly ────────────────────────────────────────────────
        user_message = build_user_message(question, chunks)
        messages = self._build_messages(user_message)

        # ── 6. Stream from NVIDIA NIM (lazy) ──────────────────────────────────
        logger.info("Calling %s via %s (streaming)...", LLM_MODEL, LLM_BASE_URL)

        def _streaming_with_memory() -> TokenStream:
            parts: list[str] = []
            try:
                for token in self._nvidia_stream(messages):
                    parts.append(token)
                    yield token
            finally:
                full = "".join(parts)
                if full:
                    self._memory.add_exchange(question, full)
                    logger.debug(
                        "Memory updated | exchanges=%s", self._memory.exchange_count
                    )

        return chunks, _streaming_with_memory()
