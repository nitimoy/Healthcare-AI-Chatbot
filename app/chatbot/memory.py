"""
app/chatbot/memory.py
──────────────────────
Conversation memory — rolling window of last N exchanges.

Responsibility
──────────────
Maintain a lightweight, in-process conversation history that is injected
into the Gemini API call so the model understands conversational context
(e.g. pronoun references, follow-up questions).

Design decisions
────────────────
1.  deque(maxlen=MAX_EXCHANGES):
    Python's deque with a fixed maxlen automatically evicts the oldest
    exchange when a new one is appended. No manual trimming needed.

2.  Memory is context-only — retrieval is always re-run:
    The ConversationMemory only feeds prior turns to the LLM so it can
    follow the conversation. It does NOT cache or reuse retrieved chunks.
    Every new question triggers a fresh ChromaDB search to guarantee the
    most relevant context is always used.

3.  get_history() returns the Gemini messages format:
    [ {"role": "user", "parts": [{"text": "..."}]},
      {"role": "model", "parts": [{"text": "..."}]}, ... ]
    This maps directly to google-generativeai's ChatSession history format.

4.  Clear on demand:
    Streamlit's "Clear Chat" button calls memory.clear() to reset state
    without recreating the Retriever or reloading the model.
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)

# Maximum number of (user, assistant) exchange pairs to keep.
# At 5 exchanges = 10 messages — enough for continuity without bloating the prompt.
MAX_EXCHANGES: int = 5


class ConversationMemory:
    """Rolling window of the last N conversation exchanges.

    Each "exchange" is one (user_message, assistant_response) pair.
    The memory automatically drops the oldest pair when the limit is reached.

    Parameters
    ----------
    max_exchanges:
        Maximum number of (user, assistant) pairs to retain.

    Examples
    --------
    >>> memory = ConversationMemory()
    >>> memory.add_exchange("What is diabetes?", "Diabetes is...")
    >>> history = memory.get_history()
    >>> len(history)  # 2 messages: user + model
    2
    """

    def __init__(self, max_exchanges: int = MAX_EXCHANGES) -> None:
        self._max_exchanges = max_exchanges
        # Each element is a (user_text, assistant_text) tuple.
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=max_exchanges)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_exchange(self, user_message: str, assistant_response: str) -> None:
        """Append a completed exchange to the memory window.

        Parameters
        ----------
        user_message:
            The raw user question (before context injection).
        assistant_response:
            The full assistant response text.
        """
        self._exchanges.append((user_message, assistant_response))
        logger.debug(
            "Memory: added exchange %s/%s", len(self._exchanges), self._max_exchanges
        )

    def get_history(self) -> list[dict]:
        """Return the conversation history in Gemini API message format.

        Returns
        -------
        list of dicts with keys ``role`` and ``parts``, where
        ``role`` is ``"user"`` or ``"model"`` and ``parts`` is a list
        containing a single ``{"text": "..."}`` dict.
        """
        messages: list[dict] = []
        for user_text, assistant_text in self._exchanges:
            messages.append({"role": "user", "parts": [{"text": user_text}]})
            messages.append({"role": "model", "parts": [{"text": assistant_text}]})
        return messages

    def clear(self) -> None:
        """Erase all stored exchanges (e.g. when the user clicks Clear Chat)."""
        self._exchanges.clear()
        logger.debug("Memory cleared.")

    @property
    def exchange_count(self) -> int:
        """Number of exchanges currently stored."""
        return len(self._exchanges)

    @property
    def is_empty(self) -> bool:
        """True if no exchanges have been recorded yet."""
        return len(self._exchanges) == 0
