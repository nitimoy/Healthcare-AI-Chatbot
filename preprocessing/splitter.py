"""
preprocessing/splitter.py
─────────────────────────
Structural document split: header / summary / footer.

Design decision — operate on RawRecord fields, NOT on the concatenated string.

The old approach (split_header_summary_footer) concatenated all document
sections into a single string in build_embedding_documents.py, then tried
to find the boundaries again by searching for marker substrings. That is
fragile: marker-search fails when optional sections are absent (the "See
Also" footer-detection bug) and the code is harder to reason about.

This module builds (header, summary, footer) directly from the structured
RawRecord fields so the boundaries are always exact and no marker-string
parsing is needed at all.
"""

from __future__ import annotations

from typing import NamedTuple

from preprocessing.parser import RawRecord


class DocumentParts(NamedTuple):
    """Structural decomposition of a single medical document.

    header:  Topic name, aliases, and category labels.
             Prepended to EVERY chunk so each chunk is self-contained.
    summary: Free-text clinical description — this is what gets chunked.
    footer:  Related topics, see-also references, and primary institute.
             Appended to the LAST chunk only.
    """

    header: str
    summary: str
    footer: str


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _section(label: str, body: str) -> str:
    return f"{label}\n{body}"


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────


def split(record: RawRecord) -> DocumentParts:
    """Decompose *record* into (header, summary, footer).

    All three parts may be empty strings if the source data is absent,
    which the chunker handles gracefully.
    """
    header_parts: list[str] = [f"Medical Topic: {record['title']}"]

    if record["also_called"]:
        header_parts.append(_section("Also Known As:", _bullet(record["also_called"])))

    group_names = [g["name"] for g in record["groups"] if g.get("name")]
    if group_names:
        header_parts.append(_section("Medical Categories:", _bullet(group_names)))

    header = "\n\n".join(header_parts)

    # ── Summary ────────────────────────────────────────────────
    summary = record["summary"]

    # ── Footer ─────────────────────────────────────────────────
    footer_parts: list[str] = []

    related_names = [t["name"] for t in record["related_topics"] if t.get("name")]
    if related_names:
        footer_parts.append(_section("Related Medical Topics:", _bullet(related_names)))

    if record["see_references"]:
        footer_parts.append(_section("See Also:", _bullet(record["see_references"])))

    institute = record["primary_institute"].get("name", "")
    if institute:
        footer_parts.append(_section("Primary Institute:", institute))

    footer = "\n\n".join(footer_parts)

    return DocumentParts(header=header, summary=summary, footer=footer)
