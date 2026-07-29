"""
preprocessing/parser.py
───────────────────────
MedlinePlus XML → list[RawRecord] using lxml.

Why lxml over stdlib xml.etree.ElementTree?
  • iterparse enables streaming — the 30 MB XML is never fully in memory.
  • lxml's C parser is 5-10× faster than the Python-based ElementTree.
  • lxml exposes .text_content() on subtrees and handles encoding edge cases
    that can trip up ElementTree's itertext().
  • Better error messages on malformed XML.

The parser is intentionally decoupled from the rest of the pipeline:
it converts the XML into plain Python dicts (RawRecord) so that the
downstream cleaner, splitter, and chunker are format-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from lxml import etree

from preprocessing.cleaner import normalise


# ─────────────────────────────────────────────
# Output types
# ─────────────────────────────────────────────


class GroupRecord(TypedDict):
    id: str
    name: str
    url: str


class RelatedTopicRecord(TypedDict):
    id: str
    name: str
    url: str


class MeshRecord(TypedDict):
    id: str
    name: str


class InstituteRecord(TypedDict):
    name: str
    url: str


class RawRecord(TypedDict):
    id: str
    title: str
    url: str
    language: str
    date_created: str
    meta_description: str
    summary: str
    also_called: list[str]
    groups: list[GroupRecord]
    related_topics: list[RelatedTopicRecord]
    see_references: list[str]
    mesh_headings: list[MeshRecord]
    primary_institute: InstituteRecord


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _text(element: etree._Element | None) -> str:
    """Extract all text from *element*, normalised via the cleaner."""
    if element is None:
        return ""
    raw = "".join(element.itertext())
    return normalise(raw)


def _attrib(element: etree._Element | None, key: str, default: str = "") -> str:
    if element is None:
        return default
    return element.get(key, default)


# ─────────────────────────────────────────────
# Per-element extractors
# ─────────────────────────────────────────────


def _parse_groups(topic: etree._Element) -> list[GroupRecord]:
    return [
        GroupRecord(
            id=g.get("id", ""),
            name=_text(g),
            url=g.get("url", ""),
        )
        for g in topic.findall("group")
    ]


def _parse_related_topics(topic: etree._Element) -> list[RelatedTopicRecord]:
    return [
        RelatedTopicRecord(
            id=item.get("id", ""),
            name=_text(item),
            url=item.get("url", ""),
        )
        for item in topic.findall("related-topic")
    ]


def _parse_mesh(topic: etree._Element) -> list[MeshRecord]:
    result: list[MeshRecord] = []
    for mh in topic.findall("mesh-heading"):
        descriptor = mh.find("descriptor")
        if descriptor is None:
            continue
        result.append(MeshRecord(id=descriptor.get("id", ""), name=_text(descriptor)))
    return result


def _parse_simple_list(topic: etree._Element, tag: str) -> list[str]:
    return [t for item in topic.findall(tag) if (t := _text(item))]


def _parse_topic(topic: etree._Element) -> RawRecord | None:
    """Parse a single <health-topic> element; returns None for non-English topics."""
    if topic.get("language") != "English":
        return None

    institute = topic.find("primary-institute")

    return RawRecord(
        id=topic.get("id", ""),
        title=topic.get("title", ""),
        url=topic.get("url", ""),
        language=topic.get("language", ""),
        date_created=topic.get("date-created", ""),
        meta_description=topic.get("meta-desc", ""),
        summary=_text(topic.find("full-summary")),
        also_called=_parse_simple_list(topic, "also-called"),
        groups=_parse_groups(topic),
        related_topics=_parse_related_topics(topic),
        see_references=_parse_simple_list(topic, "see-reference"),
        mesh_headings=_parse_mesh(topic),
        primary_institute=InstituteRecord(
            name=_text(institute),
            url=_attrib(institute, "url"),
        ),
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────


def parse_xml(xml_path: str | Path) -> list[RawRecord]:
    """Stream-parse *xml_path* and return all English health-topic records.

    Uses ``lxml.etree.iterparse`` so the 30 MB XML file is processed
    element-by-element rather than loaded entirely into memory.
    """
    xml_path = Path(xml_path)
    records: list[RawRecord] = []

    context = etree.iterparse(str(xml_path), events=("end",), tag="health-topic")

    for _, topic in context:
        record = _parse_topic(topic)
        if record is not None:
            records.append(record)
        # Free memory — critical for large XML files
        topic.clear()

    return records
