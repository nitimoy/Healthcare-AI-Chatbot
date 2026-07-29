"""
preprocessing/cleaner.py
────────────────────────
Text normalisation utilities.

Why ftfy + lxml.html instead of manual unescape + regex?
  • The old pipeline called html.unescape() BEFORE stripping tags.
    This meant `&lt;5mg&gt;` became `<5mg>` which the tag-strip regex
    then deleted as if it were an HTML element. lxml.html parses the
    fragment correctly before any unescaping occurs.
  • ftfy.fix_text handles Unicode mojibake, mis-encoded em-dashes, smart
    quotes, control characters, and whitespace normalisation in one call —
    replacing a fragile chain of re.sub patterns.
  • lxml.html.fromstring is lenient on malformed HTML (common in scraped
    medical text) whereas stdlib html.parser raises on bad markup.
"""

from __future__ import annotations

import ftfy
from lxml import html as lhtml


def strip_html(raw: str) -> str:
    """Remove all HTML tags from *raw* and decode entities.

    Uses lxml's lenient HTML parser so that fragments like
    ``<b>Aspirin</b> &amp; ibuprofen`` become ``Aspirin & ibuprofen``
    without accidentally deleting text that looks like a tag.
    """
    if not raw or not raw.strip():
        return ""

    # lxml.html.fromstring wraps bare text in <span>/<div> automatically,
    # so text_content() safely retrieves all inner text.
    try:
        doc = lhtml.fromstring(raw)
        text = doc.text_content()
    except Exception:
        # Fallback: if lxml cannot parse at all, use the raw string
        text = raw

    return text


def normalise(text: str) -> str:
    """Full normalisation pass: strip HTML → fix Unicode → collapse whitespace.

    Steps
    -----
    1. Strip HTML tags via lxml (correct entity handling).
    2. Run ftfy.fix_text to repair encoding artifacts and normalise Unicode.
    3. Collapse runs of whitespace to a single space and strip edges.
    """
    text = strip_html(text)
    text = ftfy.fix_text(text)
    # Collapse all internal whitespace (tabs, non-breaking spaces, etc.)
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text
