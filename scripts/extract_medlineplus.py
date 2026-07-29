#!/usr/bin/env python3
"""
scripts/extract_medlineplus.py
──────────────────────────────
Parse the MedlinePlus XML export and save the English health-topic
records as a canonical JSON knowledge base.

Pipeline position:

    mplus_topics_<date>.xml
            ↓
    scripts/extract_medlineplus.py
            ↓
    data/raw/medical_kb_raw.json

Changes from the original:
  • Delegates all XML parsing to preprocessing.parser (lxml-based, streaming).
  • Delegates all text cleaning to preprocessing.cleaner (ftfy + lxml.html).
  • Delegates JSON I/O to preprocessing.io (orjson).
  • This file is now a thin entry point — ~40 lines of orchestration.

Usage:
    python scripts/extract_medlineplus.py
    python scripts/extract_medlineplus.py --xml mplus_topics_2026-07-28.xml
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from preprocessing.io import save_json
from preprocessing.parser import parse_xml

DEFAULT_XML = Path("mplus_topics_2026-07-28.xml")
DEFAULT_OUTPUT = Path("data/raw/medical_kb_raw.json")


def run(xml_path: Path, output_path: Path) -> None:
    print(f"Parsing  : {xml_path}  ({xml_path.stat().st_size / 1_048_576:.1f} MB)")
    t0 = time.perf_counter()

    records = parse_xml(xml_path)

    elapsed = time.perf_counter() - t0
    print(f"Parsed   : {len(records):,} English records  ({elapsed:.1f}s)")

    save_json(output_path, records)
    print(f"Saved    : {output_path}")

    # ── Group distribution report ────────────────────────────────
    group_counter: Counter[str] = Counter()
    for r in records:
        for g in r["groups"]:
            if g.get("name"):
                group_counter[g["name"]] += 1

    print("\n── Top 20 Medical Groups ──────────────────────────────────")
    for name, count in group_counter.most_common(20):
        print(f"  {name:<40} {count}")

    print("\n── Sample Record ──────────────────────────────────────────")
    import orjson
    print(orjson.dumps(records[0], option=orjson.OPT_INDENT_2).decode())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MedlinePlus XML to JSON.")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(xml_path=args.xml, output_path=args.output)