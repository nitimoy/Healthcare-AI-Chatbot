"""
preprocessing/io.py
───────────────────
Centralised JSON I/O backed by orjson.

Why orjson instead of stdlib json?
  • 3-10× faster serialisation/deserialisation.
  • Produces correct UTF-8 bytes — no `ensure_ascii=False` workaround.
  • Raises on NaN/Infinity by default (safer for ML pipelines).
  • Serialises datetime, numpy arrays, and dataclasses natively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def load_json(path: str | Path) -> Any:
    """Read and deserialise a JSON file using orjson."""
    return orjson.loads(Path(path).read_bytes())


def save_json(path: str | Path, data: Any, *, indent: bool = True) -> None:
    """Serialise *data* to *path* using orjson.

    Parameters
    ----------
    path:
        Destination file. Parent directories are created automatically.
    data:
        Any orjson-serialisable object.
    indent:
        If True (default) the output is pretty-printed with 2-space indentation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    option = orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS if indent else orjson.OPT_NON_STR_KEYS
    p.write_bytes(orjson.dumps(data, option=option))
