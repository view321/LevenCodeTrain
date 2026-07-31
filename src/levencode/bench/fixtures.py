from __future__ import annotations

import json
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures" / "python_snippets.json"


def load_snippets() -> list[str]:
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))
