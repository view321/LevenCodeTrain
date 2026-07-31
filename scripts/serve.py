"""Serve the training dashboard:  python scripts/serve.py --runs-dir runs"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    import uvicorn

    from levencode.webui.server import create_app

    uvicorn.run(create_app(args.runs_dir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
