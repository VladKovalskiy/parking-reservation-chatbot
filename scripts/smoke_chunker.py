"""Smoke test: inspect how data/static/*.md gets loaded and chunked.

Run:  uv run python scripts/smoke_chunker.py
      uv run python scripts/smoke_chunker.py data/static/rules.md
      uv run python scripts/smoke_chunker.py --chunk-size 400 --chunk-overlap 50

Loads section Documents via load_static_documents(), splits them with
chunk_documents() using the configured chunk_size/chunk_overlap, and prints
each resulting chunk with its metadata and character count. Useful for
eyeballing whether chunk boundaries land in sensible places before running
the real ingestion pipeline.

This is a manual check, not a pytest test — it must never run in CI.
"""

import argparse
import sys
from pathlib import Path

from parking_bot.config import get_settings
from parking_bot.ingestion.chunker import chunk_documents
from parking_bot.ingestion.loader import load_static_documents

DEFAULT_STATIC_DIR = Path("data/static")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_STATIC_DIR,
        help="a single .md file or a directory of .md files (default: data/static)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap

    static_dir = args.path if args.path.is_dir() else args.path.parent
    sections = load_static_documents(static_dir)

    if args.path.is_file():
        sections = [s for s in sections if s.metadata["source"] == args.path.name]

    if not sections:
        print(f"No sections found under {args.path}")
        return 1

    chunks = chunk_documents(sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print(f"[config] chunk_size={chunk_size} chunk_overlap={chunk_overlap}")
    print(f"[sections] {len(sections)} loaded from {static_dir}")
    print(f"[chunks] {len(chunks)} produced\n")

    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        chunk_index = chunk.metadata["chunk_index"]
        length = len(chunk.page_content)
        print(f"--- {doc_id} [chunk {chunk_index}] ({length} chars) ---")
        print(chunk.page_content)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
