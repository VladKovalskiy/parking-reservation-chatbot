"""Initialize the dynamic-data schema (see docs/sql-schema.md).

Run:  uv run python -m parking_bot.db.init_db

Idempotent: `checkfirst=True` (create_all's default) skips tables that
already exist, so re-running never drops or duplicates data — same spirit as
the Milvus ingestion pipeline, minus the drop-and-recreate (schema changes
here are additive, not a full reindex).
"""

import sys

from sqlalchemy import Engine

from parking_bot.db import models  # noqa: F401  (registers tables on Base.metadata)
from parking_bot.db.base import Base, build_engine


def init_db(engine: Engine | None = None) -> Engine:
    """Create every table (and the Postgres-only double-booking guard) if missing."""
    engine = engine or build_engine()
    Base.metadata.create_all(engine)
    return engine


def main() -> int:
    engine = init_db()
    print(f"Schema initialized against {engine.url!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
