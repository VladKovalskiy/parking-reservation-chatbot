"""Availability is a query, not a table (see docs/sql-schema.md).

Mirrors the `NOT EXISTS` query in the design doc, but expressed as a
portable half-open-interval overlap comparison (`starts_at < window_end AND
ends_at > window_start`) instead of Postgres' `tstzrange(...) &&` operator,
so it runs identically against SQLite in tests and Postgres in production —
unlike the double-booking guard itself, which is deliberately Postgres-only
(see `db/models.py`).
"""

import datetime as dt

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from parking_bot.db.models import Reservation, Space

_HOLDING_STATUSES = ("pending_confirmation", "confirmed")


def available_spaces(
    session: Session,
    *,
    vehicle_type: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> list[Space]:
    """Active spaces of `vehicle_type` with no holding reservation overlapping the window."""
    overlaps = (
        exists()
        .where(Reservation.space_id == Space.id)
        .where(Reservation.status.in_(_HOLDING_STATUSES))
        .where(Reservation.starts_at < window_end)
        .where(Reservation.ends_at > window_start)
    )
    stmt = (
        select(Space)
        .where(Space.vehicle_type == vehicle_type)
        .where(Space.is_active.is_(True))
        .where(~overlaps)
    )
    return list(session.scalars(stmt))
