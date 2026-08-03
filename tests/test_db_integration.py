"""Needs a live Postgres (`make up`) — the double-booking guard is a
PostgreSQL-only EXCLUDE constraint (see db/models.py) that SQLite can't
enforce, so it can't be covered by the offline tests in test_db.py.
"""

import datetime as dt

import pytest
from sqlalchemy.exc import IntegrityError

from parking_bot.config import get_settings
from parking_bot.db.base import build_engine, build_session_factory
from parking_bot.db.init_db import init_db
from parking_bot.db.models import Reservation, Space, User

pytestmark = pytest.mark.integration


def test_reservation_exclude_constraint_rejects_overlapping_booking() -> None:
    engine = init_db(build_engine(get_settings()))
    session = build_session_factory(engine)()
    try:
        user = User(external_id="integration-test-double-booking")
        space = Space(code="INTEGRATION-TEST-DOUBLE-BOOKING", level=1, vehicle_type="car")
        session.add_all([user, space])
        session.flush()

        session.add(
            Reservation(
                user_id=user.id,
                space_id=space.id,
                starts_at=dt.datetime(2026, 3, 1, 9, 0, tzinfo=dt.UTC),
                ends_at=dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC),
                status="confirmed",
                price_cents=1500,
                currency="USD",
            )
        )
        session.flush()

        session.add(
            Reservation(
                user_id=user.id,
                space_id=space.id,
                starts_at=dt.datetime(2026, 3, 1, 10, 0, tzinfo=dt.UTC),
                ends_at=dt.datetime(2026, 3, 1, 11, 0, tzinfo=dt.UTC),
                status="pending_confirmation",
                price_cents=500,
                currency="USD",
            )
        )
        with pytest.raises(IntegrityError, match="reservations_no_overlap"):
            session.flush()
    finally:
        session.rollback()
        session.close()
