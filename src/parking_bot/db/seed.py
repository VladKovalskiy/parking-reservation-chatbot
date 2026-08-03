"""Seed demo data for local development (see docs/sql-schema.md).

Run:  uv run python -m parking_bot.db.seed

Idempotent-ish for repeated dev use: clears the seeded tables first (in FK
order) so re-running gives a clean, predictable demo dataset instead of
duplicate rows or unique-constraint errors.
"""

import datetime as dt
import sys

from sqlalchemy.orm import Session

from parking_bot.db.base import build_session_factory, session_scope
from parking_bot.db.init_db import init_db
from parking_bot.db.models import OperatingHours, Reservation, Space, Tariff, User

# Mirrors general.md: levels 1-2 for cars/motorcycles, a bicycle rack,
# accessible spaces on level 1.
SPACES = [
    Space(code="A-01", level=1, vehicle_type="car", is_accessible=True),
    Space(code="A-02", level=1, vehicle_type="car", is_accessible=True),
    Space(code="A-03", level=1, vehicle_type="car"),
    Space(code="B-01", level=2, vehicle_type="car"),
    Space(code="B-02", level=2, vehicle_type="car"),
    Space(code="M-01", level=1, vehicle_type="motorcycle"),
    Space(code="M-02", level=1, vehicle_type="motorcycle"),
    Space(code="R-01", level=1, vehicle_type="bicycle"),
    Space(code="R-02", level=1, vehicle_type="bicycle"),
]

TARIFFS = [
    Tariff(
        vehicle_type="car",
        day_type="weekday",
        unit="hour",
        price_cents=500,
        valid_from=dt.date(2026, 1, 1),
    ),
    Tariff(
        vehicle_type="car",
        day_type="weekend",
        unit="hour",
        price_cents=300,
        valid_from=dt.date(2026, 1, 1),
    ),
    Tariff(
        vehicle_type="motorcycle",
        day_type="weekday",
        unit="hour",
        price_cents=250,
        valid_from=dt.date(2026, 1, 1),
    ),
    Tariff(
        vehicle_type="bicycle",
        day_type="weekday",
        unit="hour",
        price_cents=0,
        valid_from=dt.date(2026, 1, 1),
    ),
]

# 0 = Sunday, matching operating_hours.day_of_week's CHECK in docs/sql-schema.md.
OPERATING_HOURS = [
    OperatingHours(day_of_week=0, opens_at=dt.time(8, 0), closes_at=dt.time(20, 0)),
    OperatingHours(day_of_week=1, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)),
    OperatingHours(day_of_week=2, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)),
    OperatingHours(day_of_week=3, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)),
    OperatingHours(day_of_week=4, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)),
    OperatingHours(day_of_week=5, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)),
    OperatingHours(day_of_week=6, opens_at=dt.time(8, 0), closes_at=dt.time(20, 0)),
]

USERS = [
    User(external_id="demo-user-1", display_name="Demo User"),
]


def seed(session: Session) -> None:
    # FK-safe delete order: children before parents.
    session.query(Reservation).delete()
    session.query(User).delete()
    session.query(OperatingHours).delete()
    session.query(Tariff).delete()
    session.query(Space).delete()
    session.flush()

    session.add_all(SPACES)
    session.add_all(TARIFFS)
    session.add_all(OPERATING_HOURS)
    session.add_all(USERS)
    session.flush()

    car_tariff = next(t for t in TARIFFS if t.vehicle_type == "car" and t.day_type == "weekday")
    session.add(
        Reservation(
            user_id=USERS[0].id,
            space_id=SPACES[0].id,
            tariff_id=car_tariff.id,
            starts_at=dt.datetime(2026, 1, 15, 9, 0, tzinfo=dt.UTC),
            ends_at=dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC),
            status="confirmed",
            price_cents=car_tariff.price_cents * 3,
            currency=car_tariff.currency,
        )
    )


def main() -> int:
    engine = init_db()
    session_factory = build_session_factory(engine)
    with session_scope(session_factory) as session:
        seed(session)
    print(
        f"Seeded {len(SPACES)} spaces, {len(TARIFFS)} tariffs, "
        f"{len(OPERATING_HOURS)} operating-hours rows, {len(USERS)} user(s), 1 reservation."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
