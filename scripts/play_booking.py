"""Interactive playground for the booking-field intake flow (booking/collector.py).

Run:  uv run python scripts/play_booking.py

Asks for first name, last name, license plate, and a time period one field
at a time — validating each answer and re-asking on invalid input, exactly
like a future chat-driven booking flow would (see README's "Reservation
intake" section). Uses a throwaway local SQLite file (not Postgres), so this
needs no `make up`; delete `data/play_booking.db` to start fresh.

Not a pytest test, not run in CI — a manual toy for trying the collector.
"""

import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from parking_bot.booking.collector import BookingFields, collect_booking_turn
from parking_bot.db.base import Base
from parking_bot.db.models import Reservation, User

DB_PATH = Path("data/play_booking.db")
EXTERNAL_ID = "play-session"


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    session = Session(engine)

    print("Parking bot: Hi! Let's get your reservation details.")
    print(f"(throwaway local database at {DB_PATH} -- delete it to start fresh)\n")

    result = collect_booking_turn(session, EXTERNAL_ID, BookingFields(), {})

    while not result.is_complete:
        for message in result.errors.values():
            print(f"Parking bot: Hmm, {message}")
        print(f"Parking bot: {result.question}")

        try:
            answer = input("You: ").strip()
        except EOFError:
            print("\n(no more input — exiting)")
            return 1
        if not answer:
            continue

        result = collect_booking_turn(
            session, EXTERNAL_ID, result.fields, {result.next_field: answer}
        )

    print("\nParking bot: Got everything, thanks! Here's your draft reservation:\n")
    reservation = (
        session.query(Reservation)
        .join(User)
        .filter(User.external_id == EXTERNAL_ID)
        .order_by(Reservation.id.desc())
        .first()
    )
    user = reservation.user
    print(f"  Name:          {user.first_name} {user.last_name}")
    print(f"  License plate: {reservation.license_plate}")
    print(f"  Period:        {reservation.starts_at} -> {reservation.ends_at}")
    print(f"  Status:        {reservation.status}  (no space/tariff matched yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
