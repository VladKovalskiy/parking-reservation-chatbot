import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from parking_bot.booking.collector import (
    QUESTIONS,
    BookingFields,
    collect_booking_input,
    collect_booking_turn,
    persist_draft,
)
from parking_bot.db.base import Base
from parking_bot.db.models import Reservation, User

NOW = dt.datetime(2026, 2, 1, 8, 0, tzinfo=dt.UTC)

VALID_UPDATES = {
    "first_name": "John",
    "last_name": "Smith",
    "license_plate": "ab12 cde",
    "starts_at": "2026-02-10T09:00+00:00",
    "ends_at": "2026-02-10T12:00+00:00",
}


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


def test_collect_booking_turn_happy_path_persists_a_draft(db_session: Session) -> None:
    result = collect_booking_turn(
        db_session, "chat-session-1", BookingFields(), VALID_UPDATES, now=NOW
    )

    assert result.is_complete is True
    assert result.question is None
    assert result.errors == {}

    reservation = db_session.query(Reservation).one()
    assert reservation.status == "draft"
    assert reservation.license_plate == "AB12CDE"
    assert reservation.space_id is None
    assert reservation.price_cents is None

    user = db_session.query(User).one()
    assert user.external_id == "chat-session-1"
    assert user.first_name == "John"
    assert user.last_name == "Smith"


def test_collect_booking_input_with_incomplete_updates_asks_for_next_missing_field() -> None:
    result = collect_booking_input(BookingFields(), {"first_name": "John"}, now=NOW)

    assert result.is_complete is False
    assert result.next_field == "last_name"
    assert result.question == QUESTIONS["last_name"]
    assert result.errors == {}


def test_collect_booking_turn_does_not_persist_when_input_is_incomplete(
    db_session: Session,
) -> None:
    result = collect_booking_turn(
        db_session, "chat-session-2", BookingFields(), {"first_name": "John"}, now=NOW
    )

    assert result.is_complete is False
    assert db_session.query(Reservation).count() == 0
    assert db_session.query(User).count() == 0


def test_collect_booking_input_rejects_invalid_license_plate() -> None:
    updates = {**VALID_UPDATES, "license_plate": "!!!"}

    result = collect_booking_input(BookingFields(), updates, now=NOW)

    assert result.is_complete is False
    assert result.next_field == "license_plate"
    assert "license_plate" in result.errors
    assert result.fields.license_plate is None


def test_collect_booking_input_rejects_a_period_shorter_than_one_hour() -> None:
    updates = {
        **VALID_UPDATES,
        "starts_at": "2026-02-10T09:00+00:00",
        "ends_at": "2026-02-10T09:30+00:00",
    }

    result = collect_booking_input(BookingFields(), updates, now=NOW)

    assert result.is_complete is False
    assert result.next_field == "ends_at"
    assert "1 hour" in result.errors["ends_at"]
    assert result.fields.ends_at is None


def test_collect_booking_input_rejects_a_start_time_in_the_past() -> None:
    updates = {
        **VALID_UPDATES,
        "starts_at": "2026-01-01T09:00+00:00",
        "ends_at": "2026-01-01T12:00+00:00",
    }

    result = collect_booking_input(BookingFields(), updates, now=NOW)

    assert result.is_complete is False
    assert "past" in result.errors["ends_at"]


def test_collect_booking_input_accepts_a_datetime_without_a_utc_offset() -> None:
    """Regression test: `datetime.fromisoformat()` returns a naive datetime
    when the input has no UTC offset (e.g. "2026-02-10T09:00" instead of
    "...+00:00"), and comparing that against the timezone-aware `now` in
    `_validate_period()` used to raise `TypeError: can't compare
    offset-naive and offset-aware datetimes` — confirmed live via
    scripts/play_booking.py when a user typed a plain "HH:MM" answer with
    no offset. Offset-less input is now treated as UTC instead of crashing.
    """
    updates = {**VALID_UPDATES, "starts_at": "2026-02-10T09:00", "ends_at": "2026-02-10T12:00"}

    result = collect_booking_input(BookingFields(), updates, now=NOW)

    assert result.is_complete is True
    assert result.fields.starts_at == dt.datetime(2026, 2, 10, 9, 0, tzinfo=dt.UTC)


def test_collect_booking_turn_accumulates_across_multiple_calls(db_session: Session) -> None:
    step1 = collect_booking_turn(
        db_session, "chat-session-3", BookingFields(), {"first_name": "Jane"}, now=NOW
    )
    assert step1.is_complete is False

    step2 = collect_booking_turn(
        db_session,
        "chat-session-3",
        step1.fields,
        {"last_name": "Doe", "license_plate": "XY9 ZZZ"},
        now=NOW,
    )
    assert step2.is_complete is False
    assert step2.next_field == "starts_at"

    step3 = collect_booking_turn(
        db_session,
        "chat-session-3",
        step2.fields,
        {"starts_at": "2026-02-10T09:00+00:00", "ends_at": "2026-02-10T12:00+00:00"},
        now=NOW,
    )

    assert step3.is_complete is True
    reservation = db_session.query(Reservation).one()
    assert reservation.license_plate == "XY9ZZZ"


def test_persist_draft_reuses_existing_user_by_external_id(db_session: Session) -> None:
    fields = replace_fields(VALID_UPDATES, now=NOW)

    persist_draft(db_session, "chat-session-4", fields)
    persist_draft(db_session, "chat-session-4", fields)

    assert db_session.query(User).count() == 1
    assert db_session.query(Reservation).count() == 2


def replace_fields(updates: dict[str, str], *, now: dt.datetime) -> BookingFields:
    return collect_booking_input(BookingFields(), updates, now=now).fields
