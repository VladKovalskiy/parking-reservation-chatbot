"""Interactive, deterministic collection of booking-request fields.

No LLM involved: a caller (a future graph node, a form, a test) supplies
already-extracted field values turn by turn — this module only validates
each value and decides what to ask for next. Once first name, last name,
license plate, and a valid time period are all present, the draft is
persisted as a `Reservation` row with `status='draft'` (see
docs/sql-schema.md's "status = 'draft'" and `db/models.py`): no space or
tariff is matched yet, that happens in a later step.
"""

import datetime as dt
import re
from dataclasses import dataclass, replace

from sqlalchemy.orm import Session

from parking_bot.db.models import Reservation, User

FIELD_ORDER = ("first_name", "last_name", "license_plate", "starts_at", "ends_at")

QUESTIONS = {
    "first_name": "What's your first name?",
    "last_name": "What's your last name?",
    "license_plate": "What's your vehicle's license plate number?",
    "starts_at": "When would you like the reservation to start? (e.g. 2026-03-01T09:00)",
    "ends_at": "When would you like the reservation to end? (e.g. 2026-03-01T12:00)",
}

# ASCII-only, matching ADR-005 (product language is English).
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,99}$")
_PLATE_RE = re.compile(r"^[A-Z0-9]{2,10}$")

# rules.md#time-limits: "a minimum stay of 1 hour and a maximum continuous
# stay of 14 days."
MIN_DURATION = dt.timedelta(hours=1)
MAX_DURATION = dt.timedelta(days=14)


@dataclass
class BookingFields:
    first_name: str | None = None
    last_name: str | None = None
    license_plate: str | None = None
    starts_at: dt.datetime | None = None
    ends_at: dt.datetime | None = None


@dataclass
class CollectionResult:
    fields: BookingFields
    is_complete: bool
    next_field: str | None
    question: str | None
    errors: dict[str, str]


def _validate_name(raw: str) -> str:
    value = raw.strip()
    if not _NAME_RE.match(value):
        raise ValueError("Please use only letters (and - ' spaces), 1-100 characters.")
    return value


def _validate_license_plate(raw: str) -> str:
    value = raw.strip().upper().replace(" ", "").replace("-", "")
    if not _PLATE_RE.match(value):
        raise ValueError("That doesn't look like a valid license plate (2-10 letters/digits).")
    return value


def _validate_datetime(raw: str) -> dt.datetime:
    try:
        value = dt.datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError("Please provide a date/time like 2026-03-01T09:00.") from exc
    # fromisoformat() happily returns a naive datetime when the input has no
    # UTC offset (e.g. "2026-03-01T09:00"); comparing that against the
    # timezone-aware `now` in _validate_period() then raises TypeError
    # ("can't compare offset-naive and offset-aware datetimes") — confirmed
    # live via scripts/play_booking.py. Treat an offset-less input as UTC
    # rather than forcing users to always type "+00:00".
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value


_VALIDATORS = {
    "first_name": _validate_name,
    "last_name": _validate_name,
    "license_plate": _validate_license_plate,
    "starts_at": _validate_datetime,
    "ends_at": _validate_datetime,
}


def _validate_period(fields: BookingFields, *, now: dt.datetime) -> str | None:
    """Error message for an invalid (now-complete) period, else None."""
    if fields.starts_at is None or fields.ends_at is None:
        return None
    if fields.starts_at < now:
        return "The start time can't be in the past."
    duration = fields.ends_at - fields.starts_at
    if duration < MIN_DURATION:
        return "Reservations must be at least 1 hour long."
    if duration > MAX_DURATION:
        return "Reservations can't be longer than 14 days."
    return None


def collect_booking_input(
    fields: BookingFields,
    updates: dict[str, str],
    *,
    now: dt.datetime | None = None,
) -> CollectionResult:
    """Merge `updates` into `fields`, validate, and decide what's next.

    An invalid update is rejected (the field keeps its previous value, if
    any) and reported in `errors`, rather than raising — a chat-style caller
    needs to keep going and re-ask, not crash. `errors` and `next_field` can
    both be set at once: e.g. an invalid `ends_at` is cleared back to
    missing so its question is asked again.
    """
    now = now or dt.datetime.now(dt.UTC)
    fields = replace(fields)
    errors: dict[str, str] = {}

    for name, raw in updates.items():
        if name not in _VALIDATORS:
            continue
        try:
            setattr(fields, name, _VALIDATORS[name](raw))
        except ValueError as exc:
            errors[name] = str(exc)

    period_error = _validate_period(fields, now=now)
    if period_error:
        errors["ends_at"] = period_error
        fields.ends_at = None

    next_field = next((name for name in FIELD_ORDER if getattr(fields, name) is None), None)
    is_complete = next_field is None and not errors

    question = None
    if not is_complete:
        question = QUESTIONS[next_field or next(iter(errors))]

    return CollectionResult(
        fields=fields,
        is_complete=is_complete,
        next_field=next_field,
        question=question,
        errors=errors,
    )


def persist_draft(session: Session, external_id: str, fields: BookingFields) -> Reservation:
    """Write a complete `BookingFields` as a `status='draft'` Reservation.

    Gets-or-creates the `User` by `external_id`, filling in the collected
    first/last name.
    """
    user = session.query(User).filter_by(external_id=external_id).one_or_none()
    if user is None:
        user = User(external_id=external_id)
        session.add(user)
    user.first_name = fields.first_name
    user.last_name = fields.last_name
    session.flush()

    reservation = Reservation(
        user_id=user.id,
        license_plate=fields.license_plate,
        starts_at=fields.starts_at,
        ends_at=fields.ends_at,
        status="draft",
    )
    session.add(reservation)
    session.flush()
    return reservation


def collect_booking_turn(
    session: Session,
    external_id: str,
    fields: BookingFields,
    updates: dict[str, str],
    *,
    now: dt.datetime | None = None,
) -> CollectionResult:
    """One turn of the interactive intake: validate/ask, or persist once complete."""
    result = collect_booking_input(fields, updates, now=now)
    if result.is_complete:
        persist_draft(session, external_id, result.fields)
    return result
