"""Static/dynamic split: route a question to SQL or to the RAG chain.

Availability, prices, and operating hours change on their own schedule and
live in Postgres (see docs/sql-schema.md) — answering them from the vector
store risks a stale or subtly-off number, exactly what the static/dynamic
split in CLAUDE.md exists to prevent. Everything else goes through the
grounded RAG chain (`rag/chain.py`).

Classification is deterministic keyword matching, not an LLM call: these
three SQL categories are narrow and identifiable from question phrasing
alone. Same reasoning `guardrails/pii.py` (Presidio, not Haiku) and
`booking/collector.py` (regex-validated slot filling, not Haiku) already
applied to two other "cheap internal steps" named in CLAUDE.md's locked
decisions — a deterministic method is just as reliable here and far cheaper
and easier to test than a model call.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from parking_bot.config import Settings, get_settings
from parking_bot.db.availability import available_spaces
from parking_bot.db.models import OperatingHours, OperatingHoursException, Tariff
from parking_bot.rag.chain import answer_question as answer_from_rag

SqlCategory = Literal["availability", "prices", "hours"]
Destination = Literal["sql", "rag"]

_AVAILABILITY_PHRASES = (
    "available",
    "availability",
    "free spot",
    "free space",
    "any spots",
    "any space",
    "vacant",
    "open spot",
    "open space",
    "is there space",
    "is there a spot",
)
_HOURS_PHRASES = (
    "hours",
    "what time",
    "opening time",
    "closing time",
    "open until",
    "close at",
    "when do you open",
    "when do you close",
    "operating hours",
)
_PRICE_PHRASES = (
    "price",
    "prices",
    "cost",
    "costs",
    "fee",
    "fees",
    "rate",
    "rates",
    "tariff",
    "tariffs",
    "charge",
    "charges",
    "how much",
)


@dataclass
class RouteDecision:
    destination: Destination
    sql_category: SqlCategory | None


def classify_question(question: str) -> RouteDecision:
    """Availability/prices/hours phrasing -> SQL; everything else -> RAG."""
    text = question.lower()

    if any(phrase in text for phrase in _AVAILABILITY_PHRASES):
        return RouteDecision(destination="sql", sql_category="availability")
    if any(phrase in text for phrase in _HOURS_PHRASES):
        return RouteDecision(destination="sql", sql_category="hours")
    if any(phrase in text for phrase in _PRICE_PHRASES):
        return RouteDecision(destination="sql", sql_category="prices")
    return RouteDecision(destination="rag", sql_category=None)


def _day_of_week(date: dt.date) -> int:
    """Python's Monday=0 weekday -> the schema's Sunday=0 convention."""
    return (date.weekday() + 1) % 7


def answer_hours_question(session: Session, *, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.UTC)
    today = now.date()

    exception = session.query(OperatingHoursException).filter_by(exception_date=today).one_or_none()
    row = exception or (
        session.query(OperatingHours).filter_by(day_of_week=_day_of_week(today)).one_or_none()
    )

    if row is None:
        return "Operating hours are not configured for today."
    if row.is_closed:
        return "The garage is closed today."
    return f"Today's hours are {row.opens_at.strftime('%H:%M')}-{row.closes_at.strftime('%H:%M')}."


def answer_prices_question(session: Session, *, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    tariffs = (
        session.query(Tariff)
        .filter(Tariff.valid_from <= today)
        .filter((Tariff.valid_to.is_(None)) | (Tariff.valid_to >= today))
        .order_by(Tariff.vehicle_type, Tariff.day_type)
        .all()
    )
    if not tariffs:
        return "No pricing information is currently available."
    lines = [
        f"{t.vehicle_type.capitalize()} ({t.day_type}): "
        f"{t.price_cents / 100:.2f} {t.currency}/{t.unit}"
        for t in tariffs
    ]
    return "Current prices:\n" + "\n".join(lines)


def answer_availability_question(
    session: Session, *, vehicle_type: str = "car", now: dt.datetime | None = None
) -> str:
    now = now or dt.datetime.now(dt.UTC)
    spaces = available_spaces(
        session, vehicle_type=vehicle_type, window_start=now, window_end=now + dt.timedelta(hours=1)
    )
    count = len(spaces)
    noun = "spot" if count == 1 else "spots"
    verb = "is" if count == 1 else "are"
    return f"There {verb} currently {count} {vehicle_type} {noun} available."


@dataclass
class RoutedAnswer:
    """A final answer plus where it came from, for logging/debugging."""

    answer: str
    destination: Destination
    sql_category: SqlCategory | None
    sources: list[str] = field(default_factory=list)


def answer_dynamic_question(
    question: str,
    session: Session,
    *,
    chat=None,
    settings: Settings | None = None,
    now: dt.datetime | None = None,
    vehicle_type: str = "car",
    **rag_kwargs,
) -> RoutedAnswer:
    """Route `question` to SQL or the RAG chain and merge the result into one answer.

    Extra keyword arguments (`chunks`, `store`, `k`) are forwarded to the RAG
    chain when the question routes there.
    """
    settings = settings or get_settings()
    decision = classify_question(question)

    if decision.destination == "rag":
        rag_result = answer_from_rag(question, chat=chat, settings=settings, **rag_kwargs)
        return RoutedAnswer(
            answer=rag_result.answer,
            destination="rag",
            sql_category=None,
            sources=rag_result.sources,
        )

    if decision.sql_category == "hours":
        text = answer_hours_question(session, now=now)
    elif decision.sql_category == "prices":
        text = answer_prices_question(session, today=now.date() if now else None)
    else:
        text = answer_availability_question(session, vehicle_type=vehicle_type, now=now)

    return RoutedAnswer(
        answer=text, destination="sql", sql_category=decision.sql_category, sources=["postgres"]
    )
