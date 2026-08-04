import datetime as dt
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from parking_bot.db.base import Base
from parking_bot.db.models import OperatingHours, OperatingHoursException, Space, Tariff
from parking_bot.rag.router import (
    answer_availability_question,
    answer_dynamic_question,
    answer_hours_question,
    answer_prices_question,
    classify_question,
)

NOW = dt.datetime(2026, 2, 3, 10, 0, tzinfo=dt.UTC)  # 2026-02-03 is a Tuesday


class _MockChat:
    def __init__(self, content: str = "mocked rag answer") -> None:
        self._content = content
        self.invoked_with: list | None = None

    def invoke(self, messages: list) -> SimpleNamespace:
        self.invoked_with = messages
        return SimpleNamespace(content=self._content)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


@pytest.mark.parametrize(
    "question,expected_category",
    [
        ("Is there any space available right now?", "availability"),
        ("Do you have a free spot for a motorcycle?", "availability"),
        ("What time do you open?", "hours"),
        ("What are your operating hours on Sunday?", "hours"),
        ("How much does it cost per hour?", "prices"),
        ("What's your rate for bicycles?", "prices"),
    ],
)
def test_classify_question_routes_dynamic_topics_to_sql(
    question: str, expected_category: str
) -> None:
    decision = classify_question(question)

    assert decision.destination == "sql"
    assert decision.sql_category == expected_category


def test_classify_question_routes_everything_else_to_rag() -> None:
    decision = classify_question("How do I cancel an existing reservation?")

    assert decision.destination == "rag"
    assert decision.sql_category is None


def test_answer_prices_question_merges_current_tariffs_into_text(db_session: Session) -> None:
    db_session.add_all(
        [
            Tariff(
                vehicle_type="car",
                day_type="weekday",
                unit="hour",
                price_cents=500,
                currency="USD",
                valid_from=dt.date(2026, 1, 1),
            ),
            Tariff(
                vehicle_type="bicycle",
                day_type="weekday",
                unit="hour",
                price_cents=0,
                currency="USD",
                valid_from=dt.date(2026, 1, 1),
            ),
        ]
    )
    db_session.commit()

    text = answer_prices_question(db_session, today=dt.date(2026, 2, 3))

    assert "Car (weekday): 5.00 USD/hour" in text
    assert "Bicycle (weekday): 0.00 USD/hour" in text


def test_answer_hours_question_merges_todays_operating_hours(db_session: Session) -> None:
    db_session.add(
        OperatingHours(day_of_week=2, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0))
    )  # Tuesday
    db_session.commit()

    text = answer_hours_question(db_session, now=NOW)

    assert text == "Today's hours are 06:00-23:00."


def test_answer_hours_question_prefers_an_exception_over_the_regular_schedule(
    db_session: Session,
) -> None:
    db_session.add(OperatingHours(day_of_week=2, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)))
    db_session.add(OperatingHoursException(exception_date=dt.date(2026, 2, 3), is_closed=True))
    db_session.commit()

    text = answer_hours_question(db_session, now=NOW)

    assert text == "The garage is closed today."


def test_answer_availability_question_counts_free_spaces(db_session: Session) -> None:
    db_session.add_all(
        [
            Space(code="A-01", level=1, vehicle_type="car"),
            Space(code="A-02", level=1, vehicle_type="car"),
        ]
    )
    db_session.commit()

    text = answer_availability_question(db_session, vehicle_type="car", now=NOW)

    assert text == "There are currently 2 car spots available."


def test_answer_dynamic_question_routes_sql_category_without_touching_the_llm(
    db_session: Session,
) -> None:
    db_session.add(
        Tariff(
            vehicle_type="car",
            day_type="weekday",
            unit="hour",
            price_cents=500,
            currency="USD",
            valid_from=dt.date(2026, 1, 1),
        )
    )
    db_session.commit()
    chat = _MockChat()

    result = answer_dynamic_question("How much does parking cost?", db_session, chat=chat, now=NOW)

    assert result.destination == "sql"
    assert result.sql_category == "prices"
    assert "5.00 USD/hour" in result.answer
    assert result.sources == ["postgres"]
    assert chat.invoked_with is None


def test_answer_dynamic_question_classifies_the_raw_question_before_masking(
    db_session: Session,
) -> None:
    """Regression test: masking before classification breaks routing.

    "Working hours?" gets NER-tagged as a single DATE_TIME span and masked
    to "<DATE_TIME>?", losing the word "hours" the classifier keys on — the
    question then wrongly falls through to RAG instead of SQL. Confirmed
    live before this test was added; see CLAUDE.md's Known traps.
    """
    db_session.add(OperatingHours(day_of_week=2, opens_at=dt.time(6, 0), closes_at=dt.time(23, 0)))
    db_session.commit()
    chat = _MockChat()

    result = answer_dynamic_question("Working hours?", db_session, chat=chat, now=NOW)

    assert result.destination == "sql"
    assert result.sql_category == "hours"
    assert result.answer == "Today's hours are 06:00-23:00."
    assert chat.invoked_with is None


def test_answer_dynamic_question_falls_back_to_rag_for_non_dynamic_questions(
    db_session: Session,
) -> None:
    mocked_answer = "Cancel a reservation by asking the chatbot before your arrival window."
    chat = _MockChat(mocked_answer)
    chunks = [
        Document(
            page_content="Reservations can be cancelled any time before arrival.",
            metadata={"doc_id": "booking.md#cancel"},
        )
    ]

    result = answer_dynamic_question(
        "How do I cancel my reservation?",
        db_session,
        chunks=chunks,
        chat=chat,
        now=NOW,
    )

    assert result.destination == "rag"
    assert result.sql_category is None
    assert result.answer == mocked_answer
    assert result.sources == ["booking.md#cancel"]
    assert chat.invoked_with is not None


def test_answer_dynamic_question_masks_pii_in_the_question_sent_to_the_llm(
    db_session: Session,
) -> None:
    chat = _MockChat("I don't have information about that.")
    chunks = [Document(page_content="...", metadata={"doc_id": "general.md#contact"})]

    answer_dynamic_question(
        "My phone number is (212) 555-0198, can you note that down?",
        db_session,
        chunks=chunks,
        chat=chat,
        now=NOW,
    )

    _system_message, human_message = chat.invoked_with
    assert "(212) 555-0198" not in human_message.content
    assert "<PHONE_NUMBER>" in human_message.content
