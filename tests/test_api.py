import datetime as dt
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from parking_bot.api.app import app
from parking_bot.api.dependencies import get_chat_model, get_db_session, get_rag_chunks
from parking_bot.db.base import Base
from parking_bot.db.models import Tariff


class _MockChat:
    def __init__(self, content: str) -> None:
        self._content = content
        self.invoked_with: list | None = None

    def invoke(self, messages: list) -> SimpleNamespace:
        self.invoked_with = messages
        return SimpleNamespace(content=self._content)


@pytest.fixture
def sqlite_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


@pytest.fixture
def client(sqlite_session: Session):
    def _override_session():
        yield sqlite_session

    app.dependency_overrides[get_db_session] = _override_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_endpoint_routes_a_price_question_through_sql(
    client: TestClient, sqlite_session: Session
) -> None:
    sqlite_session.add(
        Tariff(
            vehicle_type="car",
            day_type="weekday",
            unit="hour",
            price_cents=500,
            currency="USD",
            valid_from=dt.date(2026, 1, 1),
        )
    )
    sqlite_session.commit()

    response = client.post(
        "/chat", json={"message": "How much does parking cost?", "session_id": "s1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "sql"
    assert "5.00 USD/hour" in body["answer"]
    assert body["sources"] == ["postgres"]


def test_chat_endpoint_routes_a_general_question_through_rag_and_masks_output_pii(
    client: TestClient,
) -> None:
    # Presidio's built-in EMAIL_ADDRESS regex doesn't match the reserved
    # ".example" TLD used elsewhere in this project's static docs (7 letters,
    # past whatever length its pattern allows) — use a realistic domain here
    # so this test actually exercises masking. See CLAUDE.md's Known traps.
    mocked_answer = "Contact support at office@centralparking.com for help."
    chunks = [
        Document(
            page_content="For questions the chatbot cannot answer, contact the parking office.",
            metadata={"doc_id": "general.md#contact"},
        )
    ]
    app.dependency_overrides[get_chat_model] = lambda: _MockChat(mocked_answer)
    app.dependency_overrides[get_rag_chunks] = lambda: chunks

    response = client.post(
        "/chat", json={"message": "Who do I contact for help?", "session_id": "s2"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "rag"
    assert "office@centralparking.com" not in body["answer"]
    assert "<EMAIL_ADDRESS>" in body["answer"]
    assert body["sources"] == ["general.md#contact"]


def test_chat_endpoint_does_not_mask_sql_answers(
    client: TestClient, sqlite_session: Session
) -> None:
    """Regression test: output PII masking must not touch SQL-templated answers.

    Presidio's NER has no user-typed context to work with on a price list,
    so it free-associates on capitalized domain words instead — "Bicycle"
    as PERSON, "weekday" as DATE_TIME — corrupting a correct answer instead
    of protecting one. Confirmed live against a real deployment before this
    test was added.
    """
    sqlite_session.add_all(
        [
            Tariff(
                vehicle_type="bicycle",
                day_type="weekday",
                unit="hour",
                price_cents=0,
                currency="USD",
                valid_from=dt.date(2026, 1, 1),
            ),
            Tariff(
                vehicle_type="motorcycle",
                day_type="weekday",
                unit="hour",
                price_cents=250,
                currency="USD",
                valid_from=dt.date(2026, 1, 1),
            ),
        ]
    )
    sqlite_session.commit()

    response = client.post(
        "/chat", json={"message": "How much does parking cost?", "session_id": "s4"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Current prices:\nBicycle (weekday): 0.00 USD/hour\nMotorcycle (weekday): 2.50 USD/hour"
    )


def test_chat_endpoint_rejects_an_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "", "session_id": "s3"})

    assert response.status_code == 422


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
