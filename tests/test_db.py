import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from parking_bot.db.availability import available_spaces
from parking_bot.db.base import Base
from parking_bot.db.models import OperatingHours, Reservation, Space, Tariff, User
from parking_bot.db.seed import OPERATING_HOURS, SPACES, TARIFFS, USERS, seed


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


def test_create_and_read_space(db_session: Session) -> None:
    db_session.add(Space(code="A-01", level=1, vehicle_type="car", is_accessible=True))
    db_session.commit()

    space = db_session.query(Space).filter_by(code="A-01").one()

    assert space.level == 1
    assert space.vehicle_type == "car"
    assert space.is_accessible is True
    assert space.is_active is True


def test_update_space_marks_inactive(db_session: Session) -> None:
    space = Space(code="A-02", level=1, vehicle_type="car")
    db_session.add(space)
    db_session.commit()

    space.is_active = False
    db_session.commit()

    reloaded = db_session.query(Space).filter_by(code="A-02").one()
    assert reloaded.is_active is False


def test_delete_space(db_session: Session) -> None:
    space = Space(code="A-03", level=1, vehicle_type="car")
    db_session.add(space)
    db_session.commit()

    db_session.delete(space)
    db_session.commit()

    assert db_session.query(Space).filter_by(code="A-03").one_or_none() is None


def test_tariff_unique_constraint_rejects_duplicate_window(db_session: Session) -> None:
    kwargs = {
        "vehicle_type": "car",
        "day_type": "weekday",
        "unit": "hour",
        "price_cents": 500,
        "valid_from": dt.date(2026, 1, 1),
    }
    db_session.add(Tariff(**kwargs))
    db_session.commit()

    db_session.add(Tariff(**kwargs))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_operating_hours_check_constraint_rejects_invalid_day_of_week(db_session: Session) -> None:
    db_session.add(OperatingHours(day_of_week=7, opens_at=dt.time(8, 0), closes_at=dt.time(20, 0)))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_reservation_crud_and_relationships(db_session: Session) -> None:
    user = User(external_id="chat-session-1")
    space = Space(code="B-01", level=2, vehicle_type="car")
    tariff = Tariff(
        vehicle_type="car",
        day_type="weekday",
        unit="hour",
        price_cents=500,
        valid_from=dt.date(2026, 1, 1),
    )
    db_session.add_all([user, space, tariff])
    db_session.flush()

    reservation = Reservation(
        user_id=user.id,
        space_id=space.id,
        tariff_id=tariff.id,
        starts_at=dt.datetime(2026, 2, 1, 9, 0),
        ends_at=dt.datetime(2026, 2, 1, 11, 0),
        status="pending_confirmation",
        price_cents=1000,
        currency="USD",
    )
    db_session.add(reservation)
    db_session.commit()

    assert reservation.space.code == "B-01"
    assert reservation.user.external_id == "chat-session-1"

    reservation.status = "confirmed"
    db_session.commit()
    assert db_session.get(Reservation, reservation.id).status == "confirmed"

    db_session.delete(reservation)
    db_session.commit()
    assert db_session.get(Reservation, reservation.id) is None


def test_reservation_check_constraint_rejects_end_before_start(db_session: Session) -> None:
    user = User(external_id="chat-session-2")
    space = Space(code="B-02", level=2, vehicle_type="car")
    db_session.add_all([user, space])
    db_session.flush()

    db_session.add(
        Reservation(
            user_id=user.id,
            space_id=space.id,
            starts_at=dt.datetime(2026, 2, 1, 11, 0),
            ends_at=dt.datetime(2026, 2, 1, 9, 0),
            status="pending_confirmation",
            price_cents=1000,
            currency="USD",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_available_spaces_excludes_a_space_with_an_overlapping_reservation(
    db_session: Session,
) -> None:
    user = User(external_id="chat-session-3")
    free_space = Space(code="C-01", level=1, vehicle_type="car")
    booked_space = Space(code="C-02", level=1, vehicle_type="car")
    db_session.add_all([user, free_space, booked_space])
    db_session.flush()

    db_session.add(
        Reservation(
            user_id=user.id,
            space_id=booked_space.id,
            starts_at=dt.datetime(2026, 2, 1, 9, 0),
            ends_at=dt.datetime(2026, 2, 1, 12, 0),
            status="confirmed",
            price_cents=1500,
            currency="USD",
        )
    )
    db_session.commit()

    results = available_spaces(
        db_session,
        vehicle_type="car",
        window_start=dt.datetime(2026, 2, 1, 10, 0),
        window_end=dt.datetime(2026, 2, 1, 11, 0),
    )

    codes = {space.code for space in results}
    assert "C-01" in codes
    assert "C-02" not in codes


def test_available_spaces_ignores_a_cancelled_reservation(db_session: Session) -> None:
    user = User(external_id="chat-session-4")
    space = Space(code="C-03", level=1, vehicle_type="car")
    db_session.add_all([user, space])
    db_session.flush()

    db_session.add(
        Reservation(
            user_id=user.id,
            space_id=space.id,
            starts_at=dt.datetime(2026, 2, 1, 9, 0),
            ends_at=dt.datetime(2026, 2, 1, 12, 0),
            status="cancelled",
            price_cents=1500,
            currency="USD",
        )
    )
    db_session.commit()

    results = available_spaces(
        db_session,
        vehicle_type="car",
        window_start=dt.datetime(2026, 2, 1, 10, 0),
        window_end=dt.datetime(2026, 2, 1, 11, 0),
    )

    assert "C-03" in {space.code for space in results}


def test_seed_populates_expected_row_counts(db_session: Session) -> None:
    seed(db_session)
    db_session.commit()

    assert db_session.query(Space).count() == len(SPACES)
    assert db_session.query(Tariff).count() == len(TARIFFS)
    assert db_session.query(OperatingHours).count() == len(OPERATING_HOURS)
    assert db_session.query(User).count() == len(USERS)
    assert db_session.query(Reservation).count() == 1
