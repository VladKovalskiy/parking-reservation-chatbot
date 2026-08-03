"""SQLAlchemy models mirroring docs/sql-schema.md.

Deliberately no `Availability` model — see "Availability is a query, not a
table" in docs/sql-schema.md. `db/availability.py` answers that question with
a query instead, so it can't drift out of sync with `reservations`.
"""

import datetime as dt

from sqlalchemy import (
    DDL,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    SmallInteger,
    String,
    Time,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from parking_bot.db.base import Base

VEHICLE_TYPES = ("car", "motorcycle", "bicycle")
DAY_TYPES = ("weekday", "weekend", "holiday")
TARIFF_UNITS = ("hour", "day")
RESERVATION_STATUSES = (
    "pending_confirmation",
    "confirmed",
    "cancelled",
    "no_show",
    "completed",
)


class Space(Base):
    """One row per physical, individually reservable spot."""

    __tablename__ = "spaces"
    __table_args__ = (
        CheckConstraint(f"vehicle_type IN {VEHICLE_TYPES}", name="ck_spaces_vehicle_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String, nullable=False)
    is_accessible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    reservations: Mapped[list["Reservation"]] = relationship(back_populates="space")


class Tariff(Base):
    """Price history: a (vehicle_type, day_type, unit) triple over a validity window."""

    __tablename__ = "tariffs"
    __table_args__ = (
        CheckConstraint(f"vehicle_type IN {VEHICLE_TYPES}", name="ck_tariffs_vehicle_type"),
        CheckConstraint(f"day_type IN {DAY_TYPES}", name="ck_tariffs_day_type"),
        CheckConstraint(f"unit IN {TARIFF_UNITS}", name="ck_tariffs_unit"),
        CheckConstraint("price_cents >= 0", name="ck_tariffs_price_non_negative"),
        UniqueConstraint(
            "vehicle_type", "day_type", "unit", "valid_from", name="uq_tariffs_window"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_type: Mapped[str] = mapped_column(String, nullable=False)
    day_type: Mapped[str] = mapped_column(String, nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    price_cents: Mapped[int] = mapped_column(nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    valid_from: Mapped[dt.date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[dt.date | None] = mapped_column(Date, nullable=True)

    reservations: Mapped[list["Reservation"]] = relationship(back_populates="tariff")


class OperatingHours(Base):
    """Regular weekly hours, one row per day_of_week (0 = Sunday)."""

    __tablename__ = "operating_hours"
    __table_args__ = (
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="ck_operating_hours_day_of_week"),
        UniqueConstraint("day_of_week", name="uq_operating_hours_day_of_week"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    day_of_week: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    opens_at: Mapped[dt.time | None] = mapped_column(Time, nullable=True)
    closes_at: Mapped[dt.time | None] = mapped_column(Time, nullable=True)
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class OperatingHoursException(Base):
    """One-off overrides (holidays, closures) checked before the regular weekly hours."""

    __tablename__ = "operating_hours_exceptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    exception_date: Mapped[dt.date] = mapped_column(Date, unique=True, nullable=False)
    opens_at: Mapped[dt.time | None] = mapped_column(Time, nullable=True)
    closes_at: Mapped[dt.time | None] = mapped_column(Time, nullable=True)
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)


class User(Base):
    """Chat/session identity — not a real auth system (see docs/sql-schema.md)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    reservations: Mapped[list["Reservation"]] = relationship(back_populates="user")


class Reservation(Base):
    """A booking.

    The double-booking guard is a Postgres-only EXCLUDE constraint (see the
    `event.listen(...)` calls below): it can't be expressed as a portable
    SQLAlchemy `__table_args__` constraint, so this class only declares the
    columns/CHECKs/FKs that work identically on SQLite (used by unit tests)
    and Postgres; the EXCLUDE guard is layered on separately for Postgres.
    """

    __tablename__ = "reservations"
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="ck_reservations_ends_after_starts"),
        CheckConstraint(f"status IN {RESERVATION_STATUSES}", name="ck_reservations_status"),
        CheckConstraint("price_cents >= 0", name="ck_reservations_price_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    space_id: Mapped[int] = mapped_column(ForeignKey("spaces.id"), nullable=False)
    tariff_id: Mapped[int | None] = mapped_column(ForeignKey("tariffs.id"), nullable=True)
    starts_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending_confirmation")
    price_cents: Mapped[int] = mapped_column(nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    confirmed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    space: Mapped["Space"] = relationship(back_populates="reservations")
    user: Mapped["User"] = relationship(back_populates="reservations")
    tariff: Mapped["Tariff | None"] = relationship(back_populates="reservations")


# --- Postgres-only double-booking guard ---
#
# ExcludeConstraint (sqlalchemy.dialects.postgresql) is a PostgreSQL-only
# construct; declaring it in Reservation.__table_args__ makes
# Base.metadata.create_all() crash on any other dialect (verified against
# SQLite: `UnsupportedCompilationError` for
# `dialects.postgresql.ext.ExcludeConstraint`). Attaching it instead as an
# `after_create` DDL event scoped to `execute_if(dialect="postgresql")` keeps
# SQLite-based unit tests working while still creating the real guard
# against a Postgres database (`init_db.py`, `make up`).
_CREATE_BTREE_GIST = DDL("CREATE EXTENSION IF NOT EXISTS btree_gist")
event.listen(Base.metadata, "before_create", _CREATE_BTREE_GIST.execute_if(dialect="postgresql"))

_ADD_EXCLUDE_CONSTRAINT = DDL(
    "ALTER TABLE reservations ADD CONSTRAINT reservations_no_overlap "
    "EXCLUDE USING gist (space_id WITH =, tstzrange(starts_at, ends_at) WITH &&) "
    "WHERE (status IN ('pending_confirmation', 'confirmed'))"
)
event.listen(
    Reservation.__table__, "after_create", _ADD_EXCLUDE_CONSTRAINT.execute_if(dialect="postgresql")
)
