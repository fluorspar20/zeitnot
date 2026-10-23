"""App database tables (SQLAlchemy ORM). Change them only together with an Alembic migration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSON,
        list[dict[str, Any]]: JSON,
    }


class User(Base):
    """Single user in v1, but every user-owned row carries user_id."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    rating: Mapped[int]  # Lichess scale; entered by the user or the default
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    target_offset: Mapped[int]
    default_class: Mapped[str] = mapped_column(String(20))
    strictness: Mapped[str] = mapped_column(String(10))  # "median" | "strict"
    category_mix: Mapped[dict[str, Any]]  # category -> relative weight
    continue_after_budget: Mapped[bool]
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Position(Base):
    """A servable position from the offline pool. Never deleted (attempts refer to it)."""

    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    fen: Mapped[str] = mapped_column(String(100))
    prev_fen: Mapped[str] = mapped_column(String(100))  # before the opponent's last move
    opp_prev_uci: Mapped[str] = mapped_column(String(5))
    category: Mapped[str] = mapped_column(String(20))
    phase: Mapped[str] = mapped_column(String(12))
    move_number: Mapped[int]
    side: Mapped[str] = mapped_column(String(1))
    best_uci: Mapped[str] = mapped_column(String(5))
    best_win_pct: Mapped[float]
    lines: Mapped[list[dict[str, Any]]]  # engine candidates: uci, cp, mate, win_pct
    n_acceptable_lines: Mapped[int]
    gap_1_2: Mapped[float | None]
    source_game_id: Mapped[str] = mapped_column(String(16))
    source_month: Mapped[str] = mapped_column(String(7))
    source_tc_class: Mapped[str] = mapped_column(String(20))
    source_rating: Mapped[int]
    active: Mapped[bool] = mapped_column(default=True)

    __table_args__ = (Index("ix_positions_serving", "active", "category", "source_rating"),)


class CategoryBudget(Base):
    """Budgets that depend on the category, not the individual position (model M0)."""

    __tablename__ = "category_budgets"

    model: Mapped[str] = mapped_column(String(20), primary_key=True)
    tc_class: Mapped[str] = mapped_column(String(20), primary_key=True)
    category: Mapped[str] = mapped_column(String(20), primary_key=True)
    rating: Mapped[int] = mapped_column(primary_key=True)
    budget_s: Mapped[float]
    budget_strict_s: Mapped[float]
    p_inf: Mapped[float]
    sparse: Mapped[bool]


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    position_id: Mapped[str] = mapped_column(ForeignKey("positions.id"), index=True)
    tc_class: Mapped[str] = mapped_column(String(20))
    target_rating: Mapped[int]
    budget_model: Mapped[str] = mapped_column(String(20))
    strictness: Mapped[str] = mapped_column(String(10))
    budget_s: Mapped[float]
    time_ms: Mapped[int]
    move_uci: Mapped[str] = mapped_column(String(5))
    played_win_pct: Mapped[float]
    loss_win_pct: Mapped[float]
    acceptable: Mapped[bool]
    is_best: Mapped[bool]
    within_budget: Mapped[bool]
    graded_by: Mapped[str] = mapped_column(String(10))  # "lines" | "engine"
    mode: Mapped[str] = mapped_column(String(20))
    propensity: Mapped[float]  # probability the serving rule picked this position
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
