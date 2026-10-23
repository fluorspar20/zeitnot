"""load_app_db job: bring the app database up to date and load the offline artifacts.

1. Apply Alembic migrations (creates the database on first run).
2. Upsert the position pool: new positions are added, existing ones updated, and
   positions no longer in the pool are marked inactive rather than deleted, because
   stored attempts refer to them.
3. Replace the budget table of the given model.
4. Create the single local user with default settings if there is none.

Re-running with the same inputs leaves the database unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from zeitnot.api.db import make_engine
from zeitnot.api.tables import CategoryBudget, Position, User, UserSettings
from zeitnot.config import PipelineConfig, get_settings
from zeitnot.positions.categories import TRAINABLE

log = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"
CHUNK_ROWS = 5000  # rows per INSERT statement batch
BUDGET_COLUMNS = [
    "tc_class",
    "category",
    "rating",
    "budget_s",
    "budget_strict_s",
    "p_inf",
    "sparse",
]


def migrate(url: str) -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


def _chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [rows[i : i + CHUNK_ROWS] for i in range(0, len(rows), CHUNK_ROWS)]


def load_positions(engine: Engine, pool: pl.DataFrame) -> dict[str, int]:
    rows = pool.with_columns(pl.lit(True).alias("active")).to_dicts()
    dialect = {"sqlite": sqlite, "postgresql": postgresql}[engine.dialect.name]
    stmt = dialect.insert(Position)
    upsert = stmt.on_conflict_do_update(
        index_elements=[Position.id],
        set_={c.name: stmt.excluded[c.name] for c in Position.__table__.columns if c.name != "id"},
    )
    ids = set(pool["position_id"].to_list())
    with Session(engine) as session, session.begin():
        before = session.scalar(select(func.count()).select_from(Position)) or 0
        for chunk in _chunks([{"id": r.pop("position_id"), **r} for r in rows]):
            session.execute(upsert, chunk)
        stale = [
            pid for pid in session.scalars(select(Position.id).where(Position.active))
            if pid not in ids
        ]  # fmt: skip
        for i in range(0, len(stale), CHUNK_ROWS):
            session.execute(
                update(Position)
                .where(Position.id.in_(stale[i : i + CHUNK_ROWS]))
                .values(active=False)
            )
        after = session.scalar(select(func.count()).select_from(Position)) or 0
    return {"pool": len(ids), "added": after - before, "deactivated": len(stale)}


def load_budgets(engine: Engine, table: pl.DataFrame, model: str) -> int:
    rows = table.select(BUDGET_COLUMNS).with_columns(pl.lit(model).alias("model")).to_dicts()
    with Session(engine) as session, session.begin():
        session.execute(delete(CategoryBudget).where(CategoryBudget.model == model))
        for chunk in _chunks(rows):
            session.execute(insert(CategoryBudget), chunk)
    return len(rows)


def default_settings(cfg: PipelineConfig, user_id: int) -> UserSettings:
    return UserSettings(
        user_id=user_id,
        target_offset=cfg.target.rating_offset,
        default_class=cfg.time_controls.supported[0],
        strictness="median",
        category_mix=dict.fromkeys(TRAINABLE, 1.0),
        continue_after_budget=True,
    )


def ensure_user(engine: Engine, cfg: PipelineConfig) -> int:
    with Session(engine) as session, session.begin():
        user = session.scalars(select(User).order_by(User.id)).first()
        if user is None:
            user = User(name="local", rating=cfg.trainer.default_rating)
            session.add(user)
            session.flush()
            session.add(default_settings(cfg, user.id))
        return user.id


def run(
    url: str, pool_path: Path, budgets_path: Path, model: str, cfg: PipelineConfig
) -> dict[str, Any]:
    migrate(url)
    engine = make_engine(url)
    try:
        positions = load_positions(engine, pl.read_parquet(pool_path))
        budgets = load_budgets(engine, pl.read_parquet(budgets_path), model)
        user_id = ensure_user(engine, cfg)
    finally:
        engine.dispose()
    return {"positions": positions, "budget_rows": budgets, "model": model, "user_id": user_id}


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", type=Path, default=settings.processed_dir / "trainer_pool.parquet")
    p.add_argument("--budgets", type=Path, default=settings.processed_dir / "budgets_m0.parquet")
    p.add_argument("--model", default="m0")
    p.add_argument("--db", help="database URL (default: ZEITNOT_DATABASE_URL or data/app)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.db or str(settings.database_url), args.pool, args.budgets, args.model, cfg)
    log.info("%s", json.dumps(report))
    return 0
