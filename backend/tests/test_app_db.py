from pathlib import Path

import polars as pl
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from zeitnot.api.db import make_engine
from zeitnot.api.load import ensure_user, load_budgets, load_positions, migrate, run
from zeitnot.api.tables import Attempt, Base, CategoryBudget, Position, User, UserSettings
from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.positions.pool import POOL_COLUMNS

FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture
def url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'app.db').as_posix()}"


@pytest.fixture
def engine(url: str) -> Engine:
    migrate(url)
    engine = make_engine(url)
    yield engine
    engine.dispose()


def pool(*ids: str, category: str = "quiet") -> pl.DataFrame:
    line = {"uci": "b8c6", "cp": 20, "mate": None, "win_pct": 51.8}
    rows = [
        {
            "position_id": pid, "fen": FEN, "prev_fen": FEN, "opp_prev_uci": "g1f3",
            "category": category, "phase": "opening", "move_number": 2, "side": "b",
            "best_uci": "b8c6", "best_win_pct": 51.8, "lines": [line],
            "n_acceptable_lines": 3, "gap_1_2": None, "source_game_id": "g1",
            "source_month": "2026-06", "source_tc_class": "blitz", "source_rating": 1500,
        }
        for pid in ids
    ]  # fmt: skip
    return pl.DataFrame(rows).select(POOL_COLUMNS)


def budgets(budget: float) -> pl.DataFrame:
    return pl.DataFrame({
        "tc_class": ["blitz", "rapid"], "category": ["quiet", "quiet"], "rating": [1500, 1500],
        "budget_s": [budget, 2 * budget], "budget_strict_s": [budget / 2, budget],
        "p_inf": [0.6, 0.65], "sparse": [False, False], "half_width": [50, 50],
    })  # fmt: skip


def test_migrations_match_the_table_definitions(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_positions_upsert_and_deactivate(engine: Engine) -> None:
    assert load_positions(engine, pool("A", "B")) == {"pool": 2, "added": 2, "deactivated": 0}
    assert load_positions(engine, pool("A", "B")) == {"pool": 2, "added": 0, "deactivated": 0}
    # B leaves the pool and A changes category: B stays (attempts may refer to it), inactive.
    report = load_positions(engine, pool("A", category="tactical"))
    assert report == {"pool": 1, "added": 0, "deactivated": 1}
    with Session(engine) as s:
        rows = {p.id: p for p in s.scalars(select(Position))}
    assert rows["A"].active and rows["A"].category == "tactical"
    assert not rows["B"].active
    assert rows["A"].lines[0]["uci"] == "b8c6"
    # Back in the pool -> active again.
    load_positions(engine, pool("A", "B"))
    with Session(engine) as s:
        assert s.get(Position, "B").active


def test_budgets_are_replaced_per_model(engine: Engine) -> None:
    assert load_budgets(engine, budgets(5.0), "m0") == 2
    load_budgets(engine, budgets(7.0), "m1")
    load_budgets(engine, budgets(6.0), "m0")
    with Session(engine) as s:
        rows = s.scalars(select(CategoryBudget).order_by(CategoryBudget.model)).all()
    assert [(r.model, r.tc_class, r.budget_s) for r in rows if r.tc_class == "blitz"] == [
        ("m0", "blitz", 6.0),
        ("m1", "blitz", 7.0),
    ]


def test_single_user_with_defaults(engine: Engine, cfg: PipelineConfig) -> None:
    uid = ensure_user(engine, cfg)
    assert ensure_user(engine, cfg) == uid
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(User)) == 1
        user, settings = s.get(User, uid), s.get(UserSettings, uid)
    assert user.rating == cfg.trainer.default_rating
    assert settings.target_offset == cfg.target.rating_offset
    assert settings.default_class == "blitz" and settings.strictness == "median"
    assert set(settings.category_mix) == {
        "tactical", "defensive", "only-move", "endgame", "many-good-moves", "quiet",
    }  # fmt: skip


def test_foreign_keys_are_enforced(engine: Engine) -> None:
    with Session(engine) as s:
        s.add(Attempt(
            user_id=99, position_id="nope", tc_class="blitz", target_rating=1800,
            budget_model="m0", strictness="median", budget_s=5.0, time_ms=4000,
            move_uci="e2e4", played_win_pct=50.0, loss_win_pct=0.0, acceptable=True,
            is_best=True, within_budget=True, graded_by="lines", mode="mix", propensity=0.1,
        ))  # fmt: skip
        with pytest.raises(IntegrityError):
            s.commit()


def test_run_end_to_end(tmp_path: Path, url: str, cfg: PipelineConfig) -> None:
    pool("A").write_parquet(tmp_path / "pool.parquet")
    budgets(5.0).write_parquet(tmp_path / "budgets.parquet")
    report = run(url, tmp_path / "pool.parquet", tmp_path / "budgets.parquet", "m0", cfg)
    assert report["positions"]["added"] == 1 and report["budget_rows"] == 2
    again = run(url, tmp_path / "pool.parquet", tmp_path / "budgets.parquet", "m0", cfg)
    assert again["positions"]["added"] == 0 and again["user_id"] == report["user_id"]
