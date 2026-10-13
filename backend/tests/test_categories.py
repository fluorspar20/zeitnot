import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.positions.categories import CATEGORIES, TRAINABLE, category_expr

PIPELINE_YAML = REPO_ROOT / "config" / "pipeline.yaml"

# (legal moves, acceptable lines, best win%, phase, opponent just captured, opponent's
#  last move, engine's best move) -> expected category.  Config: sharp <= 2 acceptable,
#  many >= 4, tactical >= 60, defensive <= 40.
CASES = [
    ((1, 1, 50.0, "middlegame", False, "e7e5", "g1f3"), "forced"),
    # The only good move takes back on d5, where the opponent just captured.
    ((30, 1, 75.0, "middlegame", True, "c6d5", "e4d5"), "recapture"),
    # Opponent captured on d5 but the only good move is elsewhere: a real decision.
    ((30, 1, 75.0, "middlegame", True, "c6d5", "g1f3"), "tactical"),
    # Two ways to take back: not the "single acceptable move" case, judged by win%.
    ((30, 2, 70.0, "middlegame", True, "c6d5", "e4d5"), "tactical"),
    ((30, 2, 60.0, "middlegame", False, "e7e5", "g1f3"), "tactical"),  # boundary
    ((30, 2, 40.0, "middlegame", False, "e7e5", "g1f3"), "defensive"),  # boundary
    ((30, 1, 12.0, "endgame", False, "e7e5", "g1f3"), "defensive"),  # sharp beats endgame
    ((30, 1, 50.0, "middlegame", False, "e7e5", "g1f3"), "only-move"),
    ((30, 1, 50.0, "endgame", False, "e7e5", "g1f3"), "only-move"),  # sharp beats endgame
    ((30, 2, 50.0, "middlegame", False, "e7e5", "g1f3"), "quiet"),
    ((30, 2, 50.0, "endgame", False, "e7e5", "g1f3"), "endgame"),
    ((30, 3, 80.0, "opening", False, "e7e5", "g1f3"), "quiet"),
    ((30, 4, 50.0, "middlegame", False, "e7e5", "g1f3"), "many-good-moves"),
    ((30, 5, 95.0, "opening", False, "e7e5", "g1f3"), "many-good-moves"),
    ((30, 5, 50.0, "endgame", False, "e7e5", "g1f3"), "endgame"),
    # Two legal moves, both fine: nothing worse exists, so it is not "few good moves".
    ((2, 2, 80.0, "middlegame", False, "e7e5", "g1f3"), "quiet"),
    # No previous move recorded: never a recapture.
    ((20, 1, 55.0, "opening", None, None, "e2e4"), "only-move"),
]


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(PIPELINE_YAML)


def test_every_rule(cfg: PipelineConfig) -> None:
    cols = ["n_legal_moves", "n_acceptable_lines", "best_win_pct", "phase",
            "opp_prev_was_capture", "opp_prev_uci", "best_uci"]  # fmt: skip
    df = pl.DataFrame(
        [c for c, _ in CASES],
        schema=dict(zip(cols, [pl.Int16, pl.Int8, pl.Float32, pl.String, pl.Boolean, pl.String,
                               pl.String], strict=True)),
        orient="row",
    )  # fmt: skip
    got = df.select(category_expr(cfg.categories).alias("category"))["category"].to_list()
    assert got == [expected for _, expected in CASES]
    assert {expected for _, expected in CASES} == set(CATEGORIES)  # every category exercised


def test_trainable_excludes_non_decisions() -> None:
    assert "forced" not in TRAINABLE and "recapture" not in TRAINABLE
    assert len(TRAINABLE) == 6


def test_thresholds_must_not_overlap() -> None:
    with PIPELINE_YAML.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["categories"]["sharp_max_acceptable"] = 4  # would make "few" and "many" overlap
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)
