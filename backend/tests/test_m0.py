from pathlib import Path

import polars as pl
import pytest
import yaml

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.models.m0 import check, fit, isotonic, km_quantile, run

PIPELINE_YAML = REPO_ROOT / "config" / "pipeline.yaml"


def config(**m0: int) -> PipelineConfig:
    raw = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    raw["budget"]["rating_grid"] = {"min": 1000, "max": 2000, "step": 500}
    raw["budget"]["m0"].update({"min_successes": 4, "window_step": 50, "max_half_width": 200, **m0})
    return PipelineConfig.model_validate(raw)


# --- Kaplan-Meier quantile ---------------------------------------------------


def test_km_without_failures_is_the_interval_quantile() -> None:
    # Four finds at 1..4 s; the step at 2 s (0.25 -> 0.5) is spread over [1.5, 2.5].
    assert km_quantile([1, 2, 3, 4], [True] * 4, [1] * 4, 0.5) == pytest.approx(2.5)
    assert km_quantile([1, 2, 3, 4], [True] * 4, [1] * 4, 0.375) == pytest.approx(2.0)


def test_km_counts_failures_as_not_found_yet() -> None:
    # The player who moved at 2 s without finding anything leaves the risk set:
    # F(1) = 1/4, F(3) = 1 - 3/4 * 1/2 = 0.625 -> q 0.5 lies 2/3 into [2.5, 3.5].
    t, y = [1, 2, 3, 4], [True, False, True, True]
    assert km_quantile(t, y, [1] * 4, 0.5) == pytest.approx(2.5 + 0.25 / 0.375)


def test_km_censoring_later_than_finds_does_not_matter() -> None:
    # Finds at 1 and 2, then a long failed think: by 2 s half had found a move.
    assert km_quantile([1, 2, 30, 30], [True, True, False, False], [1] * 4, 0.5) == 2.5


def test_km_weights_equal_duplicated_rows() -> None:
    a = km_quantile([1, 3, 5], [True, False, True], [2, 1, 1], 0.4)
    b = km_quantile([1, 1, 3, 5], [True, True, False, True], [1, 1, 1, 1], 0.4)
    assert a == pytest.approx(b)


def test_km_unreached_and_bad_input() -> None:
    assert km_quantile([1, 5, 9], [True, False, False], [1] * 3, 0.5) is None
    with pytest.raises(ValueError):
        km_quantile([], [], [], 0.5)
    with pytest.raises(ValueError):
        km_quantile([1], [True], [1], 1.0)


def test_km_first_second_starts_at_zero() -> None:
    # A find recorded as 0 s cannot have taken negative time.
    assert km_quantile([0, 0], [True, True], [1, 1], 0.5) == pytest.approx(0.25)


# --- isotonic regression -----------------------------------------------------


def test_isotonic_pools_violators() -> None:
    assert isotonic([3, 1, 2], [1, 1, 1], increasing=True) == [2, 2, 2]
    assert isotonic([3, 1], [1, 3], increasing=True) == [1.5, 1.5]
    assert isotonic([1, 2, 1], [1, 1, 1], increasing=False) == [1.5, 1.5, 1]
    assert isotonic([5, 4, 4, 1], [1, 2, 3, 4], increasing=False) == [5, 4, 4, 1]


def test_isotonic_preserves_order_between_inputs() -> None:
    # Fits of pointwise-ordered inputs stay ordered (so strict budgets stay <= regular).
    lo, hi, w = [2, 5, 1, 3], [3, 6, 4, 3.5], [1, 2, 1, 1]
    a, b = isotonic(lo, w, increasing=False), isotonic(hi, w, increasing=False)
    assert all(x <= y for x, y in zip(a, b, strict=True))


# --- the budget table --------------------------------------------------------


def observations(rapid_factor: float) -> pl.DataFrame:
    """Quiet, balanced positions; lower ratings take longer and fail more often."""
    rows = []
    for tc, factor in (("blitz", 1.0), ("rapid", rapid_factor)):
        for elo in (1000, 1500, 2000):
            slow = (2500 - elo) / 500  # 3, 2, 1
            for i in range(8):
                rows.append({
                    "tc_class": tc, "category": "quiet", "position_state": "balanced",
                    "mover_elo": elo, "time_spent_s": round((4 + 2 * i) * slow * factor),
                    "y": i % 4 != 3 or elo == 2000, "loss_win_pct": 1.0,
                    "low_clock": False, "sampling_weight": 1.0,
                })  # fmt: skip
    # Excluded from the fit: forced moves, decided positions, low clock, unlabeled.
    extra = {"tc_class": "blitz", "position_state": "balanced", "mover_elo": 1500,
             "time_spent_s": 500.0, "y": True, "loss_win_pct": 1.0, "low_clock": False,
             "sampling_weight": 1.0}  # fmt: skip
    rows += [
        {**extra, "category": "forced"},
        {**extra, "category": "quiet", "position_state": "winning"},
        {**extra, "category": "quiet", "low_clock": True},
        {**extra, "category": "quiet", "loss_win_pct": None},
    ]
    return pl.DataFrame(rows).with_columns(pl.col("time_spent_s").cast(pl.Float32))


def quiet(table: pl.DataFrame, tc: str) -> pl.DataFrame:
    return table.filter((pl.col("tc_class") == tc) & (pl.col("category") == "quiet")).sort("rating")


def test_fit_shapes_and_exclusions() -> None:
    cfg = config()
    table = fit(observations(rapid_factor=2.0), cfg)
    assert check(table.filter(pl.col("category") == "quiet"), cfg) == []
    blitz, rapid = quiet(table, "blitz"), quiet(table, "rapid")
    assert blitz["rating"].to_list() == [1000, 1500, 2000]
    # Each grid rating sees only its own 8 decisions (window +/-50), none of the extras.
    assert blitz["n_obs"].to_list() == [8, 8, 8]
    assert blitz["n_success"].to_list() == [6, 6, 8]
    assert blitz["half_width"].to_list() == [50, 50, 50]
    b = blitz["budget_s"].to_list()
    assert b[0] > b[1] > b[2]
    assert blitz["p_inf"].to_list() == [0.75, 0.75, 1.0]
    assert (rapid["budget_s"] >= blitz["budget_s"]).all()
    assert not table["raised_to_faster"].any()
    # Categories without data have no budget, which check() reports.
    assert any("tactical" in p and "missing" in p for p in check(table, cfg))


def test_fit_raises_slower_class_and_clips() -> None:
    cfg = config()
    table = fit(observations(rapid_factor=0.25), cfg)
    blitz, rapid = quiet(table, "blitz"), quiet(table, "rapid")
    # Rapid players were faster here; a rapid budget may not be shorter than blitz.
    assert rapid["raised_to_faster"].all()
    assert rapid["budget_s"].to_list() == blitz["budget_s"].to_list()
    assert table["budget_s"].min() >= cfg.budget.clip_s["blitz"].min


def test_fit_widens_sparse_windows() -> None:
    cfg = config(min_successes=10)
    blitz = quiet(fit(observations(rapid_factor=2.0), cfg), "blitz")
    # 6-8 finds per rating: the window grows until it reaches a neighbour (+/-500 is
    # beyond max_half_width 200, so it stops there and the row is flagged sparse).
    assert blitz["half_width"].to_list() == [200, 200, 200]
    assert blitz["sparse"].all()


def test_run_writes_table_and_report(tmp_path: Path) -> None:
    cfg = config()
    path = tmp_path / "labeled_observations.parquet"
    observations(rapid_factor=2.0).write_parquet(path)
    report = run(path, tmp_path / "out", cfg)
    assert report["training_rows"] == 48
    assert (tmp_path / "out" / "budgets_m0.parquet").exists()
    assert (tmp_path / "out" / "_budgets_m0_report.md").exists()
    assert (tmp_path / "out" / "_budgets_m0_manifest.json").exists()
