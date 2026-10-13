from pathlib import Path

import polars as pl
import pytest

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.positions.labeling import SCHEMA as LABEL_SCHEMA
from zeitnot.positions.observations import build, run


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


def line(uci: str, win: float) -> dict:
    return {"uci": uci, "cp": 0, "mate": None, "win_pct": win}


def played(uci: str, win: float, best: float, in_multipv: bool) -> dict:
    return {**line(uci, win), "in_multipv": in_multipv, "loss_win_pct": best - win}


def label(pid: str, lines: list[dict], played_moves: list[dict], first_depth: int) -> dict:
    return {
        "position_id": pid, "epd": f"epd-{pid}", "best_uci": lines[0]["uci"],
        "best_win_pct": lines[0]["win_pct"], "lines": lines, "depth": 15, "nodes": 500_000,
        "best_move_first_depth": first_depth, "best_move_stable_depth": first_depth,
        "played": played_moves, "seconds": 1.0,
    }  # fmt: skip


LABELS = pl.DataFrame(
    [
        # Balanced, three lines; tau = 5: lines at 60, 57, 52 -> two acceptable.
        label(
            "A",
            [line("e2e4", 60.0), line("d2d4", 57.0), line("g1f3", 52.0)],
            [played("d2d4", 57.0, 60.0, True), played("a2a3", 40.0, 60.0, False)],
            first_depth=3,
        ),
        # Winning, one legal move.
        label("B", [line("h8g8", 97.0)], [played("h8g8", 97.0, 97.0, True)], first_depth=1),
        # Losing: best line 8%; the played move scored slightly above it (search noise).
        label(
            "C",
            [line("e8e7", 8.0), line("e8d7", 7.5)],
            [played("e8f8", 8.4, 8.0, False)],
            first_depth=1,
        ),
    ],
    schema=LABEL_SCHEMA,
)

POSITIONS = pl.DataFrame(
    {
        "position_id": ["A", "A", "B", "C"],
        "epd": ["epd-A", "epd-A", "epd-B", "epd-C"],
        "played_uci": ["d2d4", "a2a3", "h8g8", "e8f8"],
        "mover_elo": [1500, 900, 2100, 1250],
        "tc_class": ["blitz", "blitz", "rapid", "blitz"],
        "time_spent_s": [4.0, 9.0, 1.0, 6.0],
        "sampling_weight": [100.0, 300.0, 50.0, 50.0],
    }
)


def test_loss_y_and_position_features(cfg: PipelineConfig) -> None:
    obs = build(POSITIONS, LABELS.lazy(), cfg)
    assert obs.height == 4
    rows = {(r["position_id"], r["played_uci"]): r for r in obs.to_dicts()}

    good = rows[("A", "d2d4")]
    assert good["loss_win_pct"] == pytest.approx(3.0) and good["y"] is True
    assert good["is_best"] is False and good["played_in_multipv"] is True
    # Position A: lines 60 / 57 / 52 with tau 5 -> two acceptable; gaps 3 and 8.
    assert (good["n_lines"], good["n_acceptable_lines"]) == (3, 2)
    assert (good["gap_1_2"], good["gap_1_3"]) == (pytest.approx(3.0), pytest.approx(8.0))
    assert good["position_state"] == "balanced"
    assert good["best_move_first_depth"] == 3

    blunder = rows[("A", "a2a3")]
    assert blunder["loss_win_pct"] == pytest.approx(20.0) and blunder["y"] is False

    forced = rows[("B", "h8g8")]
    assert (forced["y"], forced["is_best"], forced["position_state"]) == (True, True, "winning")
    assert forced["n_acceptable_lines"] == 1
    assert forced["gap_1_2"] is None and forced["gap_1_3"] is None  # only one legal move

    noisy = rows[("C", "e8f8")]
    assert noisy["loss_win_pct"] == pytest.approx(-0.4)  # raw, not clamped
    assert noisy["y"] is True and noisy["position_state"] == "losing"
    assert noisy["gap_1_3"] is None


def test_run_report(tmp_path: Path, cfg: PipelineConfig) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    LABELS.write_parquet(labels_dir / "part-000000.parquet")
    positions_path = tmp_path / "sampled_positions.parquet"
    POSITIONS.write_parquet(positions_path)
    # Database knows position A with the played move d2d4 among its lines.
    evaldb = pl.DataFrame(
        {
            "epd": ["epd-A"],
            "best_uci": ["e2e4"],
            "best_knodes": [90_000],
            "pvs": [[{"uci": "e2e4", "cp": 0, "mate": None, "win_pct": 61.0},
                     {"uci": "d2d4", "cp": 0, "mate": None, "win_pct": 59.0}]],
        }
    )  # fmt: skip
    evaldb.write_parquet(tmp_path / "evaldb_matches.parquet")

    report = run(positions_path, labels_dir, tmp_path, cfg, tmp_path / "evaldb_matches.parquet")
    assert (report["observations"], report["labeled"]) == (4, 4)
    assert report["y_rate_sample"] == 0.75
    # Weighted: Y rows carry 100 + 50 + 50 of 500 total weight.
    assert report["y_rate_population"] == pytest.approx(0.4)
    assert report["negative_loss_share"] == 0.25
    check = report["evaldb_check"]
    assert (check["n"], check["same_y"], check["same_best_move"]) == (1, 1.0, 1.0)
    assert check["mean_abs_loss_diff"] == pytest.approx(1.0)  # ours 3.0 vs database 2.0
    obs = pl.read_parquet(tmp_path / "labeled_observations.parquet")
    assert obs.height == 4 and "y" in obs.columns
    assert (tmp_path / "_observations_report.md").exists()
