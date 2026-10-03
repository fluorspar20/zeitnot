from pathlib import Path

import polars as pl
import pytest
import zstandard

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.data import parse_games, stream_filter
from zeitnot.data.clocks import TimeControl
from zeitnot.data.parse_games import GameParseError, parse_game_moves

FIXTURE = Path(__file__).parent / "fixtures" / "filter.pgn"


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def parsed(tmp_path_factory: pytest.TempPathFactory, cfg: PipelineConfig) -> Path:
    """Run the fixture through stream_filter and parse_games once."""
    root = tmp_path_factory.mktemp("pipeline")
    zst = root / "lichess_db_standard_rated_2026-08.pgn.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(FIXTURE.read_bytes()))
    stream_filter.run(
        zst, root / "filtered", "2026-08", cfg, workers=1, chunk_mb=1, max_in_flight=2
    )
    parse_games.run(root / "filtered", root, "2026-08", cfg, workers=1)
    return root


def test_games_table(parsed: Path) -> None:
    games = pl.read_parquet(parsed / "games" / "month=2026-08" / "*.parquet").sort("game_id")
    assert games["game_id"].to_list() == ["keepBltz", "keepClas", "keepSwis"]
    assert games["n_plies"].to_list() == [7, 2, 3]
    assert games["est_duration_s"].to_list() == [260, 1800, 600]  # base + 40 x increment
    assert games["month"].unique().to_list() == ["2026-08"]
    assert "movetext" not in games.columns


def test_moves_of_blitz_game(parsed: Path) -> None:
    moves = (
        pl.read_parquet(parsed / "moves" / "month=2026-08" / "*.parquet")
        .filter(pl.col("game_id") == "keepBltz")
        .sort("ply")
    )
    # 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6?? 4. Qxf7#
    assert moves["uci"].to_list() == ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    # 180+2. White clocks 180, 178, 177, 176; Black 180, 179, 178.
    # T = before - after + 2: 180-178+2, 180-179+2, 178-177+2, 179-178+2, 177-176+2
    assert moves["time_spent_s"].to_list() == [None, None, 4, 3, 3, 3, 3]
    assert moves["opp_clk_s"].to_list() == [180, 180, 180, 178, 179, 177, 178]
    assert moves["eval_white_cp"].to_list() == [20, 20, -10, 0, 0, None, None]
    assert moves["eval_white_mate"].to_list() == [None, None, None, None, None, 1, None]
    assert moves["side"].to_list() == ["w", "b", "w", "b", "w", "b", "w"]


def test_report_and_resume(parsed: Path, cfg: PipelineConfig) -> None:
    report = parse_games.run(parsed / "filtered", parsed, "2026-08", cfg, workers=1)
    assert report["last_run"]["parts_processed"] == 0  # everything already done
    assert report["outcomes"] == {"parsed": 3}
    assert report["moves"] == 12
    assert (parsed / "moves" / "month=2026-08" / "_report.md").exists()
    assert {r["tc_class"] for r in report["quality"]["time_spent_by_class_band"]} == {
        "blitz",
        "rapid",
    }


def test_illegal_move_and_missing_clock(cfg: PipelineConfig) -> None:
    tc = TimeControl(180, 0)
    with pytest.raises(GameParseError, match="illegal_move"):
        parse_game_moves("x", "1. e5 { [%clk 0:03:00] } *", tc, "blitz", cfg.clock)
    with pytest.raises(GameParseError, match="missing_clock"):
        parse_game_moves("x", "1. e4 { [%clk 0:03:00] } 1... e5 *", tc, "blitz", cfg.clock)
