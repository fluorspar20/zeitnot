from pathlib import Path

import polars as pl
import pytest
import zstandard

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.data import parse_games, puzzles, stream_filter

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def joined(tmp_path_factory: pytest.TempPathFactory, cfg: PipelineConfig):
    root = tmp_path_factory.mktemp("puzzles")
    games_zst = root / "lichess_db_standard_rated_2026-08.pgn.zst"
    games_zst.write_bytes(
        zstandard.ZstdCompressor().compress((FIXTURES / "filter.pgn").read_bytes())
    )
    stream_filter.run(
        games_zst, root / "filtered", "2026-08", cfg, workers=1, chunk_mb=1, max_in_flight=2
    )
    parse_games.run(root / "filtered", root, "2026-08", cfg, workers=1)

    puzzles_zst = root / "lichess_db_puzzle.csv.zst"
    puzzles_zst.write_bytes(
        zstandard.ZstdCompressor().compress((FIXTURES / "puzzles.csv").read_bytes())
    )
    puzzles_path = puzzles.ingest(puzzles_zst, root / "puzzles")
    out = root / "puzzle_positions"
    report = puzzles.join(puzzles_path, root, "2026-08", out, cfg)
    return puzzles_path, report, pl.read_parquet(out / "puzzle_positions.parquet")


def test_ingest_parses_url_and_moves(joined) -> None:
    puzzles_path, _, _ = joined
    df = pl.read_parquet(puzzles_path).sort("puzzle_id")
    row = df.filter(pl.col("puzzle_id") == "PzMate").row(0, named=True)
    assert (row["game_id"], row["setup_ply"], row["solver_side"]) == ("keepBltz", 6, "w")
    assert (row["setup_uci"], row["solution_uci"]) == ("g8f6", "h5f7")
    assert row["themes"] == ["mate", "mateIn1", "oneMove", "opening"]
    assert row["daily_date_ms"] is None
    # No "/black": the setup move is White's, so the solver plays Black.
    assert df.filter(pl.col("puzzle_id") == "PzBadSetup")["solver_side"].item() == "b"
    assert not (puzzles_path.parent / "lichess_db_puzzle.csv").exists()  # temp CSV removed


def test_join_outcomes(joined) -> None:
    _, report, _ = joined
    # PzOther's game is not in our data, so only four puzzles join.
    assert report["puzzles_total"] == 5
    assert report["outcomes"] == {
        "joined": 4,
        "kept": 2,
        "setup_mismatch": 1,  # PzBadSetup: game's first move is e2e4, puzzle says d2d4
        "no_solver_move": 1,  # PzNoReply: the game ended before White answered
    }


def test_solver_move_and_outcome(joined) -> None:
    _, _, df = joined
    mate = df.filter(pl.col("puzzle_id") == "PzMate").row(0, named=True)
    # Source game 180+2: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6?? 4. Qxf7#; White's clocks 177 -> 176.
    assert (mate["ply"], mate["played_uci"], mate["played_solution"]) == (7, "h5f7", True)
    assert mate["time_spent_s"] == 3  # 177 - 176 + 2
    assert (mate["solver_elo"], mate["opponent_elo"], mate["tc_class"]) == (1500, 1400, "blitz")
    assert mate["fen"].startswith("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w")
    assert mate["epd"] == "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -"

    miss = df.filter(pl.col("puzzle_id") == "PzMiss").row(0, named=True)
    # 1. d4 d5 2. c4: the player chose c4 over the "solution" Nf3; 600 - 590 = 10 s.
    assert (miss["played_uci"], miss["played_solution"], miss["time_spent_s"]) == (
        "c2c4",
        False,
        10,
    )
    assert miss["solution_line"] == ["g1f3", "g8f6", "c2c4"]
