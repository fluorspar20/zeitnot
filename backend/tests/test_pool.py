from pathlib import Path

import chess
import polars as pl
import pytest

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.positions.pool import POOL_COLUMNS, build, previous_fens, run

# 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O
GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "e1g1"]


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


def board_after(n: int) -> chess.Board:
    board = chess.Board()
    for uci in GAME[:n]:
        board.push_uci(uci)
    return board


def moves_table() -> pl.LazyFrame:
    return pl.DataFrame(
        {"game_id": ["g1"] * len(GAME), "ply": list(range(1, len(GAME) + 1)), "uci": GAME}
    ).lazy()


def observation(pid: str, ply: int, **kw: object) -> dict:
    """A labeled decision before move `ply` of GAME."""
    return {
        "position_id": pid, "fen": board_after(ply - 1).fen(), "opp_prev_uci": GAME[ply - 2],
        "category": "quiet", "phase": "opening", "move_number": (ply + 1) // 2,
        "side": "w" if ply % 2 else "b", "best_uci": "e1g1", "best_win_pct": 55.0,
        "n_acceptable_lines": 3, "gap_1_2": 1.0, "game_id": "g1", "ply": ply,
        "month": "2026-06", "tc_class": "blitz", "mover_elo": 1500,
        "loss_win_pct": 0.0, "position_state": "balanced", **kw,
    }  # fmt: skip


def labels(*pids: str) -> pl.LazyFrame:
    line = {"uci": "e1g1", "cp": 20, "mate": None, "win_pct": 55.0}
    return pl.DataFrame({"position_id": list(pids), "lines": [[line]] * len(pids)}).lazy()


def test_previous_fens_replays_to_the_position_before_the_opponents_move() -> None:
    wanted = pl.DataFrame(
        {"game_id": ["g1", "g1"], "ply": [7, 9]}, schema_overrides={"ply": pl.Int16}
    )
    out = {r["ply"]: r for r in previous_fens(wanted, moves_table()).to_dicts()}
    assert out[7]["replayed_fen"] == board_after(6).fen()
    assert out[7]["prev_fen"] == board_after(5).fen()
    # Before 5. O-O: the opponent's last move was the capture dxc6; the previous position
    # still has White's bishop on c6, which cannot be recovered from the current FEN.
    assert out[9]["prev_fen"] == board_after(7).fen()
    assert chess.Board(out[9]["prev_fen"]).piece_at(chess.C6) == chess.Piece.from_symbol("B")


def test_build_selects_dedupes_and_attaches_lines(cfg: PipelineConfig) -> None:
    obs = pl.DataFrame([
        observation("A", 9),
        observation("A", 9, game_id="g0"),  # same position seen in another game: dropped
        observation("B", 7, category="forced"),  # not a training position
        observation("C", 5, position_state="winning"),  # already decided
        observation("D", 6, loss_win_pct=None),  # unlabeled
        observation("E", 6),
    ])  # fmt: skip
    obs = obs.with_columns(pl.col("ply").cast(pl.Int16))
    moves = pl.concat([moves_table(), moves_table().with_columns(pl.lit("g0").alias("game_id"))])
    pool = build(obs, labels("A", "E"), moves, cfg)
    assert pool.columns == POOL_COLUMNS
    assert pool["position_id"].to_list() == ["A", "E"]
    a = pool.row(0, named=True)
    assert a["source_game_id"] == "g0"  # first source by game id
    assert a["prev_fen"] == board_after(7).fen() and a["lines"][0]["uci"] == "e1g1"


def test_build_rejects_a_replay_mismatch(cfg: PipelineConfig) -> None:
    obs = pl.DataFrame([observation("A", 9, fen=board_after(2).fen())]).with_columns(
        pl.col("ply").cast(pl.Int16)
    )
    with pytest.raises(ValueError, match="mismatch"):
        build(obs, labels("A"), moves_table(), cfg)


def test_run_writes_pool_and_report(tmp_path: Path, cfg: PipelineConfig) -> None:
    obs_path = tmp_path / "positions" / "labeled_observations.parquet"
    obs_path.parent.mkdir()
    pl.DataFrame([observation("A", 9)]).with_columns(pl.col("ply").cast(pl.Int16)).write_parquet(
        obs_path
    )
    (tmp_path / "positions" / "labels").mkdir()
    labels("A").collect().write_parquet(tmp_path / "positions" / "labels" / "part-000000.parquet")
    (tmp_path / "moves" / "month=2026-06").mkdir(parents=True)
    moves_table().collect().write_parquet(tmp_path / "moves" / "month=2026-06" / "part-0.parquet")
    report = run(
        obs_path, tmp_path / "positions" / "labels", tmp_path / "moves", tmp_path / "out", cfg
    )
    assert report["positions"] == 1 and report["by_category"] == {"quiet": 1}
    assert (tmp_path / "out" / "trainer_pool.parquet").exists()
