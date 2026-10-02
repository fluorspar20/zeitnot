"""Smoke tests against a real Stockfish binary; skipped when none is installed."""

import chess
import pytest

from zeitnot.config import REPO_ROOT, PipelineConfig, Settings
from zeitnot.engine.stockfish import (
    StockfishNotFoundError,
    analyse_multipv,
    find_stockfish,
    open_engine,
)

pytestmark = pytest.mark.engine

SMOKE_NODES = 200_000


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def engine(cfg):
    try:
        path = find_stockfish(Settings(_env_file=None))
    except StockfishNotFoundError as e:
        pytest.skip(str(e))
    with open_engine(path, cfg.engine) as eng:
        yield eng


def test_finds_mate_in_one(engine, cfg) -> None:
    # Scholar's mate: Qxf7#
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    lines = analyse_multipv(engine, board, cfg.engine, cfg.acceptability, nodes=SMOKE_NODES)
    best = lines[0]
    assert best.uci == "h5f7"
    assert best.mate == 1
    assert best.cp == cfg.acceptability.cp_ceiling


def test_multipv_sorted_and_sized(engine, cfg) -> None:
    board = chess.Board()
    lines = analyse_multipv(engine, board, cfg.engine, cfg.acceptability, nodes=SMOKE_NODES)
    assert len(lines) == cfg.engine.multipv
    assert len({line.uci for line in lines}) == len(lines)
    wins = [line.win_pct for line in lines]
    assert wins == sorted(wins, reverse=True)
    assert 45 < wins[0] < 60  # start position is roughly equal


def test_multipv_capped_by_legal_moves(engine, cfg) -> None:
    # Black king in the corner with a single legal move (Kg8)
    board = chess.Board("7k/8/6K1/8/8/8/8/1R6 b - - 0 1")
    assert board.legal_moves.count() < cfg.engine.multipv
    lines = analyse_multipv(engine, board, cfg.engine, cfg.acceptability, nodes=SMOKE_NODES)
    assert len(lines) == board.legal_moves.count()


def test_scores_from_side_to_move(engine, cfg) -> None:
    # Black to move, White is a queen up: Black's win% must be low
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    lines = analyse_multipv(engine, board, cfg.engine, cfg.acceptability, nodes=SMOKE_NODES)
    assert lines[0].win_pct < 10
