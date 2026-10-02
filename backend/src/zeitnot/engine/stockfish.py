"""Stockfish process management and multi-PV analysis via python-chess."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from zeitnot.config import AcceptabilityConfig, EngineConfig, Settings
from zeitnot.engine.winpct import score_to_cp, win_pct_from_cp


class StockfishNotFoundError(FileNotFoundError):
    pass


def find_stockfish(settings: Settings) -> Path:
    """``ZEITNOT_STOCKFISH_PATH`` if set, else the first ``stockfish*.exe`` under ``engines/``."""
    if settings.stockfish_path is not None:
        if not settings.stockfish_path.is_file():
            raise StockfishNotFoundError(f"ZEITNOT_STOCKFISH_PATH not found: {settings.stockfish_path}")
        return settings.stockfish_path
    candidates = sorted((settings.root_dir / "engines").glob("**/stockfish*.exe"))
    if not candidates:
        raise StockfishNotFoundError(
            "No Stockfish binary: set ZEITNOT_STOCKFISH_PATH or unzip the official "
            "Windows build into engines/"
        )
    return candidates[0]


@contextmanager
def open_engine(path: Path, cfg: EngineConfig) -> Iterator[chess.engine.SimpleEngine]:
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    try:
        engine.configure({"Threads": cfg.threads_per_worker, "Hash": cfg.hash_mb_per_worker})
        yield engine
    finally:
        engine.quit()


@dataclass(frozen=True, slots=True)
class PvLine:
    """One multi-PV line, scores from the side to move's point of view."""

    uci: str
    cp: int  # clipped to +/-cp_ceiling; mates map to the ceiling
    mate: int | None  # moves to mate (negative = getting mated), None if no mate
    win_pct: float
    depth: int


def analyse_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    engine_cfg: EngineConfig,
    acc_cfg: AcceptabilityConfig,
    nodes: int | None = None,
) -> list[PvLine]:
    """Multi-PV search with a fixed node budget; lines sorted best first."""
    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes or engine_cfg.nodes),
        multipv=min(engine_cfg.multipv, board.legal_moves.count()),
    )
    lines = []
    for info in infos:
        score = info["score"].pov(board.turn)
        cp = score_to_cp(score, acc_cfg)
        lines.append(
            PvLine(
                uci=info["pv"][0].uci(),
                cp=cp,
                mate=score.mate(),
                win_pct=win_pct_from_cp(cp, acc_cfg),
                depth=info.get("depth", 0),
            )
        )
    return lines
