"""Engine score -> win probability, mirroring Lichess.

Source (verified 2026-10-01): scalachess ``core/src/main/scala/eval.scala``::

    winningChances(cp) = 2 / (1 + exp(-0.00368208 * cp)) - 1     # cp clipped to +/-1000
    WinPercent         = 50 + 50 * winningChances
    mate in n          -> +/-1000 cp (sign of the mate)

All values are from the side to move's point of view.
"""

from __future__ import annotations

import math

import chess.engine

from zeitnot.config import AcceptabilityConfig

# Any mate score maps beyond the ceiling and is then clipped to exactly +/-ceiling,
# so every mate gets the same win% regardless of distance (as on Lichess).
_MATE_SENTINEL_CP = 1_000_000


def win_pct_from_cp(cp: float, cfg: AcceptabilityConfig) -> float:
    """Win% in [0, 100] for a centipawn score."""
    cp = max(-cfg.cp_ceiling, min(cfg.cp_ceiling, cp))
    return 50 + 50 * (2 / (1 + math.exp(-cfg.win_pct_k * cp)) - 1)


def score_to_cp(score: chess.engine.Score, cfg: AcceptabilityConfig) -> int:
    """Centipawns clipped to +/-ceiling; mates (including ``MateGiven``/mated) become +/-ceiling."""
    cp = score.score(mate_score=_MATE_SENTINEL_CP)
    assert cp is not None  # always set when mate_score is given
    return max(-cfg.cp_ceiling, min(cfg.cp_ceiling, cp))


def win_pct(score: chess.engine.Score, cfg: AcceptabilityConfig) -> float:
    return win_pct_from_cp(score_to_cp(score, cfg), cfg)


def move_loss(best_win_pct: float, played_win_pct: float) -> float:
    """Raw win% loss of the played move vs. the best move.

    Not clamped: a separately searched played move can score slightly above the
    best line (engine noise). Keep the raw value; consumers decide how to treat it.
    """
    return best_win_pct - played_win_pct


def is_acceptable(loss: float, cfg: AcceptabilityConfig) -> bool:
    """Y = 1 iff the win% loss is at most tau."""
    return loss <= cfg.tau_win_pct
