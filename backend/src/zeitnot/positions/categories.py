"""Rule-based position categories (v1) from engine labels and material.

One label per position, checked in this order:

- ``forced``           one legal move: nothing to decide.
- ``recapture``        the single acceptable move takes back on the square where the
                       opponent just captured: nothing to find.
- ``tactical``         few acceptable moves, and the best one leaves the mover clearly better.
- ``defensive``        few acceptable moves, and even the best one leaves the mover worse.
- ``only-move``        exactly one acceptable move in a roughly level position.
- ``endgame``          few pieces left, and none of the sharp cases above.
- ``many-good-moves``  most of the engine's top lines are acceptable.
- ``quiet``            everything else.

``forced`` and ``recapture`` are kept as data but are not training positions.

The rules use only the engine's numbers (how many lines are acceptable, the win% after
the best move), not what the move does on the board. "tactical" therefore means "one or
two moves win and the alternatives do not": usually a tactic, sometimes a precise quiet
move; and a tactic with three or more winning moves, or one that only equalizes, gets
another label.
"""

from __future__ import annotations

import polars as pl

from zeitnot.config import CategoriesConfig

CATEGORIES = [
    "forced",
    "recapture",
    "tactical",
    "defensive",
    "only-move",
    "endgame",
    "many-good-moves",
    "quiet",
]
TRAINABLE = [c for c in CATEGORIES if c not in ("forced", "recapture")]


def best_is_recapture() -> pl.Expr:
    """The engine's best move lands on the square where the opponent just captured."""
    return (
        pl.col("opp_prev_was_capture")
        & (pl.col("best_uci").str.slice(2, 2) == pl.col("opp_prev_uci").str.slice(2, 2))
    ).fill_null(False)


def category_expr(cfg: CategoriesConfig) -> pl.Expr:
    acceptable = pl.col("n_acceptable_lines")
    legal = pl.col("n_legal_moves")
    win = pl.col("best_win_pct")
    # "Few good moves" only means something if worse moves exist as well.
    sharp = (acceptable <= cfg.sharp_max_acceptable) & (legal > acceptable)
    return (
        pl.when(legal == 1)
        .then(pl.lit("forced"))
        .when((acceptable == 1) & best_is_recapture())
        .then(pl.lit("recapture"))
        .when(sharp & (win >= cfg.tactical_min_win_pct))
        .then(pl.lit("tactical"))
        .when(sharp & (win <= cfg.defensive_max_win_pct))
        .then(pl.lit("defensive"))
        .when(acceptable == 1)
        .then(pl.lit("only-move"))
        .when(pl.col("phase") == "endgame")
        .then(pl.lit("endgame"))
        .when(acceptable >= cfg.many_min_acceptable)
        .then(pl.lit("many-good-moves"))
        .otherwise(pl.lit("quiet"))
        .cast(pl.Enum(CATEGORIES))
    )
