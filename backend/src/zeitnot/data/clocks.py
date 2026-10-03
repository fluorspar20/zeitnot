"""Per-move clock state and thinking time from PGN ``%clk`` annotations.

Lichess conventions (checked against real dumps and the Lichess source):

- ``%clk`` is the mover's remaining clock *after* the move, increment included,
  rounded to the nearest second.
- Neither side's clock runs before their first move, so both first moves carry
  no thinking-time information.
- Berserk (arena tournaments) halves a player's starting clock and removes
  their increment. It is not in the headers; it shows as a first clock of base/2.
- In casual and rated non-tournament games a player can give the opponent extra
  time, which makes the opponent's clock jump up between two of their moves.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import chess.pgn

from zeitnot.config import ClockConfig, TcClass

_TC_RE = re.compile(r"(\d+)\+(\d+)")


@dataclass(frozen=True, slots=True)
class TimeControl:
    base_s: int
    increment_s: int


def parse_time_control(tag: str) -> TimeControl | None:
    """``"180+2"`` -> TimeControl(180, 2); ``"-"`` (correspondence) or malformed -> None."""
    m = _TC_RE.fullmatch(tag.strip())
    return TimeControl(int(m.group(1)), int(m.group(2))) if m else None


def read_clocks(game: chess.pgn.Game) -> list[float] | None:
    """``%clk`` value of every mainline move, or None if any move lacks one."""
    clocks = [node.clock() for node in game.mainline()]
    if any(c is None for c in clocks):
        return None
    return clocks  # type: ignore[return-value]  # no None left


@dataclass(frozen=True, slots=True)
class MoveTiming:
    ply: int  # 1-based
    side: Literal["w", "b"]
    clk_after_s: float  # mover's clock after the move (increment included)
    own_clk_before_s: float  # mover's clock when the position appeared
    opp_clk_s: float  # opponent's clock while the mover was thinking
    time_spent_s: float | None  # None for each side's first move and for time_added moves
    is_first_move: bool
    berserk: bool
    premove_suspect: bool
    low_clock: bool
    time_added: bool  # clock jumped up by more than noise: opponent gave extra time


def _is_berserk(start_s: float, base_s: int, cfg: ClockConfig) -> bool:
    tol = cfg.berserk_tolerance_s
    return base_s > 2 * tol and abs(start_s - base_s / 2) <= tol


def compute_move_timings(
    clocks: Sequence[float],
    tc: TimeControl,
    tc_class: TcClass | None,
    cfg: ClockConfig,
) -> list[MoveTiming]:
    """Clock state, thinking time and flags for each ply of a game starting from move 1.

    ``clocks[i]`` is the ``%clk`` after ply i+1 (even index = White).
    Thinking time T = own clock before - own clock after + increment.
    """
    if not clocks:
        return []
    # The first move consumes no time, so each side's first reading is its starting clock.
    start = [clocks[0], clocks[1] if len(clocks) > 1 else float(tc.base_s)]
    berserk = [_is_berserk(s, tc.base_s, cfg) for s in start]
    increment = [0 if b else tc.increment_s for b in berserk]
    low_clock_s = cfg.low_clock_s.get(tc_class) if tc_class is not None else None

    last = list(start)  # latest clock reading per side (0 = White, 1 = Black)
    rows: list[MoveTiming] = []
    for i, clk_after in enumerate(clocks):
        side = i % 2
        is_first = i < 2
        before = start[side] if is_first else last[side]
        opp = last[1 - side]

        time_spent: float | None = None
        time_added = False
        if not is_first:
            raw = before - clk_after + increment[side]
            if raw < -cfg.noise_tolerance_s:
                time_added = True
            else:
                time_spent = max(raw, 0.0)

        rows.append(
            MoveTiming(
                ply=i + 1,
                side="w" if side == 0 else "b",
                clk_after_s=clk_after,
                own_clk_before_s=before,
                opp_clk_s=opp,
                time_spent_s=time_spent,
                is_first_move=is_first,
                berserk=berserk[side],
                premove_suspect=time_spent is not None and time_spent <= cfg.premove_max_s,
                low_clock=low_clock_s is not None and before < low_clock_s,
                time_added=time_added,
            )
        )
        last[side] = clk_after
    return rows
