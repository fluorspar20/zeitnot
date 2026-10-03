"""Fast tokenizer for Lichess dump movetext (mainline only, no variations).

Lichess dumps contain SAN moves with optional ``?``/``!`` suffixes, move numbers
(``1.`` / ``1...``), one ``{ ... }`` comment after each move holding ``[%clk]``
and optionally ``[%eval]``, and a result token. python-chess's full PGN reader
also handles variations, NAGs and arbitrary tags, which these files never use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\{([^}]*)\}|(\S+)")
_MOVE_NUMBER_RE = re.compile(r"\d+\.(?:\.\.)?")
_CLK_RE = re.compile(r"\[%clk (\d+):(\d\d):(\d\d(?:\.\d+)?)\]")
_EVAL_RE = re.compile(r"\[%eval (#?)(-?\d+(?:\.\d+)?)\]")
RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


@dataclass(slots=True)
class MoveToken:
    san: str
    clock_s: float | None = None
    eval_white_cp: int | None = None  # %eval in centipawns, White's point of view
    eval_white_mate: int | None = None  # %eval "#N": mate in N, White's point of view


def _apply_comment(move: MoveToken, comment: str) -> None:
    if (m := _CLK_RE.search(comment)) is not None:
        move.clock_s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if (m := _EVAL_RE.search(comment)) is not None:
        if m.group(1):
            move.eval_white_mate = int(m.group(2))
        else:
            move.eval_white_cp = round(float(m.group(2)) * 100)


def parse_movetext(movetext: str) -> list[MoveToken]:
    """SAN moves with the clock/eval from the comment that follows each move."""
    moves: list[MoveToken] = []
    for comment, token in _TOKEN_RE.findall(movetext):
        if token:
            if token in RESULTS or _MOVE_NUMBER_RE.fullmatch(token) or token.startswith("$"):
                continue
            moves.append(MoveToken(san=token.rstrip("?!")))
        elif moves:
            _apply_comment(moves[-1], comment)
    return moves
