"""Header-level game filter: decide keep/drop from tags without parsing moves."""

from __future__ import annotations

import re
from dataclasses import dataclass

from zeitnot.config import FilterConfig, TcClass, TimeControlConfig
from zeitnot.data.clocks import parse_time_control

_TAG_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$', re.MULTILINE)
_SITE_ID_RE = re.compile(r"/(\w{8})$")


def split_game(game_text: str) -> tuple[dict[str, str], str]:
    """``(headers, movetext)`` of one game's PGN text."""
    head, sep, movetext = game_text.partition("\n\n")
    headers = dict(_TAG_RE.findall(head))
    return headers, movetext.strip() if sep else ""


@dataclass(frozen=True, slots=True)
class KeptGame:
    game_id: str
    utc_date: str  # "YYYY.MM.DD"
    utc_time: str
    event: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    white_title: str | None
    black_title: str | None
    result: str
    termination: str
    eco: str
    opening: str
    base_s: int
    increment_s: int
    tc_class: TcClass
    has_evals: bool
    movetext: str


def _elo(value: str | None) -> int | None:
    return int(value) if value is not None and value.isdigit() else None


def filter_game(
    game_text: str, filt: FilterConfig, tcs: TimeControlConfig
) -> tuple[KeptGame | None, str]:
    """Return ``(game, "kept")`` or ``(None, drop_reason)``. Checks run cheapest first."""
    h, movetext = split_game(game_text)

    result = h.get("Result", "")
    if not movetext or not result or not movetext.endswith(result):
        return None, "incomplete"
    if h.get("Variant", "Standard") != "Standard":
        return None, "variant"
    date = h.get("UTCDate", "")
    if date[:7].replace(".", "-") < filt.min_month:
        return None, "before_min_month"

    tc = parse_time_control(h.get("TimeControl", "-"))
    if tc is None:
        return None, "no_time_control"
    tc_class = tcs.classify(tc.base_s, tc.increment_s)
    if tc_class not in tcs.collected:
        return None, f"class_{tc_class or 'correspondence'}"

    white_elo, black_elo = _elo(h.get("WhiteElo")), _elo(h.get("BlackElo"))
    if white_elo is None or black_elo is None:
        return None, "missing_elo"
    if not all(filt.min_elo <= e <= filt.max_elo for e in (white_elo, black_elo)):
        return None, "elo_out_of_range"

    white_title, black_title = h.get("WhiteTitle"), h.get("BlackTitle")
    if white_title in filt.exclude_titles or black_title in filt.exclude_titles:
        return None, "excluded_title"
    termination = h.get("Termination", "")
    if termination in filt.exclude_terminations:
        return None, "excluded_termination"
    if filt.require_clock and "[%clk " not in movetext:
        return None, "no_clock"

    site_id = _SITE_ID_RE.search(h.get("Site", ""))
    if site_id is None:
        return None, "no_game_id"

    return (
        KeptGame(
            game_id=site_id.group(1),
            utc_date=date,
            utc_time=h.get("UTCTime", ""),
            event=h.get("Event", ""),
            white=h.get("White", ""),
            black=h.get("Black", ""),
            white_elo=white_elo,
            black_elo=black_elo,
            white_title=white_title,
            black_title=black_title,
            result=result,
            termination=termination,
            eco=h.get("ECO", ""),
            opening=h.get("Opening", ""),
            base_s=tc.base_s,
            increment_s=tc.increment_s,
            tc_class=tc_class,
            has_evals="[%eval " in movetext,
            movetext=movetext,
        ),
        "kept",
    )
