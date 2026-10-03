from pathlib import Path

import chess.pgn
import pytest

from zeitnot.config import REPO_ROOT, ClockConfig, PipelineConfig
from zeitnot.data.clocks import (
    MoveTiming,
    TimeControl,
    compute_move_timings,
    parse_time_control,
    read_clocks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "clocks.pgn"


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def games() -> dict[str, chess.pgn.Game]:
    out = {}
    with FIXTURE.open(encoding="utf-8") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            out[game.headers["Case"]] = game
    return out


def timings(game: chess.pgn.Game, cfg: PipelineConfig) -> list[MoveTiming]:
    tc = parse_time_control(game.headers["TimeControl"])
    assert tc is not None
    clocks = read_clocks(game)
    assert clocks is not None
    tc_class = cfg.time_controls.classify(tc.base_s, tc.increment_s)
    return compute_move_timings(clocks, tc, tc_class, cfg.clock)


def col(rows: list[MoveTiming], name: str) -> list:
    return [getattr(r, name) for r in rows]


# --- time control tag ---------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("180+2", TimeControl(180, 2)),
        ("0+1", TimeControl(0, 1)),
        (" 600+0 ", TimeControl(600, 0)),
        ("-", None),
        ("180", None),
        ("", None),
    ],
)
def test_parse_time_control(tag: str, expected: TimeControl | None) -> None:
    assert parse_time_control(tag) == expected


# --- fixture games (expected values computed by hand in the comments) --------


def test_increment_game(games, cfg) -> None:
    # 180+2. White clocks 180, 177, 178, 160; Black clocks 180, 179, 170.
    rows = timings(games["increment"], cfg)
    assert col(rows, "ply") == [1, 2, 3, 4, 5, 6, 7]
    assert col(rows, "side") == ["w", "b", "w", "b", "w", "b", "w"]
    assert col(rows, "is_first_move") == [True, True, False, False, False, False, False]
    assert col(rows, "own_clk_before_s") == [180, 180, 180, 180, 177, 179, 178]
    assert col(rows, "opp_clk_s") == [180, 180, 180, 177, 179, 178, 170]
    # T = before - after + 2:  180-177+2, 180-179+2, 177-178+2, 179-170+2, 178-160+2
    assert col(rows, "time_spent_s") == [None, None, 5, 3, 1, 11, 20]
    assert not any(col(rows, "berserk"))
    assert not any(col(rows, "premove_suspect"))
    assert not any(col(rows, "time_added"))


def test_berserk_removes_increment(games, cfg) -> None:
    # 180+2 arena; White berserks (starts at 90 s, no increment), Black does not.
    # White 90, 86, 80; Black 180, 178.
    rows = timings(games["berserk"], cfg)
    assert col(rows, "berserk") == [True, False, True, False, True]
    assert col(rows, "own_clk_before_s") == [90, 180, 90, 180, 86]
    assert col(rows, "opp_clk_s") == [180, 90, 180, 86, 178]
    # White: 90-86+0, 86-80+0; Black: 180-178+2
    assert col(rows, "time_spent_s") == [None, None, 4, 4, 6]


def test_time_added_is_flagged_not_timed(games, cfg) -> None:
    # 300+0. Black's clock jumps 300 -> 310 (opponent gave +15 s, Black used ~5 s).
    rows = timings(games["time_added"], cfg)
    black_second = rows[3]
    assert black_second.time_added
    assert black_second.time_spent_s is None
    assert not black_second.premove_suspect
    # Later moves are timed from the new clock: White 295-290, Black 310-302.
    assert col(rows, "time_spent_s")[4:] == [5, 8]
    assert rows[5].own_clk_before_s == 310


def test_small_negative_is_noise(games, cfg) -> None:
    # 120+1. White 120 -> 122: T = 120-122+1 = -1, within noise -> clipped to 0.
    rows = timings(games["noise"], cfg)
    assert rows[2].time_spent_s == 0
    assert rows[2].premove_suspect
    assert not rows[2].time_added
    assert rows[3].time_spent_s == 3  # Black: 120-118+1


def test_low_clock_and_premove(games, cfg) -> None:
    # 180+0 blitz, low_clock threshold 10 s (strict <).
    # White 180, 12, 10, 9, 9; Black 180, 120, 119, 110.
    rows = timings(games["low_clock"], cfg)
    assert cfg.clock.low_clock_s["blitz"] == 10
    white = rows[0::2]
    assert col(white, "own_clk_before_s") == [180, 180, 12, 10, 9]
    assert col(white, "low_clock") == [False, False, False, False, True]  # 10 is not < 10
    assert col(white, "time_spent_s") == [None, 168, 2, 1, 0]
    assert col(white, "premove_suspect") == [False, False, False, False, True]
    black = rows[1::2]
    assert col(black, "time_spent_s") == [None, 60, 1, 9]
    assert not any(col(black, "low_clock"))


def test_single_move_game(games, cfg) -> None:
    rows = timings(games["single_move"], cfg)
    assert len(rows) == 1
    assert rows[0].is_first_move and rows[0].time_spent_s is None
    assert rows[0].opp_clk_s == 180  # Black never moved: starting clock = base


def test_missing_clock_returns_none(games) -> None:
    assert read_clocks(games["missing_clock"]) is None


# --- direct edge cases ---------------------------------------------------------


def test_empty_game() -> None:
    clock_cfg = ClockConfig(
        premove_max_s=0, noise_tolerance_s=1, berserk_tolerance_s=1, low_clock_s={}
    )
    assert compute_move_timings([], TimeControl(180, 0), "blitz", clock_cfg) == []


def test_unknown_class_never_low_clock(cfg) -> None:
    rows = compute_move_timings([60, 60, 1, 1, 0.0], TimeControl(60, 0), None, cfg.clock)
    assert not any(col(rows, "low_clock"))


def test_zero_base_is_never_berserk(cfg) -> None:
    # 0+1: both start at 0 s, which equals base/2 but must not count as berserk.
    rows = compute_move_timings([0, 0, 1, 1], TimeControl(0, 1), "bullet", cfg.clock)
    assert not any(col(rows, "berserk"))
    assert col(rows, "time_spent_s") == [None, None, 0, 0]
