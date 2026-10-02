import math
from itertools import pairwise

import pytest
from chess.engine import Cp, Mate, MateGiven

from zeitnot.config import REPO_ROOT, AcceptabilityConfig, PipelineConfig
from zeitnot.engine.winpct import (
    is_acceptable,
    move_loss,
    score_to_cp,
    win_pct,
    win_pct_from_cp,
)


@pytest.fixture(scope="module")
def cfg() -> AcceptabilityConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml").acceptability


def test_equal_position_is_fifty(cfg) -> None:
    assert win_pct_from_cp(0, cfg) == 50


def test_known_value_plus_one_pawn(cfg) -> None:
    # 50 + 50 * (2 / (1 + e^-0.368208) - 1), computed by hand
    expected = 50 + 50 * (2 / (1 + math.exp(-0.368208)) - 1)
    assert win_pct_from_cp(100, cfg) == pytest.approx(expected)
    assert win_pct_from_cp(100, cfg) == pytest.approx(59.10, abs=0.01)


@pytest.mark.parametrize("cp", [1, 37, 150, 400, 999])
def test_symmetric(cfg, cp: int) -> None:
    assert win_pct_from_cp(cp, cfg) + win_pct_from_cp(-cp, cfg) == pytest.approx(100)


def test_monotone_and_bounded(cfg) -> None:
    values = [win_pct_from_cp(cp, cfg) for cp in range(-1500, 1501, 25)]
    assert all(a <= b for a, b in pairwise(values))
    assert all(0 <= v <= 100 for v in values)


def test_clipped_at_ceiling(cfg) -> None:
    assert win_pct_from_cp(5000, cfg) == win_pct_from_cp(cfg.cp_ceiling, cfg)
    assert win_pct_from_cp(-5000, cfg) == win_pct_from_cp(-cfg.cp_ceiling, cfg)
    assert win_pct_from_cp(1000, cfg) == pytest.approx(97.55, abs=0.01)


@pytest.mark.parametrize(
    ("score", "expected_cp"),
    [
        (Cp(42), 42),
        (Cp(3000), 1000),
        (Mate(1), 1000),
        (Mate(25), 1000),
        (Mate(-2), -1000),
        (Mate(-0), -1000),  # side to move is checkmated
        (MateGiven, 1000),
    ],
)
def test_score_to_cp(cfg, score, expected_cp: int) -> None:
    assert score_to_cp(score, cfg) == expected_cp


def test_win_pct_mate_equals_ceiling(cfg) -> None:
    assert win_pct(Mate(3), cfg) == win_pct_from_cp(1000, cfg)


def test_acceptability_at_tau(cfg) -> None:
    best = win_pct_from_cp(150, cfg)
    assert is_acceptable(move_loss(best, best), cfg)
    assert is_acceptable(cfg.tau_win_pct, cfg)
    assert not is_acceptable(cfg.tau_win_pct + 1e-9, cfg)


def test_loss_not_clamped() -> None:
    assert move_loss(60.0, 61.5) == pytest.approx(-1.5)
