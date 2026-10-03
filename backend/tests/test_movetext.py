import pytest

from zeitnot.data.movetext import parse_movetext


def test_lichess_movetext_with_evals_and_suffixes() -> None:
    text = (
        "1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... c5 { [%clk 0:00:30] } "
        "2. Nf3?! { [%eval -1.35] [%clk 0:00:29] } 2... d6?? { [%eval #-4] [%clk 0:00:28] } 1-0"
    )
    moves = parse_movetext(text)
    assert [m.san for m in moves] == ["e4", "c5", "Nf3", "d6"]
    assert [m.clock_s for m in moves] == [30, 30, 29, 28]
    assert [m.eval_white_cp for m in moves] == [17, None, -135, None]
    assert [m.eval_white_mate for m in moves] == [None, None, None, -4]


def test_castling_promotion_and_checks() -> None:
    text = (
        "1. O-O { [%clk 0:01:00] } 1... O-O-O+ { [%clk 0:01:00] } 2. exd8=Q# { [%clk 0:00:59] } 1-0"
    )
    assert [m.san for m in parse_movetext(text)] == ["O-O", "O-O-O+", "exd8=Q#"]


@pytest.mark.parametrize(
    ("clk", "seconds"),
    [("0:00:00", 0), ("0:02:59.9", 179.9), ("1:30:00", 5400), ("0:10:05", 605)],
)
def test_clock_formats(clk: str, seconds: float) -> None:
    (move,) = parse_movetext(f"1. e4 {{ [%clk {clk}] }} *")
    assert move.clock_s == pytest.approx(seconds)


@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2", "*"])
def test_results_and_bare_moves(result: str) -> None:
    moves = parse_movetext(f"1. d4 d5 2. c4 {result}")
    assert [m.san for m in moves] == ["d4", "d5", "c4"]
    assert all(m.clock_s is None for m in moves)


def test_empty_movetext() -> None:
    assert parse_movetext("1-0") == []
