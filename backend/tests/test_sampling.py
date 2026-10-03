from pathlib import Path

import chess
import polars as pl
import pytest

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.data.clocks import TimeControl
from zeitnot.data.parse_games import MOVES_SCHEMA, parse_game_moves
from zeitnot.positions.sampling import (
    allocate_quotas,
    game_candidates,
    position_id,
    position_phase,
    run,
)

# Ruy Lopez, Breyer variation: 40 legal plies.
LINE = [
    *("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"),
    *("Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7"),
    *("Nbd2", "Bb7", "Bc2", "Re8", "Nf1", "Bf8", "Ng3", "g6", "a4", "c5"),
    *("d5", "c4", "Bg5", "h6", "Be3", "Nc5", "Qd2", "h5", "Bg5", "Be7"),
]
N_GAMES = 40
BERSERK_GAME = "game0001"  # White berserks
PREMOVE_GAME = "game0000"  # every Black move from ply 12 on is instant


def _clk(seconds: int) -> str:
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _movetext(game_index: int, game_id: str) -> str:
    clock = [150 if game_id == BERSERK_GAME else 300, 300]
    tokens = []
    for i, san in enumerate(LINE):
        side = i % 2
        if i >= 2:  # first moves are free
            instant = game_id == PREMOVE_GAME and side == 1 and i + 1 >= 12
            clock[side] -= 0 if instant else 1 + (game_index + i) % 4
        tokens.append(f"{san} {{ [%clk {_clk(clock[side])}] }}")
    return " ".join(tokens) + " *"


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def interim(tmp_path_factory: pytest.TempPathFactory, cfg: PipelineConfig) -> Path:
    """Synthetic games/moves tables: 40 blitz games (300+0), ratings 800..2775."""
    root = tmp_path_factory.mktemp("interim")
    games, cols = [], {name: [] for name in MOVES_SCHEMA}
    for g in range(N_GAMES):
        game_id = f"game{g:04d}"
        games.append(
            {"game_id": game_id, "tc_class": "blitz", "base_s": 300, "increment_s": 0,
             "white_elo": 800 + 50 * g, "black_elo": 825 + 50 * g}
        )  # fmt: skip
        game_cols = parse_game_moves(
            game_id, _movetext(g, game_id), TimeControl(300, 0), "blitz", cfg.clock
        )
        for name, values in game_cols.items():
            cols[name].extend(values)
    for table, df in [
        ("games", pl.DataFrame(games)),
        ("moves", pl.DataFrame(cols, schema=MOVES_SCHEMA)),
    ]:
        part = root / table / "month=2026-08" / "part-000000.parquet"
        part.parent.mkdir(parents=True)
        df.write_parquet(part)
    return root


# --- quota allocation --------------------------------------------------------


def test_small_stratum_taken_whole_rest_split_evenly() -> None:
    # 60 - 5 = 55 for two strata: 27 each, the 1 left over goes to the first key.
    assert allocate_quotas({"a": 5, "b": 100, "c": 100}, 60) == {"a": 5, "b": 28, "c": 27}


def test_equal_strata_and_remainder() -> None:
    assert allocate_quotas({"a": 10, "b": 10, "c": 10}, 10) == {"a": 4, "b": 3, "c": 3}


def test_target_above_population_takes_everything() -> None:
    sizes = {"a": 3, "b": 7}
    assert allocate_quotas(sizes, 1000) == sizes


@pytest.mark.parametrize("target", [0, 1, 17, 64, 211, 500])
def test_quotas_sum_to_target_and_respect_sizes(target: int) -> None:
    sizes = {i: n for i, n in enumerate([1, 2, 2, 9, 30, 30, 77, 60])}
    quotas = allocate_quotas(sizes, target)
    assert sum(quotas.values()) == min(target, sum(sizes.values()))
    assert all(0 <= quotas[k] <= sizes[k] for k in sizes)


# --- end-to-end sampling -----------------------------------------------------


def _sample(
    interim: Path, out: Path, cfg: PipelineConfig, target: int
) -> tuple[dict, pl.DataFrame]:
    report = run(interim, ["2026-08"], out, cfg, target)
    return report, pl.read_parquet(out / "sampled_positions.parquet")


def test_eligibility_and_per_game_cap(interim: Path, tmp_path: Path, cfg: PipelineConfig) -> None:
    report, df = _sample(interim, tmp_path, cfg, target=60)
    assert df.height == report["sampled"] == 60
    assert df["ply"].min() >= cfg.sampling.min_ply
    assert df.group_by("game_id").len()["len"].max() <= cfg.sampling.positions_per_game
    assert (df["time_spent_s"] > cfg.clock.premove_max_s).all()  # no premove-suspect rows
    assert df.filter((pl.col("game_id") == BERSERK_GAME) & (pl.col("side") == "w")).is_empty()
    # The instant Black moves of the premove game are never sampled.
    late_black = (
        (pl.col("game_id") == PREMOVE_GAME) & (pl.col("side") == "b") & (pl.col("ply") >= 12)
    )
    assert df.filter(late_black).is_empty()


def test_same_seed_same_sample_different_seed_differs(
    interim: Path, tmp_path: Path, cfg: PipelineConfig
) -> None:
    _, a = _sample(interim, tmp_path / "a", cfg, target=60)
    _, b = _sample(interim, tmp_path / "b", cfg, target=60)
    assert a.equals(b)
    other = cfg.model_copy(update={"seed": cfg.seed + 1})
    _, c = _sample(interim, tmp_path / "c", other, target=60)
    assert not a.select("game_id", "ply").equals(c.select("game_id", "ply"))


def test_fen_is_the_position_before_the_played_move(
    interim: Path, tmp_path: Path, cfg: PipelineConfig
) -> None:
    _, df = _sample(interim, tmp_path, cfg, target=60)
    for row in df.iter_rows(named=True):
        board = chess.Board()
        for san in LINE[: row["ply"] - 1]:
            board.push_san(san)
        assert row["fen"] == board.fen()
        assert row["epd"] == board.epd(en_passant="legal")
        assert row["position_id"] == position_id(row["epd"])
        assert row["played_uci"] == board.parse_san(LINE[row["ply"] - 1]).uci()
        assert row["side"] == ("w" if board.turn else "b")
        assert row["move_number"] == board.fullmove_number
        expected_elo = 800 + 50 * int(row["game_id"][4:]) + (0 if board.turn else 25)
        assert row["mover_elo"] == expected_elo


def test_weights(interim: Path, tmp_path: Path, cfg: PipelineConfig) -> None:
    # Taking every candidate: weights reduce to plies-per-game ratios and sum exactly
    # to the number of eligible rows (each game's k picks stand for all its eligible plies).
    report, df = _sample(interim, tmp_path, cfg, target=10_000)
    assert report["sampled"] == report["candidates_after_per_game_cap"]
    assert (df["stratum_sampled"] == df["stratum_candidates"]).all()
    assert df["sampling_weight"].sum() == pytest.approx(report["eligible_rows"])
    # A smaller sample: weight = (eligible / sampled in game) x (candidates / sampled in stratum).
    _, small = _sample(interim, tmp_path / "small", cfg, target=30)
    expected = (
        small["game_eligible_plies"] / small["game_sampled_plies"]
        * small["stratum_candidates"] / small["stratum_sampled"]
    )  # fmt: skip
    assert (small["sampling_weight"] - expected).abs().max() < 1e-9
    assert small.group_by("rating_band").len()["len"].max() <= 3  # spread across bands


def test_phase_rule(cfg: PipelineConfig) -> None:
    phase = cfg.sampling.phase
    assert position_phase(chess.Board(), 10, phase) == "opening"
    assert position_phase(chess.Board(), 40, phase) == "middlegame"  # full board, but late
    rook_ending = chess.Board("8/5pk1/6p1/8/8/6P1/r4PK1/1R6 w - - 0 40")
    assert position_phase(rook_ending, 79, phase) == "endgame"


def test_stage_two_randomness_is_independent_of_stage_one(
    interim: Path, cfg: PipelineConfig
) -> None:
    # Stage 1 keeps each game's k smallest draws (mean ~ 2/32 here). If those draws were
    # reused for the stratum stage, rows from long games would be favored and the weights
    # would overestimate the population. The survivors must carry fresh uniform draws.
    month_dir = "month=2026-08"
    picked, n_eligible = game_candidates(
        interim / "moves" / month_dir / "part-000000.parquet",
        interim / "games" / month_dir / "part-000000.parquet",
        "2026-08", 0, cfg,
    )  # fmt: skip
    assert picked.height < n_eligible
    assert 0.4 < picked["u"].mean() < 0.6
