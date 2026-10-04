from pathlib import Path

import chess
import polars as pl
import pytest

from zeitnot.config import REPO_ROOT, PipelineConfig, Settings
from zeitnot.engine.stockfish import StockfishNotFoundError, find_stockfish, open_engine
from zeitnot.positions.labeling import depth_profile, label_position, positions_to_label, run
from zeitnot.positions.sampling import position_id

SCHOLARS = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"  # Qxf7#
KQ_VS_K = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"  # Black to move, lost


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    """Small node budgets so the engine tests stay fast."""
    full = PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")
    engine = full.engine.model_copy(
        update={"nodes": 60_000, "played_move_nodes": 20_000, "batch_size": 2}
    )
    return full.model_copy(update={"engine": engine})


@pytest.fixture(scope="module")
def engine_path() -> Path:
    try:
        return find_stockfish(Settings(_env_file=None))
    except StockfishNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def engine(cfg: PipelineConfig, engine_path: Path):
    with open_engine(engine_path, cfg.engine) as eng:
        yield eng


def _positions(fens_and_moves: list[tuple[str, str]]) -> pl.DataFrame:
    rows = []
    for fen, uci in fens_and_moves:
        epd = chess.Board(fen).epd(en_passant="legal")
        rows.append({"position_id": position_id(epd), "epd": epd, "fen": fen, "played_uci": uci})
    return pl.DataFrame(rows)


# --- no engine needed --------------------------------------------------------


def test_depth_profile() -> None:
    # The final best move e2e4 first tops at depth 2, loses the lead at 4, holds from 5 on.
    tops = {1: "d2d4", 2: "e2e4", 3: "e2e4", 4: "g1f3", 5: "e2e4", 6: "e2e4"}
    assert depth_profile(tops, "e2e4") == (2, 5)
    assert depth_profile({1: "e2e4", 2: "e2e4"}, "e2e4") == (1, 1)
    assert depth_profile({1: "d2d4"}, "e2e4") == (None, None)
    assert depth_profile({}, "e2e4") == (None, None)


def test_positions_are_grouped_and_evaldb_covered_ones_skipped(
    tmp_path: Path, cfg: PipelineConfig
) -> None:
    start = chess.STARTING_FEN
    positions = _positions(
        [(start, "e2e4"), (start, "d2d4"), (SCHOLARS, "h5f7"), (KQ_VS_K, "e8e7")]
    )
    path = tmp_path / "sampled_positions.parquet"
    positions.write_parquet(path)

    grouped = positions_to_label(path, None, cfg)
    assert grouped.height == 3  # the start position appears once, with both played moves
    start_row = grouped.filter(pl.col("fen") == start).row(0, named=True)
    assert start_row["played_ucis"] == ["d2d4", "e2e4"]

    def pvs(*ucis: str) -> list[dict]:
        return [{"uci": u, "cp": 0, "mate": None, "win_pct": 50.0} for u in ucis]

    epd = {fen: chess.Board(fen).epd(en_passant="legal") for fen in (start, SCHOLARS, KQ_VS_K)}
    evaldb = pl.DataFrame(
        {
            "epd": [epd[start], epd[SCHOLARS], epd[KQ_VS_K]],
            "n_pvs": [5, 5, 2],
            "pvs": [
                pvs("e2e4", "d2d4", "g1f3", "c2c4", "e2e3"),  # 5 lines, both played moves inside
                pvs("h5e5", "c4f7", "h5g5", "h5h4", "h5h3"),  # 5 lines, played h5f7 missing
                pvs("e8e7", "e8f7"),  # played move inside, but too few lines
            ],
        }
    )
    evaldb_path = tmp_path / "evaldb_matches.parquet"
    evaldb.write_parquet(evaldb_path)
    remaining = positions_to_label(path, evaldb_path, cfg)
    assert sorted(remaining["fen"]) == sorted([SCHOLARS, KQ_VS_K])  # only the start is covered


# --- real engine -------------------------------------------------------------


@pytest.mark.engine
def test_mate_in_one_label(engine, cfg: PipelineConfig) -> None:
    label = label_position(engine, SCHOLARS, ["h5f7", "a2a3"], cfg.engine, cfg.acceptability)
    assert label["best_uci"] == "h5f7"
    assert label["lines"][0]["mate"] == 1
    assert label["best_win_pct"] == pytest.approx(97.55, abs=0.01)
    assert len(label["lines"]) == cfg.engine.multipv
    wins = [line["win_pct"] for line in label["lines"]]
    assert wins == sorted(wins, reverse=True)
    assert label["best_move_first_depth"] is not None
    assert label["best_move_first_depth"] <= label["best_move_stable_depth"] <= 3  # obvious mate

    mate, blunder = label["played"]
    assert (mate["uci"], mate["in_multipv"], mate["loss_win_pct"]) == ("h5f7", True, 0)
    assert blunder["uci"] == "a2a3"
    assert blunder["loss_win_pct"] > cfg.acceptability.tau_win_pct  # throws away the mate


@pytest.mark.engine
def test_fewer_legal_moves_than_multipv(engine, cfg: PipelineConfig) -> None:
    board = chess.Board("7k/8/6K1/8/8/8/8/1R6 b - - 0 1")  # only Kg8
    label = label_position(engine, board.fen(), ["h8g8"], cfg.engine, cfg.acceptability)
    assert [line["uci"] for line in label["lines"]] == ["h8g8"]
    assert label["played"][0]["in_multipv"] and label["played"][0]["loss_win_pct"] == 0


@pytest.mark.engine
def test_labels_are_reproducible(engine, cfg: PipelineConfig) -> None:
    fen = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 4 8"
    first = label_position(engine, fen, ["e1g1"], cfg.engine, cfg.acceptability)
    label_position(engine, SCHOLARS, [], cfg.engine, cfg.acceptability)  # disturb the hash
    second = label_position(engine, fen, ["e1g1"], cfg.engine, cfg.acceptability)
    for key in ("lines", "depth", "nodes", "best_move_first_depth", "played"):
        assert first[key] == second[key], key


@pytest.mark.engine
def test_run_writes_parts_and_resumes(
    tmp_path: Path, cfg: PipelineConfig, engine_path: Path
) -> None:
    positions = _positions(
        [(chess.STARTING_FEN, "e2e4"), (chess.STARTING_FEN, "a2a4"), (SCHOLARS, "h5f7"),
         (KQ_VS_K, "e8e7"), ("7k/8/6K1/8/8/8/8/1R6 b - - 0 1", "h8g8")]
    )  # fmt: skip
    path = tmp_path / "sampled_positions.parquet"
    positions.write_parquet(path)
    out = tmp_path / "labels"

    report = run(path, out, cfg, engine_path, workers=2)
    assert (report["positions"], report["parts"]) == (4, 2)  # batch_size 2
    labels = pl.read_parquet(out / "part-*.parquet")
    assert sorted(labels["position_id"]) == sorted(positions["position_id"].unique())
    start = labels.filter(pl.col("epd").str.starts_with("rnbqkbnr/pppppppp/8/8/8/8")).row(
        0, named=True
    )
    assert [p["uci"] for p in start["played"]] == ["a2a4", "e2e4"]
    assert 40 < start["best_win_pct"] < 60

    again = run(path, out, cfg, engine_path, workers=2)
    assert again["last_run"]["positions_labeled"] == 0
    assert again["last_run"]["batches_skipped"] == 2

    deeper = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"nodes": 70_000})})
    with pytest.raises(SystemExit):
        run(path, out, deeper, engine_path, workers=2)
