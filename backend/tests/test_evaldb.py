import json
from pathlib import Path

import chess
import polars as pl
import pytest
import zstandard

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.data.stream import iter_line_chunks
from zeitnot.positions.evaldb import parse_record, run


def epd_after(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.epd(en_passant="legal")


START = epd_after()
AFTER_E4 = epd_after("e4")  # Black to move
ITALIAN = epd_after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")  # White can castle short
FOOLS = epd_after("f3", "e5", "g4")  # Black mates with Qh4
SHALLOW = epd_after("d4")
NOT_IN_DB = epd_after("c4")
UNWANTED = epd_after("Nf3")


def record(epd: str, *evals: dict) -> bytes:
    return json.dumps({"fen": epd, "evals": list(evals)}, separators=(",", ":")).encode()


def ev(knodes: int, depth: int, *pvs: dict) -> dict:
    return {"pvs": list(pvs), "knodes": knodes, "depth": depth}


RECORDS = [
    # A deep single-line eval and a shallower three-line eval.
    record(
        START,
        ev(50_000, 40, {"cp": 30, "line": "e2e4 e7e5"}),
        ev(
            20_000,
            30,
            {"cp": 25, "line": "d2d4 d7d5"},
            {"cp": 28, "line": "e2e4 e7e5"},
            {"cp": 20, "line": "g1f3 d7d5"},
        ),
    ),
    # Black to move: scores are White's, so Black's best line has the LOWEST cp.
    record(
        AFTER_E4, ev(3_000, 25, {"cp": 20, "line": "e7e5 g1f3"}, {"cp": 35, "line": "c7c5 g1f3"})
    ),
    # Castling in UCI_Chess960 notation: king takes own rook.
    record(ITALIAN, ev(8_000, 28, {"cp": 15, "line": "e1h1 g8f6"})),
    # White gets mated in 1 (White's point of view: mate -1).
    record(FOOLS, ev(2_000, 99, {"mate": -1, "line": "d8h4"})),
    # Only a very shallow eval: not usable.
    record(SHALLOW, ev(5, 8, {"cp": 0, "line": "d7d5"})),
    record(UNWANTED, ev(9_000, 30, {"cp": 10, "line": "d7d5"})),
]


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


def test_best_and_multipv_evals_are_chosen_separately(cfg: PipelineConfig) -> None:
    row = parse_record(RECORDS[0], cfg)
    assert row is not None
    assert (row["best_uci"], row["best_cp"], row["best_knodes"]) == ("e2e4", 30, 50_000)
    assert (row["n_evals"], row["n_pvs"], row["multi_knodes"]) == (2, 3, 20_000)
    assert [p["uci"] for p in row["pvs"]] == ["e2e4", "d2d4", "g1f3"]  # best first
    assert [p["cp"] for p in row["pvs"]] == [28, 25, 20]


def test_black_to_move_flips_the_sign(cfg: PipelineConfig) -> None:
    row = parse_record(RECORDS[1], cfg)
    assert row is not None
    assert (row["best_uci"], row["best_cp"]) == ("e7e5", -20)
    assert [(p["uci"], p["cp"]) for p in row["pvs"]] == [("e7e5", -20), ("c7c5", -35)]
    assert row["best_win_pct"] < 50


def test_chess960_castling_becomes_standard_uci(cfg: PipelineConfig) -> None:
    row = parse_record(RECORDS[2], cfg)
    assert row is not None
    assert row["best_uci"] == "e1g1"
    assert row["pvs"][0]["uci"] == "e1g1"


def test_mate_score_for_side_to_move(cfg: PipelineConfig) -> None:
    row = parse_record(RECORDS[3], cfg)
    assert row is not None
    assert (row["best_uci"], row["best_cp"], row["best_mate"]) == ("d8h4", None, 1)
    assert row["best_win_pct"] == pytest.approx(97.55, abs=0.01)  # mate maps to the cp ceiling


def test_shallow_eval_is_not_usable(cfg: PipelineConfig) -> None:
    assert parse_record(RECORDS[4], cfg) is None


def test_line_chunks(tmp_path: Path) -> None:
    text = b"\n".join(RECORDS) + b"\n"
    path = tmp_path / "db.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(text + b'{"fen":"cut off'))
    chunks = [data for _, data in iter_line_chunks(path, 300)]
    assert len(chunks) > 2
    assert all(c.endswith(b"\n") for c in chunks)
    assert b"".join(chunks) == text  # the partial last line is dropped


def test_run_and_coverage(tmp_path: Path, cfg: PipelineConfig) -> None:
    db = tmp_path / "db.jsonl.zst"
    db.write_bytes(zstandard.ZstdCompressor().compress(b"\n".join(RECORDS) + b"\n"))
    positions = pl.DataFrame(
        {
            "epd": [START, START, AFTER_E4, ITALIAN, FOOLS, SHALLOW, NOT_IN_DB],
            # two observations of the start position: one played move is in the lines, one not
            "played_uci": ["d2d4", "a2a3", "c7c5", "e1g1", "d8h4", "d7d5", "e7e5"],
            "n_legal_moves": [20, 20, 20, 33, 30, 20, 20],
            "phase": ["opening"] * 7,
            "ply_bucket": [10] * 7,
            "tc_class": ["blitz"] * 6 + ["rapid"],
        }
    )
    positions_path = tmp_path / "sampled_positions.parquet"
    positions.write_parquet(positions_path)

    report = run(db, positions_path, tmp_path, cfg, workers=1, chunk_mb=1, max_in_flight=2)
    matches = pl.read_parquet(tmp_path / "evaldb_matches.parquet")

    assert report["lines_scanned"] == len(RECORDS)
    assert report["raw_hits"] == 5  # all wanted keys present in the file, including SHALLOW
    assert sorted(matches["epd"]) == sorted([START, AFTER_E4, ITALIAN, FOOLS])  # SHALLOW unusable
    assert report["observations"] == 7 and report["distinct_positions"] == 6
    assert report["share_covered"] == pytest.approx(5 / 7, abs=1e-4)
    # No entry has the 5 lines the engine config asks for (most: 3).
    assert report["share_covered_multipv"] == 0
    assert report["still_need_engine"] == 7
    # Played move among stored lines: d2d4, c7c5, e1g1, d8h4 (not a2a3, SHALLOW, NOT_IN_DB).
    assert report["share_played_move_in_pvs"] == pytest.approx(4 / 7, abs=1e-4)
    assert (tmp_path / "_evaldb_report.md").exists()


def test_extra_epd_files_are_looked_up_in_the_same_pass(
    tmp_path: Path, cfg: PipelineConfig
) -> None:
    db = tmp_path / "db.jsonl.zst"
    db.write_bytes(zstandard.ZstdCompressor().compress(b"\n".join(RECORDS) + b"\n"))
    main_path, extra_path = tmp_path / "sampled_positions.parquet", tmp_path / "puzzles.parquet"
    pl.DataFrame(
        {"epd": [START], "played_uci": ["e2e4"], "n_legal_moves": [20], "phase": ["opening"],
         "ply_bucket": [10], "tc_class": ["blitz"]}
    ).write_parquet(main_path)  # fmt: skip
    pl.DataFrame({"epd": [UNWANTED, ITALIAN]}).write_parquet(extra_path)

    report = run(
        db, main_path, tmp_path, cfg, workers=1, chunk_mb=1, max_in_flight=2,
        extra_epd_paths=(extra_path,),
    )  # fmt: skip
    matches = pl.read_parquet(tmp_path / "evaldb_matches.parquet")
    assert sorted(matches["epd"]) == sorted([START, UNWANTED, ITALIAN])
    assert report["matched_positions_all_files"] == 3
    # The coverage numbers describe the main file only.
    assert (report["observations"], report["matched_positions"]) == (1, 1)
    assert report["share_covered"] == 1.0
