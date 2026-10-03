"""Lichess puzzle DB: ingest to Parquet and join puzzles to their source games.

Puzzle CSV conventions (checked against the full file):
- ``FEN`` is the position *before* the opponent's move ``Moves[0]``; the puzzle
  position is after it, and the solution starts at ``Moves[1]``. Moves are
  standard UCI (castling as e1g1).
- ``GameUrl`` = ``https://lichess.org/<game id>[/black]#<N>``: ``N`` is the ply of
  ``Moves[0]`` in the source game, so the real player's answer is ply ``N + 1``.
  ``/black`` means the side playing ``Moves[0]`` is Black, i.e. the solver is White.
- Every solution move is an only move, except mate-in-one where any mate counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import chess
import polars as pl
import zstandard

from zeitnot.config import PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic

log = logging.getLogger(__name__)

PUZZLES_FILE = "puzzles.parquet"


def _decompress(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    with src.open("rb") as fin, tmp.open("wb") as fout:
        zstandard.ZstdDecompressor().copy_stream(fin, fout)
    tmp.replace(dst)


def ingest(puzzles_zst: Path, out_dir: Path) -> Path:
    """Puzzle CSV (.zst) -> ``<out_dir>/puzzles.parquet`` with parsed URL fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lichess_db_puzzle.csv"
    out = out_dir / PUZZLES_FILE
    log.info("decompressing %s", puzzles_zst.name)
    _decompress(puzzles_zst, csv_path)
    moves = pl.col("Moves").str.split(" ")
    url = pl.col("GameUrl")
    lf = pl.scan_csv(csv_path, infer_schema=False).select(
        pl.col("PuzzleId").alias("puzzle_id"),
        pl.col("FEN").alias("fen_before"),
        moves.alias("moves"),
        moves.list.get(0).alias("setup_uci"),
        moves.list.get(1).alias("solution_uci"),
        pl.col("Rating").cast(pl.Int16).alias("puzzle_rating"),
        pl.col("RatingDeviation").cast(pl.Int16).alias("rating_deviation"),
        pl.col("Popularity").cast(pl.Int8).alias("popularity"),
        pl.col("NbPlays").cast(pl.Int32).alias("nb_plays"),
        pl.col("Themes").str.split(" ").alias("themes"),
        url.str.extract(r"lichess\.org/(\w{8})", 1).alias("game_id"),
        url.str.extract(r"#(\d+)$", 1).cast(pl.Int16).alias("setup_ply"),
        pl.when(url.str.contains("/black#")).then(pl.lit("w")).otherwise(pl.lit("b"))
        .cast(pl.Enum(["w", "b"]))
        .alias("solver_side"),
        pl.col("OpeningTags").str.split(" ").alias("opening_tags"),
        pl.col("DailyDate").cast(pl.Int64).alias("daily_date_ms"),
    )  # fmt: skip
    tmp = out.with_name(out.name + ".tmp")
    lf.sink_parquet(tmp, compression="zstd")
    tmp.replace(out)
    csv_path.unlink()
    return out


def _solves(board: chess.Board, played: chess.Move, solution_uci: str, mate_in_one: bool) -> bool:
    if played.uci() == solution_uci:
        return True
    if mate_in_one and played in board.legal_moves:
        board.push(played)
        mate = board.is_checkmate()
        board.pop()
        return mate
    return False


def join(
    puzzles_path: Path, interim: Path, month: str, out_dir: Path, cfg: PipelineConfig
) -> dict[str, Any]:
    """Puzzles whose source game is in ``month`` -> ``puzzle_positions`` + report."""
    t0 = time.perf_counter()
    games = pl.scan_parquet(interim / "games" / f"month={month}" / "*.parquet")
    moves = pl.scan_parquet(interim / "moves" / f"month={month}" / "*.parquet")
    puzzles = pl.scan_parquet(puzzles_path)

    joined = puzzles.join(
        games.select("game_id", "white_elo", "black_elo", "tc_class", "base_s", "increment_s"),
        on="game_id",
        how="inner",
    ).collect()
    keys = joined.select("game_id", "setup_ply")
    move_cols = [
        "game_id", "ply", "uci", "time_spent_s", "own_clk_before_s", "opp_clk_s",
        "premove_suspect", "low_clock", "berserk", "time_added",
    ]  # fmt: skip
    game_moves = (
        moves.select(move_cols)
        .join(keys.lazy(), on="game_id", how="inner")
        .filter((pl.col("ply") == pl.col("setup_ply")) | (pl.col("ply") == pl.col("setup_ply") + 1))
        .collect()
    )
    by_key = {(r["game_id"], r["ply"]): r for r in game_moves.iter_rows(named=True)}

    rows: list[dict[str, Any]] = []
    outcomes = {"joined": joined.height, "kept": 0, "setup_mismatch": 0, "no_solver_move": 0}
    for p in joined.iter_rows(named=True):
        setup_row = by_key.get((p["game_id"], p["setup_ply"]))
        if setup_row is None or setup_row["uci"] != p["setup_uci"]:
            outcomes["setup_mismatch"] += 1
            continue
        board = chess.Board(p["fen_before"])
        board.push(chess.Move.from_uci(p["setup_uci"]))
        solver_white = board.turn == chess.WHITE
        if ("w" if solver_white else "b") != p["solver_side"]:
            outcomes["setup_mismatch"] += 1
            continue
        solver = by_key.get((p["game_id"], p["setup_ply"] + 1))
        if solver is None:
            outcomes["no_solver_move"] += 1
            continue
        played = chess.Move.from_uci(solver["uci"])
        mate_in_one = "mateIn1" in (p["themes"] or [])
        rows.append(
            {
                "puzzle_id": p["puzzle_id"],
                "game_id": p["game_id"],
                "month": month,
                "ply": p["setup_ply"] + 1,
                "fen": board.fen(),
                "epd": board.epd(en_passant="legal"),
                "solution_uci": p["solution_uci"],
                "solution_line": p["moves"][1:],
                "puzzle_rating": p["puzzle_rating"],
                "rating_deviation": p["rating_deviation"],
                "popularity": p["popularity"],
                "nb_plays": p["nb_plays"],
                "themes": p["themes"],
                "tc_class": p["tc_class"],
                "base_s": p["base_s"],
                "increment_s": p["increment_s"],
                "solver_elo": p["white_elo"] if solver_white else p["black_elo"],
                "opponent_elo": p["black_elo"] if solver_white else p["white_elo"],
                "played_uci": solver["uci"],
                "played_solution": _solves(board, played, p["solution_uci"], mate_in_one),
                **{k: solver[k] for k in move_cols[3:]},
            }
        )
        outcomes["kept"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    write_parquet_atomic(df, out_dir / "puzzle_positions.parquet")
    report = {
        "month": month,
        "puzzles_total": pl.scan_parquet(puzzles_path).select(pl.len()).collect().item(),
        "outcomes": outcomes,
        "seconds": round(time.perf_counter() - t0, 1),
        **(_quality(df, cfg) if df.height else {}),
    }
    write_atomic_json(out_dir / "_report.json", report)
    (out_dir / "_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _quality(df: pl.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    band = cfg.puzzles.report_band_width
    timed = df.filter(pl.col("time_spent_s").is_not_null())
    return {
        "solve_rate": round(df["played_solution"].mean(), 4),  # type: ignore[arg-type]
        "solve_rate_by_puzzle_rating": df.group_by(
            (pl.col("puzzle_rating") // band * band).alias("puzzle_band")
        )
        .agg(pl.len().alias("n"), pl.col("played_solution").mean().round(3).alias("solve_rate"))
        .sort("puzzle_band")
        .to_dicts(),
        "solve_rate_by_solver_elo": df.group_by(
            (pl.col("solver_elo") // band * band).alias("solver_band")
        )
        .agg(pl.len().alias("n"), pl.col("played_solution").mean().round(3).alias("solve_rate"))
        .sort("solver_band")
        .to_dicts(),
        "median_time_by_outcome": timed.group_by("tc_class", "played_solution")
        .agg(pl.len().alias("n"), pl.col("time_spent_s").median().alias("median_s"))
        .sort("tc_class", "played_solution")
        .to_dicts(),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# join_puzzles report — {report['month']}",
        "",
        f"- puzzles in DB: {report['puzzles_total']:,}",
        f"- outcomes: {json.dumps(report['outcomes'])}",
    ]
    if "solve_rate" in report:
        lines.append(f"- overall: the real player found the solution {report['solve_rate']:.1%}")
        for key, label in [
            ("solve_rate_by_puzzle_rating", "puzzle_band"),
            ("solve_rate_by_solver_elo", "solver_band"),
        ]:
            lines += ["", f"| {label} | n | solve rate |", "|---:|---:|---:|"]
            lines += [f"| {r[label]} | {r['n']:,} | {r['solve_rate']:.1%} |" for r in report[key]]
        lines += ["", "| class | solved | n | median T (s) |", "|---|---|---:|---:|"]
        lines += [
            f"| {r['tc_class']} | {r['played_solution']} | {r['n']:,} | {r['median_s']:.0f} |"
            for r in report["median_time_by_outcome"]
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description="Ingest the Lichess puzzle DB and join to games.")
    p.add_argument("month", help="YYYY-MM partition of games/moves to join against")
    p.add_argument(
        "--puzzles-zst", type=Path, default=settings.raw_dir / "lichess_db_puzzle.csv.zst"
    )
    p.add_argument("--interim", type=Path, default=settings.interim_dir, help="games/moves root")
    p.add_argument("--out", type=Path, help="default: <interim>/puzzle_positions/month=YYYY-MM")
    p.add_argument("--reingest", action="store_true", help="rebuild puzzles.parquet")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    puzzles_dir = settings.interim_dir / "puzzles"
    puzzles_path = puzzles_dir / PUZZLES_FILE
    if args.reingest or not puzzles_path.exists():
        t0 = time.perf_counter()
        ingest(args.puzzles_zst, puzzles_dir)
        log.info("ingested puzzles in %.0f s -> %s", time.perf_counter() - t0, puzzles_path)
    out = args.out or args.interim / "puzzle_positions" / f"month={args.month}"
    report = join(puzzles_path, args.interim, args.month, out, cfg)
    log.info("outcomes %s; report in %s", report["outcomes"], out)
    return 0
