"""parse_games job: filtered game parts -> ``games`` and ``moves`` Parquet + quality report.

Input:  ``<filtered>/month=YYYY-MM/part-NNNNNN.parquet`` (from stream_filter)
Output: ``<out>/games/month=YYYY-MM/part-NNNNNN.parquet``
        ``<out>/moves/month=YYYY-MM/part-NNNNNN.parquet`` (+ ``.stats.json`` sidecar,
        written last, marks the part done) and ``<out>/moves/month=YYYY-MM/_report.*``.
Output parts correspond one-to-one to input parts, so reruns skip finished parts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import chess
import polars as pl

from zeitnot.config import ClockConfig, PipelineConfig, get_settings
from zeitnot.data.clocks import TimeControl, compute_move_timings
from zeitnot.data.movetext import parse_movetext
from zeitnot.data.parts import part_paths, read_stats, write_atomic_json, write_parquet_atomic

log = logging.getLogger(__name__)

GAMES_DROP = ["movetext"]
MOVES_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "ply": pl.Int16(),
    "side": pl.Enum(["w", "b"]),
    "uci": pl.String(),
    "clk_after_s": pl.Float32(),
    "own_clk_before_s": pl.Float32(),
    "opp_clk_s": pl.Float32(),
    "time_spent_s": pl.Float32(),
    "eval_white_cp": pl.Int16(),
    "eval_white_mate": pl.Int16(),
    "is_first_move": pl.Boolean(),
    "berserk": pl.Boolean(),
    "premove_suspect": pl.Boolean(),
    "low_clock": pl.Boolean(),
    "time_added": pl.Boolean(),
}
_TIMING_FIELDS = [
    "ply", "side", "clk_after_s", "own_clk_before_s", "opp_clk_s", "time_spent_s",
    "is_first_move", "berserk", "premove_suspect", "low_clock", "time_added",
]  # fmt: skip
_EVAL_CP_LIMIT = 32_000  # Int16 range; real %eval values are far smaller


class GameParseError(ValueError):
    pass


def parse_game_moves(
    game_id: str, movetext: str, tc: TimeControl, tc_class: str, cfg: ClockConfig
) -> dict[str, list[Any]]:
    """Replay one game; return column lists for its ``moves`` rows."""
    tokens = parse_movetext(movetext)
    clocks = [t.clock_s for t in tokens]
    if any(c is None for c in clocks):
        raise GameParseError("missing_clock")
    board = chess.Board()
    ucis = []
    try:
        for t in tokens:
            ucis.append(board.push_san(t.san).uci())
    except ValueError as e:
        raise GameParseError("illegal_move") from e

    timings = compute_move_timings(clocks, tc, tc_class, cfg)  # type: ignore[arg-type]
    cols: dict[str, list[Any]] = {name: [] for name in MOVES_SCHEMA}
    cols["game_id"] = [game_id] * len(tokens)
    cols["uci"] = ucis
    for t in tokens:
        cp = t.eval_white_cp
        cols["eval_white_cp"].append(
            None if cp is None else max(-_EVAL_CP_LIMIT, min(_EVAL_CP_LIMIT, cp))
        )
        cols["eval_white_mate"].append(t.eval_white_mate)
    for name in _TIMING_FIELDS:
        cols[name] = [getattr(m, name) for m in timings]
    return cols


def process_part(
    in_path: str, games_dir: str, moves_dir: str, index: int, month: str, cfg: ClockConfig
) -> dict[str, Any]:
    """Worker: parse every game of one filtered part."""
    t0 = time.perf_counter()
    games = pl.read_parquet(in_path)
    cols: dict[str, list[Any]] = {name: [] for name in MOVES_SCHEMA}
    n_plies: list[int | None] = []
    outcomes: Counter[str] = Counter()
    for game_id, movetext, base_s, inc_s, tc_class in games.select(
        "game_id", "movetext", "base_s", "increment_s", "tc_class"
    ).iter_rows():
        try:
            game_cols = parse_game_moves(
                game_id, movetext, TimeControl(base_s, inc_s), tc_class, cfg
            )
        except GameParseError as e:
            outcomes[str(e)] += 1
            n_plies.append(None)
            continue
        outcomes["parsed"] += 1
        n_plies.append(len(game_cols["uci"]))
        for name, values in game_cols.items():
            cols[name].extend(values)

    games_out = (
        games.with_columns(
            pl.Series("n_plies", n_plies, dtype=pl.Int16),
            pl.lit(month).alias("month"),
            (pl.col("base_s") + 40 * pl.col("increment_s").cast(pl.Int32)).alias("est_duration_s"),
        )
        .filter(pl.col("n_plies").is_not_null())
        .drop(GAMES_DROP)
    )
    moves_out = pl.DataFrame(cols, schema=MOVES_SCHEMA)

    games_path, _ = part_paths(Path(games_dir), index)
    moves_path, stats_path = part_paths(Path(moves_dir), index)
    write_parquet_atomic(games_out, games_path)
    write_parquet_atomic(moves_out, moves_path)
    stats = {
        "index": index,
        "games_in": games.height,
        "outcomes": dict(outcomes),
        "moves": moves_out.height,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    write_atomic_json(stats_path, stats)
    return stats


def quality_report(games_dir: Path, moves_dir: Path, cfg: PipelineConfig) -> dict[str, Any]:
    """Distributions used to sanity-check the data (lazy scans; nothing large in memory)."""
    games = pl.scan_parquet(games_dir / "part-*.parquet")
    moves = pl.scan_parquet(moves_dir / "part-*.parquet")
    band = cfg.sampling.rating_band_width
    timed = (
        moves.filter(pl.col("time_spent_s").is_not_null() & ~pl.col("berserk"))
        .join(
            games.select("game_id", "tc_class", "white_elo", "black_elo", "base_s", "increment_s"),
            on="game_id",
        )
        .with_columns(
            pl.when(pl.col("side") == "w")
            .then(pl.col("white_elo"))
            .otherwise(pl.col("black_elo"))
            .floordiv(band)
            .mul(band)
            .alias("band")
        )
    )
    t_by_class_band = (
        timed.group_by("tc_class", "band")
        .agg(
            pl.len().alias("n"),
            pl.col("time_spent_s").median().alias("median_s"),
            pl.col("time_spent_s").quantile(0.9).alias("p90_s"),
        )
        .sort("tc_class", "band")
        .collect()
    )
    t_by_tc = (
        timed.with_columns(
            (pl.col("base_s").cast(pl.String) + "+" + pl.col("increment_s").cast(pl.String)).alias(
                "tc"
            )
        )
        .group_by("tc_class", "tc")
        .agg(pl.len().alias("n"), pl.col("time_spent_s").median().alias("median_s"))
        .filter(pl.col("n") >= 10_000)
        .sort("tc_class", "median_s")
        .collect()
    )
    flags = moves.select(
        pl.len().alias("moves"),
        *[
            pl.col(f).mean().alias(f"share_{f}")
            for f in ("is_first_move", "berserk", "premove_suspect", "low_clock", "time_added")
        ],
        pl.col("eval_white_cp")
        .is_not_null()
        .or_(pl.col("eval_white_mate").is_not_null())
        .mean()
        .alias("share_with_eval"),
    ).collect()
    flag_row = flags.row(0, named=True)
    return {
        "flags": {k: round(v, 5) if isinstance(v, float) else v for k, v in flag_row.items()},
        "time_spent_by_class_band": t_by_class_band.to_dicts(),
        "median_time_spent_by_time_control": t_by_tc.to_dicts(),
    }


def _report_markdown(month: str, report: dict[str, Any]) -> str:
    q = report["quality"]
    lines = [
        f"# parse_games report — {month}",
        "",
        f"- games in: {report['games_in']:,}; outcomes: {report['outcomes']}",
        f"- moves: {report['moves']:,}; last run: {json.dumps(report['last_run'])}",
        f"- flag shares: {json.dumps(q['flags'])}",
        "",
        "## Median time spent by time control (n >= 10k)",
        "",
        "| class | tc | moves | median s |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['tc_class']} | {r['tc']} | {r['n']:,} | {r['median_s']:.0f} |"
        for r in q["median_time_spent_by_time_control"]
    ]
    lines += [
        "",
        "## Time spent by class x rating band (mover's rating)",
        "",
        "| class | band | moves | median s | p90 s |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['tc_class']} | {r['band']} | {r['n']:,} | {r['median_s']:.0f} | {r['p90_s']:.0f} |"
        for r in q["time_spent_by_class_band"]
    ]
    return "\n".join(lines) + "\n"


def run(
    filtered_dir: Path,
    out_root: Path,
    month: str,
    cfg: PipelineConfig,
    *,
    workers: int,
    max_parts: int | None = None,
) -> dict[str, Any]:
    in_dir = filtered_dir / f"month={month}"
    inputs = sorted(in_dir.glob("part-*.parquet"))[:max_parts]
    if not inputs:
        raise SystemExit(f"no filtered parts in {in_dir}")
    games_dir = out_root / "games" / f"month={month}"
    moves_dir = out_root / "moves" / f"month={month}"
    games_dir.mkdir(parents=True, exist_ok=True)
    moves_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for path in inputs:
        index = int(path.stem.split("-")[1])
        if not part_paths(moves_dir, index)[1].exists():
            todo.append((index, path))

    t0 = time.perf_counter()
    games_done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                process_part, str(path), str(games_dir), str(moves_dir), index, month, cfg.clock
            )
            for index, path in todo
        ]
        for n, fut in enumerate(as_completed(futures), 1):
            games_done += fut.result()["games_in"]
            elapsed = time.perf_counter() - t0
            log.info(
                "parts %d/%d | %s games | %.0f games/s",
                n, len(todo), f"{games_done:,}", games_done / elapsed,
            )  # fmt: skip

    elapsed = time.perf_counter() - t0
    stats = read_stats(moves_dir)
    outcomes: Counter[str] = Counter()
    for s in stats:
        outcomes.update(s["outcomes"])
    report = {
        "parts": len(stats),
        "games_in": sum(s["games_in"] for s in stats),
        "outcomes": dict(outcomes),
        "moves": sum(s["moves"] for s in stats),
        "last_run": {
            "parts_processed": len(todo),
            "parts_skipped": len(inputs) - len(todo),
            "games_processed": games_done,
            "seconds": round(elapsed, 1),
            "games_per_s": round(games_done / elapsed) if todo else None,
            "workers": workers,
        },
        "quality": quality_report(games_dir, moves_dir, cfg),
    }
    write_atomic_json(moves_dir / "_report.json", report)
    (moves_dir / "_report.md").write_text(_report_markdown(month, report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("month", help="YYYY-MM partition to parse")
    p.add_argument("--filtered", type=Path, default=settings.interim_dir / "filtered")
    p.add_argument("--out", type=Path, default=settings.interim_dir)
    p.add_argument("--workers", type=int, default=cfg.parse.workers)
    p.add_argument("--max-parts", type=int, help="only the first N parts (benchmarks)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(
        args.filtered, args.out, args.month, cfg, workers=args.workers, max_parts=args.max_parts
    )
    log.info(
        "%s games -> %s moves; outcomes %s", f"{report['games_in']:,}", f"{report['moves']:,}",
        report["outcomes"],
    )  # fmt: skip
    return 0
