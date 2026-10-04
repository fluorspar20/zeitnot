"""engine_label job: Stockfish labels for sampled positions.

Per position (fixed node budget, single thread, hash cleared first, so labels are
reproducible):

- the top multi-PV lines with score and win% from the side to move's point of view;
- the depth at which the final best move first became the engine's top choice, and
  the depth from which it stayed on top (a proxy for how hard the move is to find);
- for each move actually played in our data that is not among those lines, a second
  search restricted to that move.

Layout: ``<out>/part-NNNNNN.parquet`` (+ ``.stats.json`` done marker) per batch of
positions, ``_manifest.json`` (engine version and settings; a mismatch refuses to
resume) and ``_report.json``/``.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import chess
import chess.engine
import polars as pl

from zeitnot.config import AcceptabilityConfig, EngineConfig, PipelineConfig, get_settings
from zeitnot.data.parts import (
    part_paths,
    read_stats,
    write_atomic_json,
    write_parquet_atomic,
)
from zeitnot.engine.stockfish import find_stockfish, open_engine
from zeitnot.engine.winpct import win_pct

log = logging.getLogger(__name__)

_LINE = pl.Struct({"uci": pl.String, "cp": pl.Int32, "mate": pl.Int32, "win_pct": pl.Float32})
_PLAYED = pl.Struct(
    {
        "uci": pl.String,
        "cp": pl.Int32,
        "mate": pl.Int32,
        "win_pct": pl.Float32,
        "in_multipv": pl.Boolean,
        "loss_win_pct": pl.Float32,
    }
)
SCHEMA: dict[str, pl.DataType] = {
    "position_id": pl.String(),
    "epd": pl.String(),
    "best_uci": pl.String(),
    "best_win_pct": pl.Float32(),
    "lines": pl.List(_LINE),
    "depth": pl.Int16(),
    "nodes": pl.Int64(),
    "best_move_first_depth": pl.Int16(),
    "best_move_stable_depth": pl.Int16(),
    "played": pl.List(_PLAYED),
    "seconds": pl.Float32(),
}


def depth_profile(top_by_depth: dict[int, str], best_uci: str) -> tuple[int | None, int | None]:
    """(first depth at which ``best_uci`` was the top move, depth from which it stayed on top)."""
    depths = sorted(top_by_depth)
    first = next((d for d in depths if top_by_depth[d] == best_uci), None)
    stable = None
    for d in reversed(depths):
        if top_by_depth[d] != best_uci:
            break
        stable = d
    return first, stable


def _line(info: chess.engine.InfoDict, turn: chess.Color, acc: AcceptabilityConfig) -> dict:
    score = info["score"].pov(turn)
    return {
        "uci": info["pv"][0].uci(),
        "cp": score.score(),
        "mate": score.mate(),
        "win_pct": win_pct(score, acc),
    }


def label_position(
    engine: chess.engine.SimpleEngine,
    fen: str,
    played_ucis: list[str],
    engine_cfg: EngineConfig,
    acc: AcceptabilityConfig,
) -> dict[str, Any]:
    """Multi-PV label of one position plus the evaluation of each played move."""
    t0 = time.perf_counter()
    board = chess.Board(fen)
    game = object()  # a new game token makes python-chess send ucinewgame (clears the hash)
    multipv = min(engine_cfg.multipv, board.legal_moves.count())
    top_by_depth: dict[int, str] = {}
    with engine.analysis(
        board, chess.engine.Limit(nodes=engine_cfg.nodes), multipv=multipv, game=game
    ) as analysis:
        for info in analysis:
            exact = "pv" in info and not info.get("lowerbound") and not info.get("upperbound")
            if exact and info.get("multipv", 1) == 1 and "depth" in info:
                top_by_depth[info["depth"]] = info["pv"][0].uci()
        final = [i for i in analysis.multipv if "pv" in i and "score" in i]

    lines = sorted((_line(i, board.turn, acc) for i in final), key=lambda x: -x["win_pct"])
    best = lines[0]
    first_depth, stable_depth = depth_profile(top_by_depth, best["uci"])
    by_uci = {line["uci"]: line for line in lines}

    played = []
    for uci in played_ucis:
        if uci in by_uci:
            entry = {**by_uci[uci], "in_multipv": True}
        else:
            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=engine_cfg.played_move_nodes),
                root_moves=[chess.Move.from_uci(uci)],
                game=game,
            )
            entry = {**_line(info, board.turn, acc), "uci": uci, "in_multipv": False}
        # Raw loss: a separately searched move can score slightly above the best line.
        entry["loss_win_pct"] = best["win_pct"] - entry["win_pct"]
        played.append(entry)

    return {
        "best_uci": best["uci"],
        "best_win_pct": best["win_pct"],
        "lines": lines,
        "depth": final[0].get("depth"),
        "nodes": final[0].get("nodes"),
        "best_move_first_depth": first_depth,
        "best_move_stable_depth": stable_depth,
        "played": played,
        "seconds": time.perf_counter() - t0,
    }


def label_batch(
    index: int,
    rows: list[tuple[str, str, str, list[str]]],
    out_dir: str,
    engine_path: str,
    engine_cfg: EngineConfig,
    acc: AcceptabilityConfig,
) -> dict[str, Any]:
    """Worker: label one batch of (position_id, epd, fen, played_ucis) and write its part.

    The engine lives for one batch only: a long-lived engine per worker process would
    keep that process (and the pool) from shutting down, and could outlive a crash.
    """
    t0 = time.perf_counter()
    out = []
    with open_engine(Path(engine_path), engine_cfg) as engine:
        for position_id, epd, fen, played_ucis in rows:
            label = label_position(engine, fen, played_ucis, engine_cfg, acc)
            out.append({"position_id": position_id, "epd": epd, **label})
    parquet, stats_path = part_paths(Path(out_dir), index)
    write_parquet_atomic(pl.DataFrame(out, schema=SCHEMA), parquet)
    stats = {
        "index": index,
        "positions": len(out),
        "played_searches": sum(not p["in_multipv"] for r in out for p in r["played"]),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    write_atomic_json(stats_path, stats)
    return stats


def positions_to_label(
    positions_path: Path, evaldb_path: Path | None, cfg: PipelineConfig
) -> pl.DataFrame:
    """One row per distinct position with the moves played from it.

    With ``evaldb_path``, positions the evaluation database fully covers are skipped:
    enough lines, and every played move among them.
    """
    df = (
        pl.read_parquet(positions_path, columns=["position_id", "epd", "fen", "played_uci"])
        .group_by("position_id")
        .agg(
            pl.col("epd").first(),
            pl.col("fen").first(),
            pl.col("played_uci").unique().sort().alias("played_ucis"),
        )
        .sort("position_id")
    )
    if evaldb_path is None:
        return df
    db = pl.read_parquet(evaldb_path, columns=["epd", "n_pvs", "pvs"]).with_columns(
        pl.col("pvs").list.eval(pl.element().struct.field("uci")).alias("db_ucis")
    )
    covered = (
        df.join(db, on="epd", how="inner")
        .filter(
            (pl.col("n_pvs") >= cfg.engine.multipv)
            & (pl.col("played_ucis").list.set_difference("db_ucis").list.len() == 0)
        )
        .select("position_id")
    )
    return df.join(covered, on="position_id", how="anti")


def _check_manifest(out_dir: Path, current: dict[str, Any]) -> None:
    manifest = out_dir / "_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != current:
            raise SystemExit(
                f"{out_dir} holds labels made with different settings "
                f"(previous: {previous}); use a fresh output directory or the same settings"
            )
    else:
        write_atomic_json(manifest, current)


def build_report(out_dir: Path, run_info: dict[str, Any]) -> dict[str, Any]:
    stats = read_stats(out_dir)
    labels = pl.scan_parquet(out_dir / "part-*.parquet")
    summary = labels.select(
        pl.len().alias("positions"),
        pl.col("seconds").mean().round(3).alias("mean_seconds"),
        pl.col("depth").median().alias("median_depth"),
        pl.col("nodes").median().alias("median_nodes"),
        pl.col("best_move_first_depth").median().alias("median_first_depth"),
        (pl.col("best_move_first_depth") <= 1).mean().round(4).alias("share_best_at_depth_1"),
        pl.col("lines").list.len().mean().round(2).alias("mean_lines"),
    ).collect()
    return {
        "parts": len(stats),
        "played_searches": sum(s["played_searches"] for s in stats),
        **summary.row(0, named=True),
        "last_run": run_info,
    }


def run(
    positions_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    engine_path: Path,
    *,
    workers: int,
    evaldb_path: Path | None = None,
    max_positions: int | None = None,
) -> dict[str, Any]:
    todo_positions = positions_to_label(positions_path, evaldb_path, cfg)
    if max_positions is not None:
        todo_positions = todo_positions.head(max_positions)
    out_dir.mkdir(parents=True, exist_ok=True)
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as probe:
        engine_name = probe.id.get("name", "unknown")
    _check_manifest(
        out_dir,
        {
            "engine": engine_name,
            "engine_settings": cfg.engine.model_dump(exclude={"workers"}),
            "acceptability": cfg.acceptability.model_dump(exclude={"tau_win_pct"}),
            "positions_file": positions_path.name,
            "positions": todo_positions.height,
            "skip_evaldb_covered": evaldb_path is not None,
        },
    )

    size = cfg.engine.batch_size
    batches = [
        (i // size, todo_positions.slice(i, size).rows())
        for i in range(0, todo_positions.height, size)
    ]
    todo = [(i, rows) for i, rows in batches if not part_paths(out_dir, i)[1].exists()]
    log.info(
        "%s positions in %d batches; %d batches to do with %d workers (%s)",
        f"{todo_positions.height:,}", len(batches), len(todo), workers, engine_name,
    )  # fmt: skip

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                label_batch, i, rows, str(out_dir), str(engine_path), cfg.engine, cfg.acceptability
            )
            for i, rows in todo
        ]
        remaining = sum(len(rows) for _, rows in todo)
        for n, fut in enumerate(as_completed(futures), 1):
            done += fut.result()["positions"]
            rate = done / (time.perf_counter() - t0)
            log.info(
                "batch %d/%d | %s positions | %.2f positions/s | ETA %.0f min",
                n, len(todo), f"{done:,}", rate, (remaining - done) / rate / 60,
            )  # fmt: skip

    elapsed = time.perf_counter() - t0
    report = build_report(
        out_dir,
        {
            "positions_labeled": done,
            "batches_skipped": len(batches) - len(todo),
            "seconds": round(elapsed, 1),
            "positions_per_s": round(done / elapsed, 3) if done else None,
            "workers": workers,
        },
    )
    write_atomic_json(out_dir / "_report.json", report)
    (out_dir / "_report.md").write_text(
        "# engine_label report\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items()) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    positions_dir = settings.interim_dir / "positions"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--positions", type=Path, default=positions_dir / "sampled_positions.parquet")
    p.add_argument("--out", type=Path, help="default: <positions folder>/labels")
    p.add_argument("--workers", type=int, default=cfg.engine.workers)
    p.add_argument("--max-positions", type=int, help="label only the first N (benchmarks)")
    p.add_argument(
        "--skip-evaldb-covered",
        type=Path,
        metavar="EVALDB_MATCHES",
        help="skip positions fully covered by this evaldb_matches.parquet",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(
        args.positions,
        args.out or args.positions.parent / "labels",
        cfg,
        find_stockfish(settings),
        workers=args.workers,
        evaldb_path=args.skip_evaldb_covered,
        max_positions=args.max_positions,
    )
    log.info("report: %s", json.dumps(report))
    return 0
