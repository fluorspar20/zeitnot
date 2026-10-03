"""lookup_evals job: find our positions in the Lichess evaluation database.

The database is one JSON object per line (~410M lines)::

    {"fen": "<pieces> <side> <castling> <ep>", "evals": [{"pvs": [{"cp"|"mate", "line"}],
                                                          "knodes", "depth"}, ...]}

- ``fen`` has no move counters and lists the en passant square only when a capture
  is legal: exactly python-chess ``board.epd(en_passant="legal")``.
- ``cp``/``mate`` are from White's point of view.
- ``line`` is in UCI_Chess960 notation (castling written as king-takes-rook).

The file is streamed once. Each line's key is sliced out as bytes and tested against
an in-memory set of wanted keys; only matching lines are parsed.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import chess
import polars as pl

from zeitnot.config import AcceptabilityConfig, PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic
from zeitnot.data.stream import iter_line_chunks
from zeitnot.engine.winpct import win_pct_from_cp

log = logging.getLogger(__name__)

_KEY_START = len(b'{"fen":"')
_WANTED: frozenset[bytes] = frozenset()

PV_TYPE = pl.List(
    pl.Struct({"uci": pl.String, "cp": pl.Int32, "mate": pl.Int32, "win_pct": pl.Float32})
)
SCHEMA: dict[str, pl.DataType] = {
    "epd": pl.String(),
    "n_evals": pl.Int16(),
    "best_uci": pl.String(),
    "best_cp": pl.Int32(),
    "best_mate": pl.Int32(),
    "best_win_pct": pl.Float32(),
    "best_knodes": pl.Int64(),
    "best_depth": pl.Int16(),
    "n_pvs": pl.Int16(),
    "multi_knodes": pl.Int64(),
    "multi_depth": pl.Int16(),
    "pvs": PV_TYPE,
}


def _init_worker(wanted_paths: list[str]) -> None:
    global _WANTED
    wanted: set[bytes] = set()
    for path in wanted_paths:
        epds = pl.read_parquet(path, columns=["epd"])["epd"]
        wanted.update(e.encode("ascii") for e in epds)
    _WANTED = frozenset(wanted)


def scan_chunk(data: bytes) -> tuple[int, list[bytes]]:
    """Worker: ``(lines seen, matching raw lines)`` for one chunk."""
    wanted = _WANTED
    hits = []
    lines = data.split(b"\n")
    for line in lines:
        if line[_KEY_START : line.find(b'"', _KEY_START)] in wanted:
            hits.append(line)
    return len(lines) - 1, hits


def _stm_score(pv: dict[str, Any], white_to_move: bool) -> tuple[int | None, int | None]:
    """(cp, mate) from the side to move's point of view."""
    sign = 1 if white_to_move else -1
    cp, mate = pv.get("cp"), pv.get("mate")
    return (None if cp is None else sign * cp), (None if mate is None else sign * mate)


def _win_pct(cp: int | None, mate: int | None, acc: AcceptabilityConfig) -> float:
    if mate is not None:
        # "mate": 0 does not occur for positions with a move to play; treat as lost.
        return win_pct_from_cp(acc.cp_ceiling if mate > 0 else -acc.cp_ceiling, acc)
    return win_pct_from_cp(cp or 0, acc)


def _first_move_uci(board: chess.Board, line: str) -> str | None:
    """First move of a PV in standard UCI (e1h1 -> e1g1), or None if it is not legal."""
    try:
        move = chess.Move.from_uci(line.split(" ", 1)[0])
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return board.uci(move, chess960=False)


def parse_record(raw: bytes, cfg: PipelineConfig) -> dict[str, Any] | None:
    """One database line -> a row of SCHEMA, or None if no eval is usable."""
    rec = json.loads(raw)
    board = chess.Board(rec["fen"] + " 0 1")
    white = board.turn == chess.WHITE
    evals = [
        e for e in rec["evals"] if e.get("knodes", 0) >= cfg.evaldb.min_knodes and e.get("pvs")
    ]
    if not evals:
        return None
    best = max(evals, key=lambda e: e["knodes"])
    multi = max(evals, key=lambda e: (len(e["pvs"]), e["knodes"]))
    best_uci = _first_move_uci(board, best["pvs"][0].get("line", ""))
    if best_uci is None:
        return None
    best_cp, best_mate = _stm_score(best["pvs"][0], white)
    pvs = []
    for pv in multi["pvs"]:
        uci = _first_move_uci(board, pv.get("line", ""))
        if uci is None:
            continue
        cp, mate = _stm_score(pv, white)
        pvs.append(
            {"uci": uci, "cp": cp, "mate": mate, "win_pct": _win_pct(cp, mate, cfg.acceptability)}
        )
    pvs.sort(key=lambda p: -p["win_pct"])
    return {
        "epd": rec["fen"],
        "n_evals": len(rec["evals"]),
        "best_uci": best_uci,
        "best_cp": best_cp,
        "best_mate": best_mate,
        "best_win_pct": _win_pct(best_cp, best_mate, cfg.acceptability),
        "best_knodes": best["knodes"],
        "best_depth": best["depth"],
        "n_pvs": len(pvs),
        "multi_knodes": multi["knodes"],
        "multi_depth": multi["depth"],
        "pvs": pvs,
    }


def coverage_report(positions: pl.DataFrame, matches: pl.DataFrame, cfg: PipelineConfig) -> dict:
    """How much Stockfish work the database saves, per position and per observation."""
    k = cfg.engine.multipv
    obs = positions.join(matches, on="epd", how="left").with_columns(
        pl.col("best_uci").is_not_null().alias("covered"),
        # Enough lines to grade most moves: the configured multi-PV, or every legal move.
        (pl.col("n_pvs") >= pl.min_horizontal(pl.lit(k), pl.col("n_legal_moves")))
        .fill_null(False)
        .alias("covered_multipv"),
        pl.col("pvs")
        .list.eval(pl.element().struct.field("uci"))
        .list.contains(pl.col("played_uci"))
        .fill_null(False)
        .alias("played_in_pvs"),
    )

    def shares(by: str) -> list[dict[str, Any]]:
        return (
            obs.group_by(by)
            .agg(
                pl.len().alias("n"),
                pl.col("covered").mean().round(4),
                pl.col("covered_multipv").mean().round(4),
                pl.col("played_in_pvs").mean().round(4),
            )
            .sort(by)
            .to_dicts()
        )

    covered = obs.filter(pl.col("covered"))
    return {
        "observations": obs.height,
        "distinct_positions": positions["epd"].n_unique(),
        "matched_positions": positions.select("epd")
        .unique()
        .join(matches, on="epd", how="semi")
        .height,
        "share_covered": round(obs["covered"].mean(), 4),  # type: ignore[arg-type]
        "share_covered_multipv": round(obs["covered_multipv"].mean(), 4),  # type: ignore[arg-type]
        "share_played_move_in_pvs": round(obs["played_in_pvs"].mean(), 4),  # type: ignore[arg-type]
        "still_need_engine": int((~obs["covered_multipv"]).sum()),
        "median_best_knodes": covered["best_knodes"].median(),
        "median_best_depth": covered["best_depth"].median(),
        "by_phase": shares("phase"),
        "by_ply_bucket": shares("ply_bucket"),
        "by_class": shares("tc_class"),
    }


def _markdown(r: dict[str, Any]) -> str:
    lines = [
        "# lookup_evals report",
        "",
        f"- database lines scanned: {r['lines_scanned']:,} in {r['seconds']:.0f} s "
        f"({r['lines_per_s']:,} lines/s)",
        f"- observations: {r['observations']:,} ({r['distinct_positions']:,} distinct positions); "
        f"matched positions: {r['matched_positions']:,}",
        f"- covered by any usable eval: {r['share_covered']:.1%}",
        f"- covered with enough lines (multi-PV): {r['share_covered_multipv']:.1%}",
        f"- played move among the stored lines: {r['share_played_move_in_pvs']:.1%}",
        f"- observations still needing our own engine: {r['still_need_engine']:,}",
        f"- matched evals: median {r['median_best_knodes']:,.0f} knodes, "
        f"depth {r['median_best_depth']:.0f}",
    ]
    for key in ("by_phase", "by_ply_bucket", "by_class"):
        name = key.removeprefix("by_")
        lines += [
            "",
            f"| {name} | n | covered | multi-PV | played in PVs |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {row[name if name != 'class' else 'tc_class']} | {row['n']:,} | "
            f"{row['covered']:.1%} | {row['covered_multipv']:.1%} | {row['played_in_pvs']:.1%} |"
            for row in r[key]
        ]
    return "\n".join(lines) + "\n"


def run(
    evaldb_path: Path,
    positions_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    *,
    workers: int,
    chunk_mb: int,
    max_in_flight: int,
    max_chunks: int | None = None,
    extra_epd_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Scan the database once for the EPDs of ``positions_path`` and of any extra files.

    All matches are written; the coverage report describes ``positions_path`` only.
    """
    t0 = time.perf_counter()
    positions = pl.read_parquet(positions_path)
    raw_hits: list[bytes] = []
    n_lines = 0
    pending: set[Future[tuple[int, list[bytes]]]] = set()

    def collect(futures: set[Future[tuple[int, list[bytes]]]]) -> None:
        nonlocal n_lines
        for fut in futures:
            seen, hits = fut.result()
            n_lines += seen
            raw_hits.extend(hits)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=([str(positions_path), *map(str, extra_epd_paths)],),
    ) as pool:
        for index, data in iter_line_chunks(evaldb_path, chunk_mb * 1_000_000):
            if max_chunks is not None and index >= max_chunks:
                break
            if len(pending) >= max_in_flight:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)
                if index % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    log.info(
                        "chunk %d | %s lines | %s hits | %.0f lines/s",
                        index, f"{n_lines:,}", f"{len(raw_hits):,}", n_lines / elapsed,
                    )  # fmt: skip
            pending.add(pool.submit(scan_chunk, data))
        if pending:
            collect(set(wait(pending).done))

    rows = [r for r in (parse_record(raw, cfg) for raw in raw_hits) if r is not None]
    matches = pl.DataFrame(rows, schema=SCHEMA).unique("epd", keep="first", maintain_order=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(matches, out_dir / "evaldb_matches.parquet")

    seconds = time.perf_counter() - t0
    report = {
        "evaldb": evaldb_path.name,
        "lines_scanned": n_lines,
        "raw_hits": len(raw_hits),
        "seconds": round(seconds, 1),
        "lines_per_s": round(n_lines / seconds),
        "min_knodes": cfg.evaldb.min_knodes,
        "extra_epd_files": [p.name for p in extra_epd_paths],
        "matched_positions_all_files": matches.height,
        **coverage_report(positions, matches, cfg),
    }
    write_atomic_json(out_dir / "_evaldb_report.json", report)
    (out_dir / "_evaldb_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--evaldb", type=Path, default=settings.raw_dir / "lichess_db_eval.jsonl.zst")
    p.add_argument(
        "--positions",
        type=Path,
        default=settings.interim_dir / "positions" / "sampled_positions.parquet",
    )
    p.add_argument(
        "--also",
        type=Path,
        nargs="*",
        default=[],
        help="more Parquet files with an 'epd' column to look up in the same pass (e.g. puzzles)",
    )
    p.add_argument("--out", type=Path, help="default: the positions file's folder")
    p.add_argument("--workers", type=int, default=cfg.stream.workers)
    p.add_argument("--chunk-mb", type=int, default=cfg.stream.chunk_mb)
    p.add_argument("--max-in-flight", type=int, default=cfg.stream.max_in_flight)
    p.add_argument("--max-chunks", type=int, help="stop after this many chunks (benchmarks)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(
        args.evaldb,
        args.positions,
        args.out or args.positions.parent,
        cfg,
        workers=args.workers,
        chunk_mb=args.chunk_mb,
        max_in_flight=args.max_in_flight,
        max_chunks=args.max_chunks,
        extra_epd_paths=tuple(args.also),
    )
    log.info(
        "scanned %s lines; covered %.1f%% of observations (%.1f%% with multi-PV)",
        f"{report['lines_scanned']:,}",
        100 * report["share_covered"],
        100 * report["share_covered_multipv"],
    )
    return 0
