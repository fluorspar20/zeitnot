"""stream_filter job: Lichess ``.pgn.zst`` -> filtered games as Parquet parts + quality report.

Layout: ``<out>/month=YYYY-MM/part-NNNNNN.parquet`` with a ``.stats.json`` sidecar per
part. The sidecar is written last and marks the chunk as done, so an interrupted
run resumes by re-streaming the input and skipping finished chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import fields
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl

from zeitnot.config import FilterConfig, PipelineConfig, TimeControlConfig, get_settings
from zeitnot.data.game_filter import KeptGame, filter_game
from zeitnot.data.stream import iter_game_chunks, split_games

log = logging.getLogger(__name__)

SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "utc_date": pl.String(),
    "utc_time": pl.String(),
    "event": pl.String(),
    "white": pl.String(),
    "black": pl.String(),
    "white_elo": pl.Int16(),
    "black_elo": pl.Int16(),
    "white_title": pl.String(),
    "black_title": pl.String(),
    "result": pl.String(),
    "termination": pl.String(),
    "eco": pl.String(),
    "opening": pl.String(),
    "base_s": pl.Int32(),
    "increment_s": pl.Int16(),
    "tc_class": pl.String(),
    "has_evals": pl.Boolean(),
    "movetext": pl.String(),
}
assert list(SCHEMA) == [f.name for f in fields(KeptGame)]

_MONTH_RE = re.compile(r"(\d{4}-\d{2})")
_ELO_BAND = 100


def part_paths(out_dir: Path, index: int) -> tuple[Path, Path]:
    stem = out_dir / f"part-{index:06d}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".stats.json")


def _write_atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def process_chunk(
    index: int, data: bytes, out_dir: str, filt: FilterConfig, tcs: TimeControlConfig
) -> dict[str, Any]:
    """Worker: filter one chunk, write its Parquet part and stats sidecar."""
    t0 = time.perf_counter()
    reasons: Counter[str] = Counter()
    kept_by_class: Counter[str] = Counter()
    player_bands: Counter[str] = Counter()  # "<class>|<band>" per player-game
    kept: list[KeptGame] = []
    for text in split_games(data.decode("utf-8", errors="replace")):
        game, reason = filter_game(text, filt, tcs)
        reasons[reason] += 1
        if game is not None:
            kept.append(game)
            kept_by_class[game.tc_class] += 1
            for elo in (game.white_elo, game.black_elo):
                player_bands[f"{game.tc_class}|{elo // _ELO_BAND * _ELO_BAND}"] += 1

    df = pl.DataFrame(
        {name: [getattr(g, name) for g in kept] for name in SCHEMA}, schema=SCHEMA
    ).with_columns(pl.col("utc_date").str.strptime(pl.Date, "%Y.%m.%d", strict=False))
    parquet, stats_path = part_paths(Path(out_dir), index)
    tmp = parquet.with_name(parquet.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(parquet)

    stats = {
        "index": index,
        "bytes": len(data),
        "games": sum(reasons.values()),
        "kept": len(kept),
        "reasons": dict(reasons),
        "kept_by_class": dict(kept_by_class),
        "player_games_by_class_band": dict(player_bands),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    _write_atomic_json(stats_path, stats)
    return stats


def _check_manifest(out_dir: Path, input_path: Path, chunk_mb: int) -> None:
    """Chunk indices are only comparable across runs with the same input and chunk size."""
    manifest = out_dir / "_manifest.json"
    current = {"input": input_path.name, "chunk_mb": chunk_mb}
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != current:
            raise SystemExit(
                f"{out_dir} was produced with {previous}, not {current}; "
                "use a fresh output directory or the same settings"
            )
    else:
        _write_atomic_json(manifest, current)


def build_report(out_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Aggregate all chunk sidecars (from this and earlier runs) into one report."""
    totals: dict[str, Counter[str]] = {
        "reasons": Counter(),
        "kept_by_class": Counter(),
        "player_games_by_class_band": Counter(),
    }
    n_chunks = n_games = n_kept = n_bytes = 0
    for path in sorted(out_dir.glob("part-*.stats.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        n_chunks += 1
        n_games += s["games"]
        n_kept += s["kept"]
        n_bytes += s["bytes"]
        for key, counter in totals.items():
            counter.update(s[key])
    return {
        "chunks": n_chunks,
        "decompressed_bytes": n_bytes,
        "games_in": n_games,
        "games_kept": n_kept,
        "kept_share": round(n_kept / n_games, 4) if n_games else None,
        "drop_reasons": dict(totals["reasons"].most_common()),
        "kept_by_class": dict(totals["kept_by_class"].most_common()),
        "player_games_by_class_band": dict(sorted(totals["player_games_by_class_band"].items())),
        "last_run": run,
    }


def _report_markdown(month: str, report: dict[str, Any]) -> str:
    lines = [
        f"# stream_filter report — {month}",
        "",
        f"- chunks: {report['chunks']}, decompressed: {report['decompressed_bytes'] / 1e9:.2f} GB",
        f"- games in: {report['games_in']:,}, kept: {report['games_kept']:,} "
        f"({(report['kept_share'] or 0):.1%})",
        f"- last run: {json.dumps(report['last_run'])}",
        "",
        "| outcome | games | share |",
        "|---|---:|---:|",
    ]
    for reason, n in report["drop_reasons"].items():
        lines.append(f"| {reason} | {n:,} | {n / max(report['games_in'], 1):.2%} |")
    lines += ["", "| kept class | games |", "|---|---:|"]
    lines += [f"| {c} | {n:,} |" for c, n in report["kept_by_class"].items()]
    return "\n".join(lines) + "\n"


def run(
    input_path: Path,
    out_root: Path,
    month: str,
    cfg: PipelineConfig,
    *,
    workers: int,
    chunk_mb: int,
    max_in_flight: int,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    out_dir = out_root / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_manifest(out_dir, input_path, chunk_mb)

    t0 = time.perf_counter()
    done = skipped = games = bytes_done = 0
    pending: set[Future[dict[str, Any]]] = set()

    def collect(futures: set[Future[dict[str, Any]]]) -> None:
        nonlocal done, games, bytes_done
        for fut in futures:
            stats = fut.result()  # re-raises worker errors
            done += 1
            games += stats["games"]
            bytes_done += stats["bytes"]
        elapsed = time.perf_counter() - t0
        log.info(
            "chunks done %d (skipped %d) | %s games | %.0f games/s | %.0f MB/s decompressed",
            done, skipped, f"{games:,}", games / elapsed, bytes_done / 1e6 / elapsed,
        )  # fmt: skip

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for index, data in iter_game_chunks(input_path, chunk_mb * 1_000_000):
            if max_chunks is not None and index >= max_chunks:
                break
            if part_paths(out_dir, index)[1].exists():
                skipped += 1
                continue
            if len(pending) >= max_in_flight:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)
            pending.add(
                pool.submit(process_chunk, index, data, str(out_dir), cfg.filter, cfg.time_controls)
            )
        if pending:
            collect(set(wait(pending).done))

    elapsed = time.perf_counter() - t0
    run_info = {
        "input": input_path.name,
        "chunks_processed": done,
        "chunks_skipped": skipped,
        "games_processed": games,
        "seconds": round(elapsed, 1),
        "games_per_s": round(games / elapsed) if elapsed else None,
        "decompressed_mb_per_s": round(bytes_done / 1e6 / elapsed, 1) if elapsed else None,
        "workers": workers,
        "chunk_mb": chunk_mb,
    }
    report = build_report(out_dir, run_info)
    _write_atomic_json(out_dir / "_report.json", report)
    (out_dir / "_report.md").write_text(_report_markdown(month, report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="Lichess .pgn.zst file (complete or a prefix)")
    p.add_argument("--month", help="YYYY-MM; default: parsed from the file name")
    p.add_argument("--out", type=Path, default=settings.interim_dir / "filtered")
    p.add_argument("--workers", type=int, default=cfg.stream.workers)
    p.add_argument("--chunk-mb", type=int, default=cfg.stream.chunk_mb)
    p.add_argument("--max-in-flight", type=int, default=cfg.stream.max_in_flight)
    p.add_argument("--max-chunks", type=int, help="stop after this many chunks (benchmarks)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    month = args.month
    if month is None:
        m = _MONTH_RE.search(args.input.name)
        if m is None:
            p.error("cannot infer --month from the file name")
        month = m.group(1)

    report = run(
        args.input,
        args.out,
        month,
        cfg,
        workers=args.workers,
        chunk_mb=args.chunk_mb,
        max_in_flight=args.max_in_flight,
        max_chunks=args.max_chunks,
    )
    log.info(
        "kept %s of %s games (%.1f%%); report in %s",
        f"{report['games_kept']:,}", f"{report['games_in']:,}",
        100 * (report["kept_share"] or 0), args.out / f"month={month}",
    )  # fmt: skip
    return 0
