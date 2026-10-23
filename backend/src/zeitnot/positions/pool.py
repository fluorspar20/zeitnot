"""build_pool job: the positions the Trainer serves, with everything the app needs.

One row per distinct position from the labeled observations that the Trainer may
serve (trainable categories, configured position states), with:

- the engine's candidate moves (multi-PV lines with win%) for grading and display;
- ``prev_fen``: the position before the opponent's last move, so the app can show
  that move being played before the clock starts. A FEN cannot be stepped back
  (a captured piece is not recorded), so each source game is replayed; the replay
  must reproduce the stored position exactly.

A position reached in several sampled games keeps its first source (by game id); the
budgets do not depend on the source, only the "games near the target rating" filter
does.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import chess
import polars as pl

from zeitnot.config import PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic
from zeitnot.positions.categories import TRAINABLE

log = logging.getLogger(__name__)

POOL_COLUMNS = [
    "position_id", "fen", "prev_fen", "opp_prev_uci", "category", "phase", "move_number",
    "side", "best_uci", "best_win_pct", "lines", "n_acceptable_lines", "gap_1_2",
    "source_game_id", "source_month", "source_tc_class", "source_rating",
]  # fmt: skip


def select_pool(obs: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Distinct servable positions, first source game by id."""
    return (
        obs.filter(
            pl.col("loss_win_pct").is_not_null()
            & pl.col("category").cast(pl.String).is_in(TRAINABLE)
            & pl.col("position_state").cast(pl.String).is_in(cfg.trainer.position_states)
            & pl.col("opp_prev_uci").is_not_null()
        )
        .sort("game_id", "ply")
        .unique("position_id", keep="first", maintain_order=True)
        .select(
            "position_id", "fen", "opp_prev_uci",
            pl.col("category").cast(pl.String), "phase", "move_number",
            pl.col("side").cast(pl.String), "best_uci", "best_win_pct", "n_acceptable_lines",
            "gap_1_2", "game_id", "ply", "month", "tc_class", "mover_elo",
        )
    )  # fmt: skip


def previous_fens(wanted: pl.DataFrame, moves: pl.LazyFrame) -> pl.DataFrame:
    """FEN before the opponent's last move and the replayed FEN, per (game_id, ply)."""
    needed: dict[str, set[int]] = {}
    for game_id, ply in wanted.select("game_id", "ply").iter_rows():
        needed.setdefault(game_id, set()).add(ply)
    games = (
        moves.select("game_id", "ply", "uci")
        .join(wanted.select("game_id").unique().lazy(), on="game_id", how="semi")
        .sort("game_id", "ply")
        .group_by("game_id", maintain_order=True)
        .agg("uci")
        .collect()
    )
    rows = []
    for game_id, ucis in games.iter_rows():
        plies = needed[game_id]
        last = max(plies)
        board = chess.Board()
        prev_fen = None
        # Board before move `ply` = after ply - 1 moves; before the opponent's move = ply - 2.
        for ply, uci in enumerate(ucis, start=1):
            if ply in plies:
                rows.append((game_id, ply, prev_fen, board.fen()))
            if ply == last:
                break
            prev_fen = board.fen()
            board.push(chess.Move.from_uci(uci))
    return pl.DataFrame(
        rows,
        schema={"game_id": pl.String, "ply": pl.Int16, "prev_fen": pl.String,
                "replayed_fen": pl.String},
        orient="row",
    )  # fmt: skip


def build(
    obs: pl.DataFrame, labels: pl.LazyFrame, moves: pl.LazyFrame, cfg: PipelineConfig
) -> pl.DataFrame:
    chosen = select_pool(obs, cfg)
    replayed = previous_fens(chosen, moves)
    lines = labels.select("position_id", "lines").collect()
    pool = (
        chosen.join(replayed, on=["game_id", "ply"], how="left")
        .join(lines, on="position_id", how="left")
        .rename({
            "game_id": "source_game_id", "month": "source_month",
            "tc_class": "source_tc_class", "mover_elo": "source_rating",
        })
    )  # fmt: skip
    missing = pool.filter(pl.col("replayed_fen").is_null()).height
    mismatched = pool.filter(pl.col("replayed_fen") != pl.col("fen")).height
    if missing or mismatched:
        raise ValueError(f"replay failed: {missing} games missing, {mismatched} FEN mismatches")
    if pool["lines"].is_null().any():
        raise ValueError("positions without engine lines")
    return pool.select(POOL_COLUMNS).sort("position_id")


def quality_report(pool: pl.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    band = cfg.puzzles.report_band_width
    return {
        "positions": pool.height,
        "position_states": cfg.trainer.position_states,
        "by_category": dict(pool.group_by("category").len().sort("category").iter_rows()),
        "by_source_class": dict(pool.group_by("source_tc_class").len().iter_rows()),
        "by_source_band": {
            str(k): v
            for k, v in pool.group_by((pl.col("source_rating") // band * band).alias("b"))
            .len()
            .sort("b")
            .iter_rows()
        },
        "by_phase": dict(pool.group_by("phase").len().sort("phase").iter_rows()),
    }


def _markdown(r: dict[str, Any]) -> str:
    lines = [
        "# build_pool report",
        "",
        f"- positions: {r['positions']:,} (states: {', '.join(r['position_states'])})",
        f"- by category: {json.dumps(r['by_category'])}",
        f"- by source class: {json.dumps(r['by_source_class'])}",
        f"- by phase: {json.dumps(r['by_phase'])}",
        "",
        "| source rating band | positions |",
        "|---:|---:|",
    ]
    lines += [f"| {b} | {n:,} |" for b, n in r["by_source_band"].items()]
    return "\n".join(lines) + "\n"


def run(
    observations_path: Path, labels_dir: Path, moves_dir: Path, out_dir: Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:  # fmt: skip
    obs = pl.read_parquet(observations_path)
    labels = pl.scan_parquet(labels_dir / "part-*.parquet")
    moves = pl.scan_parquet(moves_dir / "**" / "*.parquet")
    pool = build(obs, labels, moves, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(pool, out_dir / "trainer_pool.parquet")
    report = quality_report(pool, cfg)
    write_atomic_json(out_dir / "_pool_report.json", report)
    (out_dir / "_pool_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--observations",
        type=Path,
        default=settings.interim_dir / "positions" / "labeled_observations.parquet",
    )
    p.add_argument("--labels", type=Path, help="default: <observations folder>/labels")
    p.add_argument("--moves", type=Path, help="default: <observations folder>/../moves")
    p.add_argument("--out", type=Path, default=settings.processed_dir)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = args.observations.parent
    report = run(
        args.observations,
        args.labels or folder / "labels",
        args.moves or folder.parent / "moves",
        args.out,
        cfg,
    )
    log.info("%s", json.dumps({k: v for k, v in report.items() if k != "by_source_band"}))
    return 0
