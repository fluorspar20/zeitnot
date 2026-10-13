"""build_observations job: join sampled decisions with their engine labels.

One row per sampled decision (a player, a position, the move played, the time taken)
with the label of that move: its win% loss against the engine's best move and whether
it is acceptable (Y). Position-level difficulty signals from the label are carried
along: how many lines are acceptable, the gaps between the top lines, and the depth at
which the best move surfaced.

The raw loss is always stored, so Y can be recomputed for a different threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from zeitnot.config import PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic

log = logging.getLogger(__name__)

STATES = ["losing", "balanced", "winning"]


def position_features(labels: pl.LazyFrame, tau: float) -> pl.LazyFrame:
    """Per-position columns derived from the multi-PV lines."""
    wins = pl.col("lines").list.eval(pl.element().struct.field("win_pct"))
    return labels.select(
        "position_id",
        "best_uci",
        "best_win_pct",
        "depth",
        "best_move_first_depth",
        "best_move_stable_depth",
        pl.col("lines").list.len().cast(pl.Int8).alias("n_lines"),
        # Lines within tau of the best one (the best line itself always counts).
        wins.list.eval(pl.element().first() - pl.element() <= tau)
        .list.sum()
        .cast(pl.Int8)
        .alias("n_acceptable_lines"),
        (wins.list.get(0) - wins.list.get(1, null_on_oob=True)).alias("gap_1_2"),
        (wins.list.get(0) - wins.list.get(2, null_on_oob=True)).alias("gap_1_3"),
    )


def build(positions: pl.DataFrame, labels: pl.LazyFrame, cfg: PipelineConfig) -> pl.DataFrame:
    tau = cfg.acceptability.tau_win_pct
    decisive = cfg.sampling.decisive_win_pct
    played = (
        labels.select("position_id", "played")
        .explode("played", empty_as_null=True)
        .unnest("played")
        .select(
            "position_id",
            pl.col("uci").alias("played_uci"),
            pl.col("win_pct").alias("played_win_pct"),
            pl.col("in_multipv").alias("played_in_multipv"),
            "loss_win_pct",
        )
    )
    out = (
        positions.lazy()
        .join(position_features(labels, tau), on="position_id", how="left")
        .join(played, on=["position_id", "played_uci"], how="left")
        .with_columns(
            (pl.col("loss_win_pct") <= tau).alias("y"),
            (pl.col("played_uci") == pl.col("best_uci")).alias("is_best"),
            # Whose game it already is, from the mover's side: when it is decided,
            # nearly every move keeps the win% and counts as acceptable.
            pl.when(pl.col("best_win_pct") >= decisive)
            .then(pl.lit("winning"))
            .when(pl.col("best_win_pct") <= 100 - decisive)
            .then(pl.lit("losing"))
            .otherwise(pl.lit("balanced"))
            .cast(pl.Enum(STATES))
            .alias("position_state"),
        )
        .collect()
    )
    return out


def _weighted_rate(col: str) -> pl.Expr:
    w = pl.col("sampling_weight")
    return ((pl.col(col).cast(pl.Float64) * w).sum() / w.sum()).round(4)


def quality_report(
    obs: pl.DataFrame, evaldb: pl.DataFrame | None, cfg: PipelineConfig
) -> dict[str, Any]:
    tau = cfg.acceptability.tau_win_pct
    band = cfg.puzzles.report_band_width
    labeled = obs.filter(pl.col("loss_win_pct").is_not_null())
    by_band_state = (
        labeled.with_columns((pl.col("mover_elo") // band * band).alias("band"))
        .group_by("tc_class", "band", "position_state")
        .agg(
            pl.len().alias("n"),
            pl.col("y").mean().round(4).alias("y_rate"),
            pl.col("is_best").mean().round(4).alias("best_rate"),
            pl.col("loss_win_pct")
            .cast(pl.Float64)
            .clip(lower_bound=0)
            .mean()
            .round(2)
            .alias("mean_loss"),
            pl.col("time_spent_s").median().alias("median_t"),
        )
        .sort("tc_class", "band", "position_state")
    )
    state_share = (
        labeled.group_by("position_state")
        .agg(pl.len().alias("n"), _weighted_rate("y").alias("y_rate_weighted"))
        .with_columns((pl.col("n") / labeled.height).round(4).alias("share_of_sample"))
        .sort("position_state")
    )
    report: dict[str, Any] = {
        "tau_win_pct": tau,
        "observations": obs.height,
        "labeled": labeled.height,
        "y_rate_sample": round(labeled["y"].mean(), 4),  # type: ignore[arg-type]
        "y_rate_population": labeled.select(_weighted_rate("y")).item(),
        "best_move_rate_population": labeled.select(_weighted_rate("is_best")).item(),
        "played_in_multipv": round(labeled["played_in_multipv"].mean(), 4),  # type: ignore[arg-type]
        "negative_loss_share": round((labeled["loss_win_pct"] < 0).mean(), 4),  # type: ignore[arg-type]
        "by_state": state_share.to_dicts(),
        "by_class_band_state": by_band_state.to_dicts(),
    }
    if evaldb is not None:
        report["evaldb_check"] = _evaldb_check(labeled, evaldb, tau)
    return report


def _evaldb_check(labeled: pl.DataFrame, evaldb: pl.DataFrame, tau: float) -> dict[str, Any]:
    """Agreement with far deeper Lichess evals, where they contain the played move."""
    db = (
        evaldb.select("epd", pl.col("best_uci").alias("db_best_uci"), "best_knodes", "pvs")
        .explode("pvs", empty_as_null=True)
        .unnest("pvs")
        .with_columns((pl.col("win_pct").max().over("epd") - pl.col("win_pct")).alias("db_loss"))
        .select("epd", pl.col("uci").alias("played_uci"), "db_best_uci", "best_knodes", "db_loss")
    )
    j = labeled.join(db, on=["epd", "played_uci"], how="inner")
    if j.is_empty():
        return {"n": 0}
    return {
        "n": j.height,
        "db_median_knodes": j["best_knodes"].median(),
        "same_y": round(((j["loss_win_pct"] <= tau) == (j["db_loss"] <= tau)).mean(), 4),  # type: ignore[arg-type]
        "same_best_move": round((j["best_uci"] == j["db_best_uci"]).mean(), 4),  # type: ignore[arg-type]
        "mean_abs_loss_diff": round((j["loss_win_pct"] - j["db_loss"]).abs().mean(), 3),  # type: ignore[arg-type]
    }


def _markdown(r: dict[str, Any]) -> str:
    lines = [
        "# build_observations report",
        "",
        f"- observations: {r['observations']:,}; labeled: {r['labeled']:,}; "
        f"tau = {r['tau_win_pct']}",
        f"- acceptable (Y): {r['y_rate_sample']:.1%} of the sample, "
        f"{r['y_rate_population']:.1%} population-weighted",
        f"- engine's best move played: {r['best_move_rate_population']:.1%} (weighted)",
        f"- played move inside the multi-PV lines: {r['played_in_multipv']:.1%}; "
        f"negative loss (search noise): {r['negative_loss_share']:.1%}",
    ]
    if r.get("evaldb_check", {}).get("n"):
        c = r["evaldb_check"]
        lines.append(
            f"- vs eval DB ({c['n']:,} played moves, median {c['db_median_knodes']:,.0f} knodes): "
            f"same Y {c['same_y']:.1%}, same best move {c['same_best_move']:.1%}, "
            f"mean |loss diff| {c['mean_abs_loss_diff']}"
        )
    lines += ["", "| position state | n | share of sample | Y (weighted) |", "|---|---:|---:|---:|"]
    lines += [
        f"| {s['position_state']} | {s['n']:,} | {s['share_of_sample']:.1%} | "
        f"{s['y_rate_weighted']:.1%} |"
        for s in r["by_state"]
    ]
    lines += [
        "",
        "## Balanced positions by class and rating band",
        "",
        "| class | band | n | Y | best move | mean loss | median T (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {b['tc_class']} | {b['band']} | {b['n']:,} | {b['y_rate']:.1%} | "
        f"{b['best_rate']:.1%} | {b['mean_loss']} | {b['median_t']:.0f} |"
        for b in r["by_class_band_state"]
        if b["position_state"] == "balanced"
    ]
    return "\n".join(lines) + "\n"


def run(
    positions_path: Path, labels_dir: Path, out_dir: Path, cfg: PipelineConfig,
    evaldb_path: Path | None = None,
) -> dict[str, Any]:  # fmt: skip
    positions = pl.read_parquet(positions_path)
    labels = pl.scan_parquet(labels_dir / "part-*.parquet")
    obs = build(positions, labels, cfg)
    assert obs.height == positions.height, "join changed the number of observations"
    write_parquet_atomic(obs, out_dir / "labeled_observations.parquet")
    evaldb = pl.read_parquet(evaldb_path) if evaldb_path and evaldb_path.exists() else None
    report = quality_report(obs, evaldb, cfg)
    write_atomic_json(out_dir / "_observations_report.json", report)
    (out_dir / "_observations_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    positions_dir = settings.interim_dir / "positions"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--positions", type=Path, default=positions_dir / "sampled_positions.parquet")
    p.add_argument("--labels", type=Path, help="default: <positions folder>/labels")
    p.add_argument("--evaldb", type=Path, help="default: <positions folder>/evaldb_matches.parquet")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = args.positions.parent
    report = run(
        args.positions,
        args.labels or folder / "labels",
        folder,
        cfg,
        args.evaldb or folder / "evaldb_matches.parquet",
    )
    log.info("%s", json.dumps({k: v for k, v in report.items() if not k.startswith("by_")}))
    return 0
