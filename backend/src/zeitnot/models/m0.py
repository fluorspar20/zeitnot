"""M0: heuristic time budgets from group statistics (the baseline every model must beat).

For each time-control class, position category and rating on the grid, take the
decisions made by players near that rating on positions of that category, and
compute:

- ``budget_s``: the time by which a share q of players would have found an acceptable
  move (``budget_strict_s`` for the stricter share), from a Kaplan-Meier estimate of
  the time-to-find distribution;
- ``p_inf``: the share of players who found an acceptable move (an empirical solve
  rate within the time they chose to spend, not a true eventual-success probability).

Why Kaplan-Meier: a player who thought for T seconds and then played a bad move had
not found a good one within T. Dropping such players and taking the median time of
those who succeeded describes the quick finders only; the hard positions, where
players search long and fail, would barely move the budget. Kaplan-Meier treats each
failure as right-censored at T instead. It assumes that players who stopped searching
would have gone on to find a move about as fast as those who continued (non-informative
censoring), which is optimistic; Trainer attempts are the real check.

Details:

- Only the kind of positions the Trainer serves (trainable categories, configured
  position states) are used, and moves on a nearly empty clock can be excluded.
- Clocks record whole seconds, so a recorded T stands for any time in
  [T - 0.5, T + 0.5]. The estimated distribution function is interpolated linearly
  across that interval, which gives smooth values instead of steps at whole seconds.
- Rows carry sampling weights (the sample over-represents rare ratings), so all
  statistics are weighted.
- If the curve never reaches q (too many failures), the budget is the class maximum.
- The rating window around a grid rating widens until it holds enough acceptable
  moves; sparse top ratings borrow from their neighbours.
- Shape constraints: budgets never increase with rating and solve rates never
  decrease (isotonic regression, weighted by the number of acceptable moves); a
  rapid budget is never shorter than the blitz one; budgets are clipped per class.

Every position of a category gets the same budget at a given rating: M0 knows
nothing about the individual position. That is the job of the learned models.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from zeitnot.config import PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic
from zeitnot.positions.categories import TRAINABLE

log = logging.getLogger(__name__)

CLOCK_RESOLUTION_S = 1.0  # Lichess %clk has whole seconds
REPORT_RATINGS = [800, 1200, 1600, 2000, 2400, 2800]


def km_quantile(
    times: Sequence[float], found: Sequence[bool], weights: Sequence[float], q: float
) -> float | None:
    """Time by which a share q has found an acceptable move (weighted Kaplan-Meier).

    ``found`` False means the player moved at that time without finding one: censored.
    At tied times, events are counted before censorings (the usual convention). The
    step at a recorded time t is spread linearly over its rounding interval
    [t - 0.5, t + 0.5]. Returns None if the estimated curve never reaches q.
    """
    if not times:
        raise ValueError("no observations")
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    half = CLOCK_RESOLUTION_S / 2
    pooled: dict[float, list[float]] = {}  # time -> [weight of finds, total weight]
    for t, f, w in zip(times, found, weights, strict=True):
        cell = pooled.setdefault(t, [0.0, 0.0])
        cell[0] += w if f else 0.0
        cell[1] += w
    at_risk = sum(total for _, total in pooled.values())
    survival = 1.0  # share that has not found an acceptable move yet
    for t in sorted(pooled):
        events, total = pooled[t]
        if events > 0:
            before = 1.0 - survival
            survival *= 1.0 - events / at_risk
            after = 1.0 - survival
            if after >= q:
                lo = max(t - half, 0.0)
                return lo + (t + half - lo) * (q - before) / (after - before)
        at_risk -= total
    return None


def isotonic(values: Sequence[float], weights: Sequence[float], increasing: bool) -> list[float]:
    """Weighted least-squares monotone fit (pool-adjacent-violators algorithm).

    Walk left to right; whenever a new value breaks the required order, merge it with
    the previous block into their weighted mean, and keep merging backwards until the
    order holds again.
    """
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # [mean, weight, count], mean in the sign-flipped space
    for v, w in zip(values, weights, strict=True):
        blocks.append([sign * v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out: list[float] = []
    for mean, _, count in blocks:
        out.extend([sign * mean] * int(count))
    return out


def training_rows(obs: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Decisions on the kind of positions the Trainer serves."""
    keep = (
        pl.col("loss_win_pct").is_not_null()
        & pl.col("tc_class").is_in(cfg.time_controls.supported)
        & pl.col("category").cast(pl.String).is_in(TRAINABLE)
        & pl.col("position_state").cast(pl.String).is_in(cfg.trainer.position_states)
    )
    if cfg.budget.m0.exclude_low_clock:
        keep &= ~pl.col("low_clock")
    return obs.filter(keep).select(
        "tc_class",
        pl.col("category").cast(pl.String),
        "mover_elo",
        "time_spent_s",
        "y",
        "sampling_weight",
    )


def _window_stats(rows: pl.DataFrame, rating: int, cfg: PipelineConfig) -> dict[str, Any]:
    """Statistics around one grid rating; ``rows`` is one class x category, sorted by rating."""
    m0 = cfg.budget.m0
    elo = rows["mover_elo"].to_list()
    half = m0.window_step
    while True:
        lo = bisect.bisect_left(elo, rating - half)
        hi = bisect.bisect_right(elo, rating + half)
        win = rows.slice(lo, hi - lo)
        found = win.filter(pl.col("y"))
        if found.height >= m0.min_successes or half >= m0.max_half_width:
            break
        half += m0.window_step
    out: dict[str, Any] = {
        "half_width": half,
        "n_obs": win.height,
        "n_success": found.height,
        "raw_budget_s": None,
        "raw_budget_strict_s": None,
        "raw_p_inf": None,
        "unreached": False,
    }
    if found.height:
        t, y = win["time_spent_s"].to_list(), win["y"].to_list()
        w = win["sampling_weight"].to_list()
        unreached = cfg.budget.clip_s[rows["tc_class"][0]].max
        for col, q in (
            ("raw_budget_s", cfg.budget.quantile),
            ("raw_budget_strict_s", cfg.budget.strict_quantile),
        ):
            value = km_quantile(t, y, w, q)
            out[col] = unreached if value is None else value
            out["unreached"] |= value is None
        weight = win["sampling_weight"]
        out["raw_p_inf"] = float((win["y"].cast(pl.Float64) * weight).sum() / weight.sum())
    return out


def _constrain(group: list[dict[str, Any]]) -> None:
    """Isotonic fits along the rating grid for one class x category (in place)."""
    ok = [r for r in group if r["n_success"]]
    w = [r["n_success"] for r in ok]
    for raw, col in (("raw_budget_s", "budget_s"), ("raw_budget_strict_s", "budget_strict_s")):
        # Fit on the log scale: times are skewed and ratios matter more than differences.
        fitted = isotonic([math.log(r[raw]) for r in ok], w, increasing=False)
        for r, f in zip(ok, fitted, strict=True):
            r[col] = math.exp(f)
    fitted = isotonic([r["raw_p_inf"] for r in ok], w, increasing=True)
    for r, f in zip(ok, fitted, strict=True):
        r["p_inf"] = f


def fit(obs: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Budget table: one row per class x category x grid rating."""
    rows = training_rows(obs, cfg).sort("mover_elo")
    grid = cfg.budget.rating_grid.values()
    out: list[dict[str, Any]] = []
    for tc in cfg.time_controls.supported:
        for cat in TRAINABLE:
            sub = rows.filter((pl.col("tc_class") == tc) & (pl.col("category") == cat))
            group = [
                {"tc_class": tc, "category": cat, "rating": g, **_window_stats(sub, g, cfg)}
                for g in grid
            ]
            _constrain(group)
            out.extend(group)
    table = pl.DataFrame(out, infer_schema_length=None)
    return _cross_class(table, cfg)


def _cross_class(table: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Longer time controls never get a shorter budget; then clip per class."""
    classes = cfg.time_controls.supported  # ordered from fastest to slowest
    keys = ["category", "rating"]
    for col in ("budget_s", "budget_strict_s"):
        running = None
        parts = []
        for tc in classes:
            part = table.filter(pl.col("tc_class") == tc)
            if running is not None:
                part = (
                    part.join(running, on=keys, how="left")
                    .with_columns(
                        pl.max_horizontal(col, "_prev").alias(col),
                        (pl.col(col) < pl.col("_prev")).fill_null(False).alias(f"_raised_{col}"),
                    )
                    .drop("_prev")
                )
            else:
                part = part.with_columns(pl.lit(False).alias(f"_raised_{col}"))
            running = part.select(*keys, pl.col(col).alias("_prev"))
            parts.append(part)
        table = pl.concat(parts)
    clip = cfg.budget.clip_s
    lo = pl.col("tc_class").replace_strict({c: clip[c].min for c in classes})
    hi = pl.col("tc_class").replace_strict({c: clip[c].max for c in classes})
    return table.with_columns(
        (pl.col("n_success") < cfg.budget.m0.min_successes).alias("sparse"),
        (pl.col("_raised_budget_s") | pl.col("_raised_budget_strict_s")).alias("raised_to_faster"),
        (pl.col("budget_s") < lo).alias("clipped_low"),
        (pl.col("budget_s") > hi).alias("clipped_high"),
        pl.col("budget_s").clip(lo, hi),
        pl.col("budget_strict_s").clip(lo, hi),
    ).drop("_raised_budget_s", "_raised_budget_strict_s")


def check(table: pl.DataFrame, cfg: PipelineConfig) -> list[str]:
    """Shape constraints the budget table must satisfy; returns the violations."""
    problems = []
    eps = 1e-9
    for (tc, cat), g in table.sort("rating").group_by("tc_class", "category"):
        b, s, p = g["budget_s"].to_list(), g["budget_strict_s"].to_list(), g["p_inf"].to_list()
        if any(v is None for v in b + s + p):
            problems.append(f"{tc}/{cat}: missing values")
            continue
        if any(b2 > b1 + eps for b1, b2 in pairwise(b)):
            problems.append(f"{tc}/{cat}: budget increases with rating")
        if any(p2 < p1 - eps for p1, p2 in pairwise(p)):
            problems.append(f"{tc}/{cat}: solve rate decreases with rating")
        if any(x > y + eps for x, y in zip(s, b, strict=True)):
            problems.append(f"{tc}/{cat}: strict budget above the regular one")
    classes = cfg.time_controls.supported
    wide = table.pivot(on="tc_class", index=["category", "rating"], values="budget_s")
    for fast, slow in pairwise(classes):
        bad = wide.filter(pl.col(fast) > pl.col(slow) + eps).height
        if bad:
            problems.append(f"{bad} rows with {fast} budget above {slow}")
    return problems


def quality_report(train: pl.DataFrame, table: pl.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    fitted = table.filter(pl.col("raw_budget_s").is_not_null())
    log_shift = (pl.col("budget_s").log() - pl.col("raw_budget_s").log()).abs()
    return {
        "training_rows": train.height,
        "training_rows_by_class": dict(train.group_by("tc_class").len().iter_rows()),
        "rows": table.height,
        "violations": check(table, cfg),
        "windows_widened": fitted.filter(
            pl.col("half_width") > cfg.budget.m0.window_step
        ).height,
        "sparse_rows": table["sparse"].sum(),
        "adjustment_median_abs_log": round(fitted.select(log_shift.median()).item(), 4),
        "adjustment_max_abs_log": round(fitted.select(log_shift.max()).item(), 4),
        "unreached": table["unreached"].sum(),
        "raised_to_faster_class": table["raised_to_faster"].sum(),
        "clipped_low": table["clipped_low"].sum(),
        "clipped_high": table["clipped_high"].sum(),
        "at_report_ratings": table.filter(
            pl.col("rating").is_in(REPORT_RATINGS) & pl.col("budget_s").is_not_null()
        )
        .select(
            "tc_class", "category", "rating", "half_width", "n_success",
            pl.col("budget_s").round(1), pl.col("budget_strict_s").round(1),
            pl.col("p_inf").round(3),
        )
        .sort("tc_class", "category", "rating")
        .to_dicts(),
    }  # fmt: skip


def _markdown(r: dict[str, Any], cfg: PipelineConfig) -> str:
    q, qs = cfg.budget.quantile, cfg.budget.strict_quantile
    lines = [
        "# M0 budgets report",
        "",
        f"- training decisions: {r['training_rows']:,} ({r['training_rows_by_class']})",
        f"- table rows: {r['rows']:,}; shape violations: {len(r['violations'])}"
        + "".join(f"\n  - {v}" for v in r["violations"]),
        f"- rating windows widened to reach {cfg.budget.m0.min_successes} acceptable moves: "
        f"{r['windows_widened']}; still short at the widest (flagged sparse): {r['sparse_rows']}",
        "- budget change from shape constraints and clipping (|log ratio|): "
        f"median {r['adjustment_median_abs_log']}, max {r['adjustment_max_abs_log']}",
        f"- curve never reached the quantile (budget = class maximum): {r['unreached']}",
        f"- rows raised to the faster class's budget: {r['raised_to_faster_class']}; "
        f"clipped low / high: {r['clipped_low']} / {r['clipped_high']}",
        "",
        f"Budget = time by which a share {q:g} (strict {qs:g}) would have found an acceptable "
        "move (Kaplan-Meier); solve = observed share of acceptable moves.",
        "",
        "| class | category | rating | window +/- | acceptable moves | budget s | strict s "
        "| solve |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {a['tc_class']} | {a['category']} | {a['rating']} | {a['half_width']} | "
        f"{a['n_success']:,} | {a['budget_s']} | {a['budget_strict_s']} | {a['p_inf']:.1%} |"
        for a in r["at_report_ratings"]
    ]
    return "\n".join(lines) + "\n"


def run(observations_path: Path, out_dir: Path, cfg: PipelineConfig) -> dict[str, Any]:
    obs = pl.read_parquet(observations_path)
    table = fit(obs, cfg)
    report = quality_report(training_rows(obs, cfg), table, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(table, out_dir / "budgets_m0.parquet")
    write_atomic_json(out_dir / "_budgets_m0_report.json", report)
    (out_dir / "_budgets_m0_report.md").write_text(_markdown(report, cfg), encoding="utf-8")
    manifest = {
        "model": "m0",
        "observations": str(observations_path),
        "budget": cfg.budget.model_dump(),
        "trainer": cfg.trainer.model_dump(),
        "tau_win_pct": cfg.acceptability.tau_win_pct,
    }
    write_atomic_json(out_dir / "_budgets_m0_manifest.json", manifest)
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
    p.add_argument("--out", type=Path, default=settings.processed_dir)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.observations, args.out, cfg)
    log.info("%s", json.dumps({k: v for k, v in report.items() if k != "at_report_ratings"}))
    return 1 if report["violations"] else 0
