"""sample_positions job: stratified, seeded sample of real-game decisions.

A sampled row is one decision: the position *before* ``ply``, the mover's rating
and clock, the move played and the time it took.

Design:
1. Per game, keep at most k eligible plies, chosen uniformly (limits within-game
   correlation).
2. Stratify the survivors by time-control class x mover's rating band x ply bucket
   and give every stratum the same quota; strata smaller than the quota are taken
   whole and the remainder is shared among the others ("water-filling"). This
   oversamples rare ratings on purpose.
3. Store each row's sampling weight (1 / inclusion probability) so population-level
   estimates remain possible.
4. Replay only the sampled games to get FEN/EPD, phase and any PGN evals.

Randomness comes from Python's Mersenne Twister, seeded per input part and drawn
in (game_id, ply) order, so the sample depends only on the seed and the inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import time
from pathlib import Path
from typing import Any

import chess
import polars as pl

from zeitnot.config import PhaseConfig, PipelineConfig, get_settings
from zeitnot.data.parts import write_atomic_json, write_parquet_atomic

log = logging.getLogger(__name__)

STRATUM = ["tc_class", "rating_band", "ply_bucket"]
_GAME_COLS = ["game_id", "tc_class", "base_s", "increment_s", "white_elo", "black_elo"]
_MOVE_COLS = [
    "game_id", "ply", "side", "uci", "time_spent_s", "own_clk_before_s", "opp_clk_s",
    "is_first_move", "berserk", "premove_suspect", "low_clock",
]  # fmt: skip


def allocate_quotas(sizes: dict[Any, int], target: int) -> dict[Any, int]:
    """Equal allocation with water-filling: no stratum gets more rows than it has.

    Strata are visited from smallest to largest; one that cannot fill an equal
    share of what is left is taken whole. Ties and remainders are resolved in
    sorted key order, so the result is deterministic.
    """
    quotas: dict[Any, int] = {}
    remaining = min(target, sum(sizes.values()))
    order = sorted(sizes, key=lambda k: (sizes[k], k))
    for i, key in enumerate(order):
        left = len(order) - i
        share = remaining // left
        if sizes[key] <= share:
            quotas[key] = sizes[key]
            remaining -= sizes[key]
            continue
        # Every remaining stratum is larger than the equal share: split evenly.
        rest = sorted(order[i:])
        extra = remaining - share * left
        for j, k in enumerate(rest):
            quotas[k] = share + (1 if j < extra else 0)
        break
    return quotas


def game_candidates(
    moves_path: Path, games_path: Path, month: str, index: int, cfg: PipelineConfig
) -> tuple[pl.DataFrame, int]:
    """Stage 1 for one part: up to k random eligible plies per game. Returns (rows, n_eligible)."""
    s = cfg.sampling
    games = pl.read_parquet(games_path, columns=_GAME_COLS).filter(
        pl.col("tc_class").is_in(cfg.time_controls.supported)
    )
    eligible = (
        ~pl.col("is_first_move")
        & pl.col("time_spent_s").is_not_null()
        & (pl.col("ply") >= s.min_ply)
    )
    if s.exclude_premove_suspect:
        eligible &= ~pl.col("premove_suspect")
    if s.exclude_berserk:
        eligible &= ~pl.col("berserk")
    white = pl.col("side") == "w"
    df = (
        pl.read_parquet(moves_path, columns=_MOVE_COLS)
        .filter(eligible)
        .join(games, on="game_id", how="inner")
        .sort("game_id", "ply")
    )
    rng = random.Random(f"{cfg.seed}:{month}:{index}")
    df = df.with_columns(
        pl.Series("u", [rng.random() for _ in range(df.height)], dtype=pl.Float64),
        pl.lit(month).alias("month"),
        pl.when(white).then("white_elo").otherwise("black_elo").alias("mover_elo"),
        pl.when(white).then("black_elo").otherwise("white_elo").alias("opp_elo"),
        pl.len().over("game_id").alias("game_eligible_plies"),
    )
    n_eligible = df.height
    picked = df.filter(
        pl.col("u").rank("ordinal").over("game_id") <= s.positions_per_game
    ).with_columns(pl.len().over("game_id").alias("game_sampled_plies"))
    # Stage 2 needs its own random numbers: the survivors' u values are per-game minima,
    # smaller in games with more eligible plies, so reusing them would favor long games.
    picked = picked.with_columns(
        pl.Series("u", [rng.random() for _ in range(picked.height)], dtype=pl.Float64)
    )
    return picked.drop("white_elo", "black_elo", "is_first_move", "berserk", "premove_suspect"), (
        n_eligible
    )


def stratified_sample(candidates: pl.DataFrame, cfg: PipelineConfig, target: int) -> pl.DataFrame:
    """Stage 2: equal-allocation sample over strata, with sampling weights."""
    s = cfg.sampling
    lower_bounds = [s.min_ply, *s.ply_bucket_edges]
    bucket_index = pl.sum_horizontal([pl.col("ply") >= e for e in s.ply_bucket_edges])
    df = candidates.with_columns(
        (pl.col("mover_elo") // s.rating_band_width * s.rating_band_width)
        .cast(pl.Int32)
        .alias("rating_band"),
        bucket_index.replace_strict(
            list(range(len(lower_bounds))), lower_bounds, return_dtype=pl.Int32
        ).alias("ply_bucket"),
    )
    sizes = {tuple(r[:3]): r[3] for r in df.group_by(STRATUM).len().sort(STRATUM).iter_rows()}
    quotas = allocate_quotas(sizes, target)
    quota_df = pl.DataFrame(
        [(*k, sizes[k], q) for k, q in quotas.items()],
        schema=[*[(c, df.schema[c]) for c in STRATUM], ("stratum_candidates", pl.Int64),
                ("stratum_sampled", pl.Int64)],
        orient="row",
    )  # fmt: skip
    return (
        df.join(quota_df, on=STRATUM)
        .filter(pl.col("u").rank("ordinal").over(STRATUM) <= pl.col("stratum_sampled"))
        .with_columns(
            (
                pl.col("game_eligible_plies")
                / pl.col("game_sampled_plies")
                * pl.col("stratum_candidates")
                / pl.col("stratum_sampled")
            ).alias("sampling_weight")
        )
        .drop("u")
    )


def position_phase(board: chess.Board, ply: int, cfg: PhaseConfig) -> str:
    pieces = chess.popcount(board.occupied & ~board.pawns & ~board.kings)
    if pieces <= cfg.endgame_max_pieces:
        return "endgame"
    if pieces >= cfg.opening_min_pieces and ply < cfg.opening_max_ply:
        return "opening"
    return "middlegame"


def position_id(epd: str) -> str:
    """Stable id of a position (pieces, side, castling, legal en passant)."""
    return hashlib.blake2b(epd.encode("ascii"), digest_size=8).hexdigest()


def replay_positions(
    sampled: pl.DataFrame, interim: Path, month: str, cfg: PipelineConfig
) -> pl.DataFrame:
    """Stage 3: FEN/EPD, phase and PGN evals for the sampled plies of one month."""
    wanted = sampled.filter(pl.col("month") == month).select("game_id", "ply")
    needed: dict[str, set[int]] = {}
    for game_id, ply in wanted.iter_rows():
        needed.setdefault(game_id, set()).add(ply)
    games = (
        pl.scan_parquet(interim / "moves" / f"month={month}" / "part-*.parquet")
        .select("game_id", "ply", "uci", "eval_white_cp", "eval_white_mate")
        .join(wanted.select("game_id").unique().lazy(), on="game_id", how="semi")
        .sort("game_id", "ply")
        .group_by("game_id", maintain_order=True)
        .agg("uci", "eval_white_cp", "eval_white_mate")
        .collect()
    )
    rows = []
    for game_id, ucis, cps, mates in games.iter_rows():
        plies = needed[game_id]
        last = max(plies)
        board = chess.Board()
        for ply, uci in enumerate(ucis, start=1):
            if ply in plies:
                epd = board.epd(en_passant="legal")
                rows.append(
                    (
                        game_id, ply, position_id(epd), board.fen(), epd,
                        position_phase(board, ply, cfg.sampling.phase),
                        cps[ply - 2] if ply >= 2 else None,
                        mates[ply - 2] if ply >= 2 else None,
                        cps[ply - 1], mates[ply - 1],
                    )
                )  # fmt: skip
            if ply == last:
                break
            board.push(chess.Move.from_uci(uci))
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.String, "ply": pl.Int16, "position_id": pl.String, "fen": pl.String,
            "epd": pl.String, "phase": pl.String,
            "eval_before_white_cp": pl.Int16, "eval_before_white_mate": pl.Int16,
            "eval_after_white_cp": pl.Int16, "eval_after_white_mate": pl.Int16,
        },
        orient="row",
    )  # fmt: skip


def run(
    interim: Path, months: list[str], out_dir: Path, cfg: PipelineConfig, target: int
) -> dict[str, Any]:
    t0 = time.perf_counter()
    parts, n_eligible = [], 0
    for month in months:
        moves_dir = interim / "moves" / f"month={month}"
        for moves_path in sorted(moves_dir.glob("part-*.parquet")):
            index = int(moves_path.stem.split("-")[1])
            games_path = interim / "games" / f"month={month}" / moves_path.name
            picked, n = game_candidates(moves_path, games_path, month, index, cfg)
            parts.append(picked)
            n_eligible += n
        log.info("%s: candidates from %d parts so far", month, len(parts))
    if not parts:
        raise SystemExit(f"no moves parts for months {months} under {interim}")
    candidates = pl.concat(parts)
    sampled = stratified_sample(candidates, cfg, target)
    log.info(
        "sampled %s of %s candidates; replaying", f"{sampled.height:,}", f"{candidates.height:,}"
    )

    replayed = pl.concat([replay_positions(sampled, interim, m, cfg) for m in months])
    out = (
        sampled.join(replayed, on=["game_id", "ply"], how="inner")
        .rename({"uci": "played_uci"})
        .with_columns(((pl.col("ply") + 1) // 2).cast(pl.Int16).alias("move_number"))
        .select(
            "position_id",
            "fen",
            "epd",
            "game_id",
            "month",
            "ply",
            "move_number",
            "side",
            "phase",
            "tc_class",
            "base_s",
            "increment_s",
            "mover_elo",
            "opp_elo",
            "played_uci",
            "time_spent_s",
            "own_clk_before_s",
            "opp_clk_s",
            "low_clock",
            "eval_before_white_cp",
            "eval_before_white_mate",
            "eval_after_white_cp",
            "eval_after_white_mate",
            "rating_band",
            "ply_bucket",
            "stratum_candidates",
            "stratum_sampled",
            "game_eligible_plies",
            "game_sampled_plies",
            "sampling_weight",
        )
        .sort("month", "game_id", "ply")
    )
    assert out.height == sampled.height, "replay lost sampled rows"

    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(out, out_dir / "sampled_positions.parquet")
    write_atomic_json(
        out_dir / "_manifest.json",
        {
            "seed": cfg.seed,
            "months": months,
            "target": target,
            "sampling": cfg.sampling.model_dump(),
            "supported_classes": cfg.time_controls.supported,
            "python": platform.python_version(),
            "polars": pl.__version__,
        },
    )
    report = _report(out, candidates.height, n_eligible, time.perf_counter() - t0)
    write_atomic_json(out_dir / "_report.json", report)
    (out_dir / "_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _report(
    out: pl.DataFrame, n_candidates: int, n_eligible: int, seconds: float
) -> dict[str, Any]:
    has_eval = (
        pl.col("eval_before_white_cp").is_not_null()
        | pl.col("eval_before_white_mate").is_not_null()
    ) & (
        pl.col("eval_after_white_cp").is_not_null() | pl.col("eval_after_white_mate").is_not_null()
    )
    by_band = (
        out.group_by(STRATUM)
        .agg(
            pl.len().alias("sampled"),
            pl.col("stratum_candidates").first().alias("candidates"),
            pl.col("sampling_weight").sum().alias("weight"),
        )
        .group_by("tc_class", "rating_band")
        .agg(
            pl.col("candidates").sum(),
            pl.col("sampled").sum(),
            (pl.col("weight").sum() / pl.col("sampled").sum()).round(1).alias("mean_weight"),
        )
        .sort("tc_class", "rating_band")
    )
    return {
        "eligible_rows": n_eligible,
        "candidates_after_per_game_cap": n_candidates,
        "sampled": out.height,
        "games": out["game_id"].n_unique(),
        "distinct_positions": out["position_id"].n_unique(),
        "strata": out.select(pl.struct(STRATUM).n_unique()).item(),
        "largest_stratum_quota": out["stratum_sampled"].max(),
        "share_with_pgn_eval": round(out.select(has_eval.mean()).item(), 4),
        "weight_sum": round(out["sampling_weight"].sum()),
        "by_phase": dict(out["phase"].value_counts().sort("phase").iter_rows()),
        "by_class": dict(out["tc_class"].value_counts().sort("tc_class").iter_rows()),
        "by_ply_bucket": dict(out["ply_bucket"].value_counts().sort("ply_bucket").iter_rows()),
        "by_class_band": by_band.to_dicts(),
        "seconds": round(seconds, 1),
    }  # fmt: skip


def _markdown(r: dict[str, Any]) -> str:
    lines = [
        "# sample_positions report",
        "",
        f"- eligible rows: {r['eligible_rows']:,} -> after per-game cap: "
        f"{r['candidates_after_per_game_cap']:,} -> sampled: {r['sampled']:,}",
        f"- games: {r['games']:,}; distinct positions: {r['distinct_positions']:,}; "
        f"strata: {r['strata']}; largest stratum quota: {r['largest_stratum_quota']:,}",
        f"- with PGN eval before and after the move: {r['share_with_pgn_eval']:.1%}",
        f"- sum of weights (estimates eligible rows): {r['weight_sum']:,}",
        f"- by phase: {json.dumps(r['by_phase'])}; by class: {json.dumps(r['by_class'])}",
        f"- by ply bucket: {json.dumps(r['by_ply_bucket'])}",
        "",
        "| class | rating band | candidates | sampled | mean weight |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {b['tc_class']} | {b['rating_band']} | {b['candidates']:,} | {b['sampled']:,} | "
        f"{b['mean_weight']} |"
        for b in r["by_class_band"]
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    cfg = settings.load_pipeline()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("months", nargs="+", help="YYYY-MM partitions of games/moves to sample from")
    p.add_argument("--interim", type=Path, default=settings.interim_dir, help="games/moves root")
    p.add_argument("--out", type=Path, help="default: <interim>/positions")
    p.add_argument("--target", type=int, default=cfg.sampling.target_positions)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out or args.interim / "positions"
    report = run(args.interim, sorted(args.months), out, cfg, args.target)
    log.info(
        "sampled %s positions from %s games in %.0f s; report in %s",
        f"{report['sampled']:,}", f"{report['games']:,}", report["seconds"], out,
    )  # fmt: skip
    return 0
