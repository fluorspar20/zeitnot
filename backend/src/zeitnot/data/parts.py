"""Helpers for pipeline steps that write one output part per work unit.

Each part has a Parquet file and a ``.stats.json`` sidecar. Both are written
atomically (temp file + rename) and the sidecar is written last, so a part is
either complete or absent; a rerun skips parts whose sidecar exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def part_paths(out_dir: Path, index: int) -> tuple[Path, Path]:
    stem = out_dir / f"part-{index:06d}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".stats.json")


def write_atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def read_stats(out_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("part-*.stats.json"))
    ]
