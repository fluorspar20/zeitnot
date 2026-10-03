"""Typed configuration.

Two layers:
- ``Settings``: machine/environment settings (paths, engine binary) from env vars
  prefixed ``ZEITNOT_`` or a ``.env`` file at the repo root.
- ``PipelineConfig``: pipeline parameters (thresholds, sample sizes, ...) from
  ``config/pipeline.yaml``. No magic numbers in code; add them here instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/zeitnot/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

TcClass = Literal["ultrabullet", "bullet", "blitz", "rapid", "classical"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- time controls -----------------------------------------------------------


class TcClassBound(_Strict):
    name: TcClass
    max_est_duration_s: PositiveInt


class ReferenceTc(_Strict):
    base_s: PositiveInt
    increment_s: int = Field(ge=0)


class TimeControlConfig(_Strict):
    increment_weight: PositiveInt
    classes: list[TcClassBound]
    supported: list[TcClass]
    collected: list[TcClass]
    reference: dict[TcClass, ReferenceTc]

    @model_validator(mode="after")
    def _check(self) -> Self:
        bounds = [c.max_est_duration_s for c in self.classes]
        if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError("time_controls.classes must have strictly increasing bounds")
        if not set(self.supported) <= set(self.collected):
            raise ValueError("supported classes must be a subset of collected classes")
        missing = set(self.supported) - set(self.reference)
        if missing:
            raise ValueError(f"no reference time control for supported classes: {missing}")
        for name, ref in self.reference.items():
            if self.classify(ref.base_s, ref.increment_s) != name:
                raise ValueError(f"reference time control for {name} falls in another class")
        return self

    def est_duration_s(self, base_s: int, increment_s: int) -> int:
        return base_s + self.increment_weight * increment_s

    def classify(self, base_s: int, increment_s: int) -> TcClass | None:
        """Lichess speed class; None means beyond classical (correspondence)."""
        est = self.est_duration_s(base_s, increment_s)
        for c in self.classes:
            if est <= c.max_est_duration_s:
                return c.name
        return None


# --- pipeline sections -------------------------------------------------------


class ClockConfig(_Strict):
    premove_max_s: float = Field(ge=0)
    noise_tolerance_s: float = Field(ge=0)
    berserk_tolerance_s: float = Field(ge=0)
    low_clock_s: dict[TcClass, float]


class AcceptabilityConfig(_Strict):
    tau_win_pct: float = Field(gt=0, le=100)
    win_pct_k: float = Field(gt=0)
    cp_ceiling: PositiveInt


class FilterConfig(_Strict):
    min_elo: PositiveInt
    max_elo: PositiveInt
    require_clock: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_elo >= self.max_elo:
            raise ValueError("filter.min_elo must be < max_elo")
        return self


class SamplingConfig(_Strict):
    positions_per_game: PositiveInt
    min_ply: int = Field(ge=0)
    rating_band_width: PositiveInt
    decisive_win_pct: float = Field(gt=50, le=100)
    decisive_max_share: float = Field(ge=0, le=1)
    target_positions: PositiveInt


class EngineConfig(_Strict):
    multipv: PositiveInt
    nodes: PositiveInt
    threads_per_worker: PositiveInt
    hash_mb_per_worker: PositiveInt
    workers: PositiveInt


class ClipRange(_Strict):
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min >= self.max:
            raise ValueError("clip min must be < max")
        return self


class RatingGrid(_Strict):
    min: PositiveInt
    max: PositiveInt
    step: PositiveInt

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min >= self.max:
            raise ValueError("rating_grid.min must be < max")
        return self

    def values(self) -> list[int]:
        return list(range(self.min, self.max + 1, self.step))


class BudgetConfig(_Strict):
    quantile: float = Field(gt=0, lt=1)
    strict_quantile: float = Field(gt=0, lt=1)
    p_inf_min: float = Field(ge=0, le=1)
    p_inf_max: float = Field(ge=0, le=1)
    clip_s: dict[TcClass, ClipRange]
    grace_s: float = Field(ge=0)
    rating_grid: RatingGrid

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.strict_quantile > self.quantile:
            raise ValueError("budget.strict_quantile must be <= quantile")
        if self.p_inf_min >= self.p_inf_max:
            raise ValueError("budget.p_inf_min must be < p_inf_max")
        return self


class TargetConfig(_Strict):
    rating_offset: int
    source_rating_window: PositiveInt


class PipelineConfig(_Strict):
    seed: int
    time_controls: TimeControlConfig
    clock: ClockConfig
    acceptability: AcceptabilityConfig
    filter: FilterConfig
    sampling: SamplingConfig
    engine: EngineConfig
    budget: BudgetConfig
    target: TargetConfig

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(self.time_controls.supported) - set(self.budget.clip_s)
        if missing:
            raise ValueError(f"budget.clip_s missing supported classes: {missing}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


# --- environment settings ----------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZEITNOT_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = REPO_ROOT
    data_dir: Path | None = None
    pipeline_config: Path | None = None
    stockfish_path: Path | None = None

    @model_validator(mode="after")
    def _defaults(self) -> Self:
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data"
        if self.pipeline_config is None:
            self.pipeline_config = self.root_dir / "config" / "pipeline.yaml"
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def load_pipeline(self) -> PipelineConfig:
        return PipelineConfig.from_yaml(self.pipeline_config)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_pipeline_config() -> PipelineConfig:
    return get_settings().load_pipeline()
