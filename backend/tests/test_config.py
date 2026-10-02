from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from zeitnot.config import REPO_ROOT, PipelineConfig, Settings

PIPELINE_YAML = REPO_ROOT / "config" / "pipeline.yaml"


@pytest.fixture
def raw() -> dict:
    with PIPELINE_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_repo_pipeline_yaml_loads() -> None:
    cfg = PipelineConfig.from_yaml(PIPELINE_YAML)
    assert cfg.acceptability.tau_win_pct == 5.0
    assert cfg.time_controls.supported == ["blitz", "rapid"]


@pytest.mark.parametrize(
    ("base_s", "inc_s", "expected"),
    [
        (15, 0, "ultrabullet"),
        (60, 0, "bullet"),
        (120, 1, "bullet"),  # 160
        (180, 0, "blitz"),
        (180, 2, "blitz"),  # 260
        (300, 3, "blitz"),  # 420
        (300, 5, "rapid"),  # 500
        (600, 0, "rapid"),
        (900, 10, "rapid"),  # 1300
        (1800, 0, "classical"),
        (86400, 0, None),
    ],
)
def test_classify_time_control(base_s: int, inc_s: int, expected: str | None) -> None:
    tc = PipelineConfig.from_yaml(PIPELINE_YAML).time_controls
    assert tc.classify(base_s, inc_s) == expected


def test_rating_grid_values() -> None:
    grid = PipelineConfig.from_yaml(PIPELINE_YAML).budget.rating_grid
    values = grid.values()
    assert values[0] == grid.min and values[-1] == grid.max
    assert all(b - a == grid.step for a, b in pairwise(values))


def test_unknown_key_rejected(raw: dict) -> None:
    raw["acceptability"]["tua_win_pct"] = 5.0  # typo must not be silently ignored
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_reference_tc_must_match_class(raw: dict) -> None:
    raw["time_controls"]["reference"]["blitz"] = {"base_s": 600, "increment_s": 0}
    with pytest.raises(ValidationError, match="falls in another class"):
        PipelineConfig.model_validate(raw)


def test_strict_quantile_not_above_default(raw: dict) -> None:
    raw["budget"]["strict_quantile"] = 0.7
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_supported_class_needs_clip_range(raw: dict) -> None:
    del raw["budget"]["clip_s"]["rapid"]
    with pytest.raises(ValidationError, match="clip_s"):
        PipelineConfig.model_validate(raw)


def test_settings_defaults_and_env_override(tmp_path: Path, monkeypatch) -> None:
    s = Settings(_env_file=None)
    assert s.data_dir == REPO_ROOT / "data"
    assert s.processed_dir == REPO_ROOT / "data" / "processed"
    assert s.pipeline_config == PIPELINE_YAML

    monkeypatch.setenv("ZEITNOT_DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.raw_dir == tmp_path / "raw"
