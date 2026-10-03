import io
import json
from pathlib import Path

import polars as pl
import pytest
import zstandard

from zeitnot.config import REPO_ROOT, PipelineConfig
from zeitnot.data.game_filter import filter_game, split_game
from zeitnot.data.stream import iter_game_chunks, split_games
from zeitnot.data.stream_filter import run

FIXTURE = Path(__file__).parent / "fixtures" / "filter.pgn"

EXPECTED = {
    "keepBltz": "kept",
    "dropBult": "class_bullet",
    "dropBot1": "excluded_title",
    "dropAban": "excluded_termination",
    "dropNoCk": "no_clock",
    "dropNoEl": "missing_elo",
    "dropHiEl": "elo_out_of_range",
    "dropCorr": "no_time_control",
    "dropOld1": "before_min_month",
    "keepSwis": "kept",
    "keepClas": "kept",
    "dropTrnc": "incomplete",
}


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig.from_yaml(REPO_ROOT / "config" / "pipeline.yaml")


@pytest.fixture(scope="module")
def text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def zst(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lichess_db_standard_rated_2026-08.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode("utf-8")))
    return path


# --- header filter -----------------------------------------------------------


def test_every_drop_reason(text: str, cfg: PipelineConfig) -> None:
    games = split_games(text)
    assert len(games) == len(EXPECTED)
    for g in games:
        game_id = split_game(g)[0]["Site"].rsplit("/", 1)[1]
        kept, reason = filter_game(g, cfg.filter, cfg.time_controls)
        assert reason == EXPECTED[game_id], game_id
        assert (kept is not None) == (reason == "kept")


def test_kept_game_fields(text: str, cfg: PipelineConfig) -> None:
    kept, _ = filter_game(split_games(text)[0], cfg.filter, cfg.time_controls)
    assert kept is not None
    assert (kept.game_id, kept.tc_class, kept.base_s, kept.increment_s) == (
        "keepBltz",
        "blitz",
        180,
        2,
    )
    assert (kept.white_elo, kept.black_elo, kept.has_evals) == (1500, 1400, True)
    assert kept.movetext.startswith("1. e4") and kept.movetext.endswith("1-0")


# --- chunking ----------------------------------------------------------------


def test_chunks_cut_at_game_boundaries(zst: Path, text: str) -> None:
    chunks = list(iter_game_chunks(zst, chunk_bytes=300))
    assert len(chunks) > 3
    assert [i for i, _ in chunks] == list(range(len(chunks)))
    assert all(data.startswith(b"[Event ") for _, data in chunks)
    assert b"".join(data for _, data in chunks).decode("utf-8") == text


def test_chunks_are_deterministic(zst: Path) -> None:
    assert list(iter_game_chunks(zst, 500)) == list(iter_game_chunks(zst, 500))


def test_truncated_input_yields_a_prefix(tmp_path: Path, text: str) -> None:
    # zstd decodes whole blocks only. Write many blocks (a flush every 64 KiB, like the
    # streaming compressor behind real dumps) and cut a few bytes past a middle block
    # boundary: everything before that boundary must decode.
    big = text * 400
    raw = big.encode("utf-8")
    buf = io.BytesIO()
    writer = zstandard.ZstdCompressor().stream_writer(buf, closefd=False)
    step = 1 << 16
    block_ends = []
    for i in range(0, len(raw), step):
        writer.write(raw[i : i + step])
        writer.flush(zstandard.FLUSH_BLOCK)
        block_ends.append(buf.tell())
    writer.close()
    middle = len(block_ends) // 2
    cut = tmp_path / "cut.pgn.zst"
    cut.write_bytes(buf.getvalue()[: block_ends[middle] + 5])

    data = b"".join(d for _, d in iter_game_chunks(cut, 50_000)).decode("utf-8")
    assert len(data) >= (middle + 1) * step  # all complete blocks decoded
    assert len(data) < len(big)
    assert big.startswith(data)


# --- end-to-end job ----------------------------------------------------------


def test_run_writes_parts_report_and_resumes(
    tmp_path: Path, zst: Path, cfg: PipelineConfig
) -> None:
    out = tmp_path / "filtered"
    kwargs = {"workers": 1, "chunk_mb": 1, "max_in_flight": 2}
    report = run(zst, out, "2026-08", cfg, **kwargs)

    assert report["games_in"] == len(EXPECTED)
    assert report["games_kept"] == 3
    assert report["kept_by_class"] == {"blitz": 1, "rapid": 1, "classical": 1}
    assert report["drop_reasons"]["class_bullet"] == 1
    month_dir = out / "month=2026-08"
    df = pl.read_parquet(month_dir / "part-*.parquet")
    assert sorted(df["game_id"]) == ["keepBltz", "keepClas", "keepSwis"]
    assert df.schema["utc_date"] == pl.Date
    assert json.loads((month_dir / "_report.json").read_text())["games_kept"] == 3
    assert (month_dir / "_report.md").exists()

    # Rerun: every chunk is already done, so nothing is reprocessed.
    again = run(zst, out, "2026-08", cfg, **kwargs)
    assert again["last_run"]["chunks_processed"] == 0
    assert again["last_run"]["chunks_skipped"] == again["chunks"]
    assert again["games_kept"] == 3

    # A different chunk size would misalign chunk indices: refuse.
    with pytest.raises(SystemExit):
        run(zst, out, "2026-08", cfg, workers=1, chunk_mb=2, max_in_flight=2)
