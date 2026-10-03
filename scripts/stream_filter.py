"""Filter a Lichess monthly dump to the games we model.

Usage (from backend/):
    uv run python ../scripts/stream_filter.py <path/to/lichess_db_standard_rated_YYYY-MM.pgn.zst>
"""

from zeitnot.data.stream_filter import main

if __name__ == "__main__":  # required: workers are started with "spawn" on Windows
    raise SystemExit(main())
