"""Label sampled positions with Stockfish (resumable; one engine per worker).

Usage (from backend/):
    uv run python ../scripts/engine_label.py [--positions FILE] [--workers N] [--max-positions N]
"""

from zeitnot.positions.labeling import main

if __name__ == "__main__":  # required: workers are started with "spawn" on Windows
    raise SystemExit(main())
