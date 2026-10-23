"""Build the Trainer position pool (servable positions with engine lines and previous FEN).

Usage (from backend/):
    uv run python ../scripts/build_pool.py [--observations FILE] [--labels DIR] [--moves DIR]
                                           [--out DIR]
"""

from zeitnot.positions.pool import main

if __name__ == "__main__":
    raise SystemExit(main())
