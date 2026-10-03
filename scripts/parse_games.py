"""Parse filtered games into the games and moves tables.

Usage (from backend/):
    uv run python ../scripts/parse_games.py YYYY-MM [--filtered DIR] [--out DIR] [--max-parts N]
"""

from zeitnot.data.parse_games import main

if __name__ == "__main__":  # required: workers are started with "spawn" on Windows
    raise SystemExit(main())
