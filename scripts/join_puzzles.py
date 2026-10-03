"""Ingest the Lichess puzzle DB (once) and join puzzles to a month's games and moves.

Usage (from backend/):
    uv run python ../scripts/join_puzzles.py YYYY-MM [--interim DIR] [--reingest]
"""

from zeitnot.data.puzzles import main

if __name__ == "__main__":
    raise SystemExit(main())
