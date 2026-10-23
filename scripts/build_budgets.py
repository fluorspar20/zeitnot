"""Build the M0 heuristic budget table from the labeled observations.

Usage (from backend/):
    uv run python ../scripts/build_budgets.py [--observations FILE] [--out DIR]
"""

from zeitnot.models.m0 import main

if __name__ == "__main__":
    raise SystemExit(main())
