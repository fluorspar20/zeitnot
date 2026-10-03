"""Look up sampled positions in the Lichess evaluation database.

Usage (from backend/):
    uv run python ../scripts/lookup_evals.py [--evaldb FILE] [--positions FILE] [--max-chunks N]
"""

from zeitnot.positions.evaldb import main

if __name__ == "__main__":  # required: workers are started with "spawn" on Windows
    raise SystemExit(main())
