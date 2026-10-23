"""Migrate the app database and load the position pool and budgets into it.

Usage (from backend/):
    uv run python ../scripts/load_app_db.py [--pool FILE] [--budgets FILE] [--model m0] [--db URL]
"""

from zeitnot.api.load import main

if __name__ == "__main__":
    raise SystemExit(main())
