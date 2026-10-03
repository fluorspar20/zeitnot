"""Draw the stratified, seeded sample of positions to label.

Usage (from backend/):
    uv run python ../scripts/sample_positions.py YYYY-MM [YYYY-MM ...] [--interim DIR] [--target N]
"""

from zeitnot.positions.sampling import main

if __name__ == "__main__":
    raise SystemExit(main())
