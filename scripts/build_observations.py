"""Join sampled decisions with their engine labels (loss, Y, difficulty signals).

Usage (from backend/):
    uv run python ../scripts/build_observations.py [--positions FILE] [--labels DIR] [--evaldb FILE]
"""

from zeitnot.positions.observations import main

if __name__ == "__main__":
    raise SystemExit(main())
