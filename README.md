# Zeitnot

*Zeitnot* (German, "time trouble") is the chess term for running short on the clock.

Zeitnot is a chess training platform for players who find good moves when they have
time, but play worse — or lose on time — when the clock gets short. It trains one
skill: **finding a good move within the time a stronger player would need.**

> **Status:** early development. The data pipeline is being built; the training app is
> not usable yet.

## The idea

Most training tools give you a position and unlimited time, or a puzzle where you
already know there is a tactic. Real games are different: the clock is running, most
positions are not puzzles, and recognizing that a moment matters is half the work.

Zeitnot shows you a position taken from a real game and gives you a **time budget**:
roughly how long a player at your **target rating** (for example, 300 points above your
current rating) would need to find a good move there, **in the time control you choose**
(blitz or rapid). Then it shows how you did — whether your move was good, how your time
compared with the budget, and what the engine and stronger players would have played.

## What makes it different

- **Positions of every kind.** Not only tactics: quiet positions, defensive moments,
  endgames, positions with one only move and positions where many moves are fine. The
  kind of position is hidden until you have moved.
- **Budgets learned from real games.** The time budgets come from a model trained on
  millions of real moves with clock data from the open Lichess database: who played
  what, at which rating, with how much time, and whether the move held up. A budget
  depends on the position, your target rating and the time control.
- **"Good move" means it keeps the game.** A move counts as good if it gives up almost
  none of your winning chances compared with the engine's best move — not only if it is
  the single best move.
- **For every level.** From beginners to titled players: a 1500 aiming for 1700 and a
  2500 aiming for 2800 train the same way. Ratings follow the Lichess rating scale.
- **Honest about how real games work.** In a game you often calculate a line once and
  then play its moves quickly. Zeitnot accounts for that, so a budget reflects a player
  seeing the position fresh — the situation in training.

## Planned features

- **Timed Trainer** — the core loop described above.
- **Think or move?** — a drill for recognizing when a position deserves time.
- **My mistakes** — your own errors, imported from your games, brought back with spaced
  repetition.
- **Game review** — where your time went in a game, compared with how critical each
  moment was.
- **Ratings and progress** — accuracy and speed ratings per time control, and skills by
  type of position.
- **Adaptive practice** — position selection that follows your progress.
- **Sparring** — play a human-like opponent at a chosen strength with a real clock.

Game import starts with Chess.com; Lichess import comes later.

## How it is built

- **Data:** Lichess open databases (rated games with clock times, puzzles and engine
  evaluations), processed as streams into Parquet files.
- **Analysis:** Stockfish for move evaluation; a pretrained human-move model for
  "what players at this rating play here".
- **Modeling:** survival-style time models in Python (Polars, PyTorch).
- **App:** FastAPI backend and a React + TypeScript frontend (planned).

## Development

Backend (Python 3.12 via [uv](https://docs.astral.sh/uv/)), run from `backend/`:

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
```

Pipeline parameters live in `config/pipeline.yaml`; machine paths come from `ZEITNOT_*`
environment variables or a `.env` file. Engine tests need the official Stockfish build
unzipped into `engines/` (or `ZEITNOT_STOCKFISH_PATH`); they are skipped otherwise.
Pipeline steps are command-line scripts in `scripts/`.

The app reads a local SQLite database (`data/app/zeitnot.db`, or `ZEITNOT_DATABASE_URL`).
Build the Trainer's inputs and load them into it:

```
uv run python ../scripts/build_budgets.py   # time budgets per category, rating and time control
uv run python ../scripts/build_pool.py      # the positions the Trainer serves
uv run python ../scripts/load_app_db.py     # apply migrations, load positions and budgets
```

## Credits

Game, puzzle and evaluation data: the [Lichess open database](https://database.lichess.org)
(CC0). Engine: [Stockfish](https://stockfishchess.org).
