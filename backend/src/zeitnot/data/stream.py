"""Stream a ``.pgn.zst`` file as chunks of whole games, without materializing it."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import zstandard

log = logging.getLogger(__name__)

# Games in a Lichess dump are separated by a blank line before the next [Event tag.
GAME_BOUNDARY = b"\n\n[Event "
_SPLIT_RE = re.compile(r"\n\n(?=\[Event )")
_READ_SIZE = 1 << 22  # 4 MiB reads from the decompressor


def iter_game_chunks(path: Path, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    """Yield ``(index, data)``: ~``chunk_bytes`` of decompressed text cut at a game boundary.

    Each chunk starts at an ``[Event`` tag. Boundaries depend only on the input and
    ``chunk_bytes``, so a rerun produces the same chunks (needed for resuming).
    A truncated input (e.g. a partial download) ends early; its last game is
    usually incomplete and is rejected downstream.
    """
    index = 0
    buffer = bytearray()
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        while True:
            try:
                block = reader.read(_READ_SIZE)
            except zstandard.ZstdError as e:
                log.warning("decompression stopped early (truncated input?): %s", e)
                block = b""
            if not block:
                break
            buffer += block
            while len(buffer) >= chunk_bytes:
                # Last boundary inside the first chunk_bytes; if one game is larger
                # than that, the first boundary after it. Independent of read sizes.
                cut = buffer.rfind(GAME_BOUNDARY, 0, chunk_bytes)
                if cut <= 0:
                    cut = buffer.find(GAME_BOUNDARY, 1)
                if cut <= 0:
                    break  # no complete game yet; keep reading
                cut += 2  # keep the blank line with the previous game
                yield index, bytes(buffer[:cut])
                index += 1
                del buffer[:cut]
    if buffer.strip():
        yield index, bytes(buffer)


def split_games(text: str) -> list[str]:
    """Split a chunk of PGN text into individual game texts."""
    return [g for g in _SPLIT_RE.split(text) if g.strip()]
