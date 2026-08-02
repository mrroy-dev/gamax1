"""Incremental token-cache support for large directories of text books.

The normal trainer is intentionally simple and reads one text file into RAM.
This module is the large-corpus path: it samples text to train a tokenizer,
then encodes each book one at a time into an int32 binary file.  The finished
file is memory mapped, so training batches do not require an 11 GB Python
string or an equally large list of token IDs.
"""

from __future__ import annotations

import json
import mmap
from array import array
from pathlib import Path
from typing import Optional

import torch

from .tokenizer import BPETokenizer


class BulkTokenStore:
    """Read-only memory-mapped token storage with a torch tensor view."""

    def __init__(self, token_path: Path, token_count: int):
        self.token_path = Path(token_path)
        self._file = self.token_path.open("rb")
        # ACCESS_COPY gives torch a writable view without copying the entire
        # token file into RAM; writes remain private and never touch the cache.
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.tensor = torch.frombuffer(self._mapping, dtype=torch.int32, count=token_count)

    def close(self):
        # Release the tensor view before closing its backing mmap.
        self.tensor = None
        self._mapping.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def book_paths(data_dir: str | Path) -> list[Path]:
    """Return deterministic, recursive .txt input paths."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"book data directory does not exist: {root}")
    paths = sorted((path for path in root.rglob("*.txt") if path.is_file()), key=lambda p: str(p))
    if not paths:
        raise ValueError(f"no .txt books found under {root}")
    return paths


def sample_book_text(paths: list[Path], sample_chars: int) -> str:
    """Read at most ``sample_chars`` across the book files for BPE training."""
    if sample_chars <= 0:
        raise ValueError("sample_chars must be positive")
    pieces = []
    remaining = sample_chars
    for path in paths:
        if remaining <= 0:
            break
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(remaining)
        if text:
            pieces.append(text)
            remaining -= len(text)
    sample = "\n\n".join(pieces)
    if not sample.strip():
        raise ValueError("book files contain no readable text")
    return sample


def _source_manifest(paths: list[Path]) -> dict:
    return {
        "file_count": len(paths),
        "source_bytes": sum(path.stat().st_size for path in paths),
        "first_file": str(paths[0]),
        "last_file": str(paths[-1]),
    }


def build_or_load_bulk_tokens(
    data_dir: str | Path,
    cache_dir: str | Path,
    *,
    bpe_vocab_size: int = 8000,
    bpe_sample_chars: int = 3_000_000,
    tokenizer: Optional[BPETokenizer] = None,
    rebuild: bool = False,
) -> tuple[BPETokenizer, BulkTokenStore, dict]:
    """Build or reuse a token cache and return its tokenizer and mapped data.

    Bulk training deliberately supports BPE only.  Its byte vocabulary can
    represent every book without an ``<unk>`` token and can be restored from a
    checkpoint during resume.
    """
    if tokenizer is not None and not isinstance(tokenizer, BPETokenizer):
        raise ValueError("bulk training requires a BPETokenizer")
    paths = book_paths(data_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    token_path = cache / "tokens.int32.bin"
    meta_path = cache / "metadata.json"
    manifest = _source_manifest(paths)

    if tokenizer is None:
        tokenizer = BPETokenizer(
            sample_book_text(paths, bpe_sample_chars),
            vocab_size=bpe_vocab_size,
            sample_chars=bpe_sample_chars,
        )

    expected = {
        **manifest,
        "tokenizer": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "merges": [list(pair) for pair in tokenizer.merges],
    }
    reusable = False
    metadata = None
    if not rebuild and token_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        reusable = all(metadata.get(key) == value for key, value in expected.items())

    if not reusable:
        token_count = 0
        with token_path.open("wb") as output:
            for index, path in enumerate(paths, start=1):
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    encoded = tokenizer.encode(handle.read())
                # Delimit books so a batch cannot silently join the final and
                # first token of adjacent files without a boundary.
                if index > 1:
                    encoded = tokenizer.encode("\n\n") + encoded
                values = array("I", encoded)
                values.tofile(output)
                token_count += len(encoded)
                if index == 1 or index % 500 == 0 or index == len(paths):
                    print(f"Encoded {index:,}/{len(paths):,} books | {token_count:,} tokens")
        metadata = {**expected, "token_count": token_count}
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        token_count = int(metadata["token_count"])
        print(f"Reusing bulk token cache: {token_path} ({token_count:,} tokens)")

    return tokenizer, BulkTokenStore(token_path, int(metadata["token_count"])), metadata
