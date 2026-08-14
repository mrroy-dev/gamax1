"""Incremental token-cache support for large directories of text books.

The normal trainer is intentionally simple and reads one text file into RAM.
This module is the large-corpus path: it samples text to train a tokenizer,
then encodes each book one at a time into an int32 binary file.  The finished
file is memory mapped, so training batches do not require an 11 GB Python
string or an equally large list of token IDs.

This version supports multiple source categories (e.g. books, wiki, qna) held
in separate sub-directories under one root data directory.  Each source's
files are tracked separately in the metadata so downstream tooling can
compute per-category token counts and, later, per-category evaluation.

It also supports resuming an interrupted encode pass.  Every
``PROGRESS_INTERVAL`` files, the current file-index and running token-count
are written to a small progress JSON file.  If the process is interrupted
(e.g. a Colab disconnect) and re-launched against the same data/cache
directories, encoding picks up right after the last completed file instead
of starting over, as long as the partially written token file has not been
truncated or edited.
"""

from __future__ import annotations

import json
import mmap
from array import array
from pathlib import Path
from typing import Optional

import torch

from .tokenizer import BPETokenizer


# Default sub-directory names searched under the root data directory.
# Any of these that exist will be included; missing ones are skipped.
DEFAULT_SOURCE_DIRS = ("books", "wiki", "qna")

# Write a progress checkpoint every this many files during encoding.
PROGRESS_INTERVAL = 500


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


def _collect_source_paths(data_dir: str | Path, source_dirs=DEFAULT_SOURCE_DIRS) -> dict[str, list[Path]]:
    """Return a mapping of source-category name -> sorted .txt file paths.

    Looks for each name in ``source_dirs`` as a sub-directory of ``data_dir``.
    A source is skipped (not an error) if its sub-directory does not exist,
    so users can add "wiki" or "qna" incrementally without breaking older
    corpora that only have "books".

    Backward compatibility: if none of ``source_dirs`` exist under
    ``data_dir`` but ``data_dir`` itself directly contains .txt files (the
    old single-folder layout), those files are treated as one source named
    "books".
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")

    sources: dict[str, list[Path]] = {}
    for name in source_dirs:
        sub_dir = root / name
        if sub_dir.is_dir():
            paths = sorted(
                (path for path in sub_dir.rglob("*.txt") if path.is_file()),
                key=lambda p: str(p),
            )
            if paths:
                sources[name] = paths

    if not sources:
        # Backward-compatible fallback: old flat layout with .txt files
        # directly under data_dir (no books/wiki/qna sub-folders).
        flat_paths = sorted(
            (path for path in root.rglob("*.txt") if path.is_file()),
            key=lambda p: str(p),
        )
        if flat_paths:
            sources["books"] = flat_paths

    if not sources:
        raise ValueError(
            f"no .txt files found under {root} "
            f"(expected sub-folders like {source_dirs}, or .txt files directly inside)"
        )
    return sources


def book_paths(data_dir: str | Path) -> list[Path]:
    """Return deterministic, recursive .txt input paths across all sources.

    Kept for backward compatibility with callers that just want a flat list
    of every file regardless of source category. Order is: source category
    name order (books, wiki, qna, ...), then path order within each source.
    """
    sources = _collect_source_paths(data_dir)
    paths: list[Path] = []
    for name in sorted(sources):
        paths.extend(sources[name])
    return paths


def sample_book_text(paths: list[Path], sample_chars: int) -> str:
    """Read at most ``sample_chars`` across the given files for BPE training."""
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
        raise ValueError("input files contain no readable text")
    return sample


def _source_manifest(sources: dict[str, list[Path]]) -> dict:
    all_paths = [p for name in sorted(sources) for p in sources[name]]
    per_source = {
        name: {
            "file_count": len(paths),
            "source_bytes": sum(path.stat().st_size for path in paths),
        }
        for name, paths in sources.items()
    }
    return {
        "file_count": len(all_paths),
        "source_bytes": sum(path.stat().st_size for path in all_paths),
        "first_file": str(all_paths[0]),
        "last_file": str(all_paths[-1]),
        "sources": per_source,
    }


def _progress_path(cache_dir: Path) -> Path:
    return cache_dir / "encode_progress.json"


def _load_resumable_progress(cache_dir: Path, token_path: Path, expected: dict) -> Optional[dict]:
    """Return a valid in-progress checkpoint dict, or None if none applies.

    A checkpoint is only usable if it was written for the exact same corpus
    manifest/tokenizer as ``expected`` (same files, same BPE merges) AND the
    partially-written token file is at least as large as the checkpoint
    claims. If anything looks inconsistent (e.g. the data directory changed,
    or the token file was truncated below the checkpoint's byte offset), we
    refuse to resume and let the caller re-encode from scratch instead of
    risking a corrupted token stream.
    """
    progress_file = _progress_path(cache_dir)
    if not progress_file.exists() or not token_path.exists():
        return None
    try:
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if progress.get("manifest_check") != expected:
        return None

    expected_bytes = int(progress.get("token_count", 0)) * array("I").itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes < expected_bytes:
        # Token file is shorter than the checkpoint expects -- something is
        # inconsistent (e.g. an interrupted write mid-flush). Not safe to
        # resume from here.
        return None
    if actual_bytes > expected_bytes:
        # A partial extra file may have been written after the last
        # checkpoint before the interruption. Truncate back to the last
        # confirmed-good checkpoint boundary so the token stream stays
        # file-aligned, then resume from there.
        with token_path.open("r+b") as handle:
            handle.truncate(expected_bytes)

    return progress


def build_or_load_bulk_tokens(
    data_dir: str | Path,
    cache_dir: str | Path,
    *,
    bpe_vocab_size: int = 8000,
    bpe_sample_chars: int = 3_000_000,
    tokenizer: Optional[BPETokenizer] = None,
    rebuild: bool = False,
    source_dirs=DEFAULT_SOURCE_DIRS,
) -> tuple[BPETokenizer, BulkTokenStore, dict]:
    """Build or reuse a token cache and return its tokenizer and mapped data.

    Bulk training deliberately supports BPE only.  Its byte vocabulary can
    represent every book without an ``<unk>`` token and can be restored from a
    checkpoint during resume.

    ``data_dir`` may contain any combination of the sub-folders named in
    ``source_dirs`` (default: "books", "wiki", "qna"). Each sub-folder's .txt
    files are tracked as one source category. Missing sub-folders are simply
    skipped, so you can add a new category later without invalidating old
    single-category caches (adding files WILL still invalidate the cache and
    trigger a rebuild, since the token stream itself changes).

    A per-source token count breakdown is written into the cache metadata
    (``metadata["source_token_counts"]``) so later tooling can evaluate the
    model separately per category (e.g. books vs wiki vs qna).

    Encoding itself is resumable: every ``PROGRESS_INTERVAL`` files, progress
    is checkpointed to ``cache_dir/encode_progress.json``. If this function
    is interrupted (e.g. a Colab disconnect) and called again with the same
    ``data_dir``/``cache_dir`` and unchanged source files, it will pick up
    encoding right after the last checkpointed file instead of starting the
    full corpus over from scratch.
    """
    if tokenizer is not None and not isinstance(tokenizer, BPETokenizer):
        raise ValueError("bulk training requires a BPETokenizer")

    sources = _collect_source_paths(data_dir, source_dirs)
    all_paths = [p for name in sorted(sources) for p in sources[name]]

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    token_path = cache / "tokens.int32.bin"
    meta_path = cache / "metadata.json"
    progress_path = _progress_path(cache)
    manifest = _source_manifest(sources)

    if tokenizer is None:
        tokenizer = BPETokenizer(
            sample_book_text(all_paths, bpe_sample_chars),
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
        # Check whether an interrupted encode pass for this exact corpus can
        # be resumed instead of starting from file 1 again.
        resume_progress = None if rebuild else _load_resumable_progress(cache, token_path, expected)

        if resume_progress is not None:
            start_index = int(resume_progress["files_done"])
            token_count = int(resume_progress["token_count"])
            source_token_counts = dict(resume_progress["source_token_counts"])
            file_mode = "r+b"
            print(
                f"Resuming encode from file {start_index:,}/{len(all_paths):,} "
                f"| {token_count:,} tokens already written"
            )
        else:
            start_index = 0
            token_count = 0
            source_token_counts = {name: 0 for name in sources}
            file_mode = "wb"

        path_to_source = {}
        for name, paths in sources.items():
            for p in paths:
                path_to_source[str(p)] = name

        with token_path.open(file_mode) as output:
            if file_mode == "r+b":
                output.seek(0, 2)  # append after the resumed, possibly truncated, prefix
            for index, path in enumerate(all_paths[start_index:], start=start_index + 1):
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    encoded = tokenizer.encode(handle.read())
                # Delimit files so a batch cannot silently join the final and
                # first token of adjacent files without a boundary.
                if index > 1:
                    encoded = tokenizer.encode("\n\n") + encoded
                values = array("I", encoded)
                values.tofile(output)
                token_count += len(encoded)
                source_name = path_to_source[str(path)]
                source_token_counts[source_name] += len(encoded)

                if index % PROGRESS_INTERVAL == 0 or index == len(all_paths):
                    output.flush()
                    progress_path.write_text(
                        json.dumps(
                            {
                                "manifest_check": expected,
                                "files_done": index,
                                "token_count": token_count,
                                "source_token_counts": source_token_counts,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"Encoded {index:,}/{len(all_paths):,} files "
                        f"| {token_count:,} tokens "
                        f"| by source: {source_token_counts} "
                        f"| checkpoint saved"
                    )
        metadata = {
            **expected,
            "token_count": token_count,
            "source_token_counts": source_token_counts,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Encoding finished cleanly; the progress file's job is done. Remove
        # it so a future run doesn't mistake a *different* future rebuild
        # for a resumable one.
        if progress_path.exists():
            progress_path.unlink()
    else:
        token_count = int(metadata["token_count"])
        print(f"Reusing bulk token cache: {token_path} ({token_count:,} tokens)")
        if "source_token_counts" in metadata:
            print(f"  by source: {metadata['source_token_counts']}")

    return tokenizer, BulkTokenStore(token_path, int(metadata["token_count"])), metadata