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

Encoding is tracked per file (path + size + mtime), not just as one
all-or-nothing corpus manifest. This means adding a brand-new source folder
(or a handful of new files to an existing one) only encodes the new/changed
files and appends them to the token stream -- files that were already
encoded, under the same tokenizer, are left untouched. This matters most on
free-tier Colab, where re-encoding tens of thousands of already-done files
just because one new folder was added would burn most of the session's
compute budget on repeated work. A tokenizer change (different vocab size or
merges) still forces a full rebuild, since every existing token ID would be
wrong under new merges -- nothing can be salvaged in that case.

Encoding itself is also resumable within one incremental batch. Every
``PROGRESS_INTERVAL`` files, the current progress (which files in this
batch are done, and the full per-file token index so far) is written to a
small progress JSON file. If the process is interrupted (e.g. a Colab
disconnect) and re-launched against the same data/cache directories with the
same new/changed file set, encoding picks up right after the last completed
file instead of starting the batch over.
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
DEFAULT_SOURCE_DIRS = ("books", "books2", "wiki", "qna")

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
    so users can add new categories incrementally without breaking older
    corpora that only have some of them.

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
    name order, then path order within each source.
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


def _file_stat(path: Path) -> dict:
    """Cheap identity check for a source file: size + mtime.

    Good enough to detect "this file changed since we encoded it" without
    hashing file contents, which would be far too slow across tens of
    thousands of books on every run.
    """
    stat = path.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _file_index_path(cache_dir: Path) -> Path:
    return cache_dir / "file_index.json"


def _progress_path(cache_dir: Path) -> Path:
    return cache_dir / "encode_progress.json"


def _load_file_index(cache_dir: Path, tokenizer_expected: dict) -> Optional[dict]:
    """Load the per-file token index, or None if absent/unusable.

    Only usable if the tokenizer identity (vocab size + BPE merges) matches
    exactly -- reusing token IDs encoded under different merges would
    silently corrupt the stream, so any tokenizer change forces a clean
    rebuild rather than a partial reuse.
    """
    path = _file_index_path(cache_dir)
    if not path.exists():
        return None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if index.get("tokenizer") != tokenizer_expected:
        return None
    return index


def _save_file_index(cache_dir: Path, index: dict) -> None:
    _file_index_path(cache_dir).write_text(json.dumps(index, indent=2), encoding="utf-8")


def _load_resumable_progress(
    cache_dir: Path, token_path: Path, tokenizer_expected: dict, batch_size: int,
) -> Optional[dict]:
    """Return a valid in-progress checkpoint for the current incremental
    batch of new/changed files, or None if none applies.

    A checkpoint is only usable if it was written for the exact same
    tokenizer AND the exact same number of files in this batch -- if the set
    of new/changed files differs from what the checkpoint expected (e.g.
    yet another folder was added mid-run), we refuse to resume and let the
    caller restart this batch cleanly instead of risking a misaligned token
    stream.
    """
    progress_file = _progress_path(cache_dir)
    if not progress_file.exists() or not token_path.exists():
        return None
    try:
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if progress.get("tokenizer") != tokenizer_expected or progress.get("batch_size") != batch_size:
        return None

    expected_bytes = int(progress["token_count"]) * array("I").itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes < expected_bytes:
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
    ``source_dirs``. Each sub-folder's .txt files are tracked as one source
    category. Missing sub-folders are simply skipped, so you can add a new
    category later -- only ITS files get encoded; existing sources' already-
    encoded files are reused as-is (see module docstring).

    A per-source token count breakdown is written into the cache metadata
    (``metadata["source_token_counts"]``) so later tooling can evaluate the
    model separately per category.

    ``rebuild=True`` discards the existing cache entirely and re-encodes
    every file from scratch (also the automatic fallback if the tokenizer
    itself changed, since old token IDs would be invalid under new merges).
    """
    if tokenizer is not None and not isinstance(tokenizer, BPETokenizer):
        raise ValueError("bulk training requires a BPETokenizer")

    sources = _collect_source_paths(data_dir, source_dirs)
    all_paths = [p for name in sorted(sources) for p in sources[name]]
    path_to_source = {str(p): name for name, paths in sources.items() for p in paths}

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    token_path = cache / "tokens.int32.bin"
    meta_path = cache / "metadata.json"
    progress_path = _progress_path(cache)

    if tokenizer is None:
        tokenizer = BPETokenizer(
            sample_book_text(all_paths, bpe_sample_chars),
            vocab_size=bpe_vocab_size,
            sample_chars=bpe_sample_chars,
        )

    tokenizer_expected = {
        "tokenizer": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "merges": [list(pair) for pair in tokenizer.merges],
    }

    file_index = None if rebuild else _load_file_index(cache, tokenizer_expected)
    if file_index is None:
        # No usable per-file index: first-ever run, an explicit rebuild, or
        # the tokenizer itself changed (every existing token ID would be
        # wrong under new merges, so nothing can be salvaged).
        if token_path.exists():
            token_path.unlink()
        files_record: dict[str, dict] = {}
        base_token_count = 0
    else:
        files_record = file_index["files"]
        base_token_count = file_index["token_count"]
    token_count = base_token_count

    # Decide which files are already correctly encoded (same path, size, and
    # mtime, under the current tokenizer) versus which need a fresh encode:
    # new files, changed files, or everything on a forced/tokenizer rebuild.
    to_encode = []
    for path in all_paths:
        key = str(path)
        stat = _file_stat(path)
        recorded = files_record.get(key)
        if recorded is not None and recorded["size"] == stat["size"] and recorded["mtime"] == stat["mtime"]:
            continue  # already encoded under this exact tokenizer, reuse as-is
        to_encode.append(path)

    stale_changed = [path for path in to_encode if str(path) in files_record]
    if stale_changed:
        print(
            f"[WARNING] {len(stale_changed)} previously-encoded file(s) changed on disk "
            "(different size/mtime). Their old tokens remain in the cache and the new "
            "content will be appended as well, so the corpus will contain both versions. "
            "Pass rebuild=True (or delete the cache_dir) for a clean re-encode if that's "
            "not what you want."
        )

    if to_encode:
        resume_progress = None if rebuild else _load_resumable_progress(
            cache, token_path, tokenizer_expected, len(to_encode),
        )
        if resume_progress is not None:
            start_index = int(resume_progress["files_done"])
            token_count = int(resume_progress["token_count"])
            files_record = resume_progress["files_record"]
            file_mode = "r+b"
            print(
                f"Resuming incremental encode from file {start_index:,}/{len(to_encode):,} "
                f"in this batch | {token_count:,} tokens in cache so far"
            )
        else:
            start_index = 0
            token_count = base_token_count
            # Guard against leftover bytes from an earlier, now-invalidated
            # attempt at this batch (e.g. a stale/mismatched progress file
            # from a run that added a different set of new files).
            expected_bytes = base_token_count * array("I").itemsize
            if token_path.exists() and token_path.stat().st_size > expected_bytes:
                with token_path.open("r+b") as handle:
                    handle.truncate(expected_bytes)
            file_mode = "ab" if token_path.exists() else "wb"

        with token_path.open(file_mode) as output:
            if file_mode == "r+b":
                output.seek(0, 2)  # append after the resumed, possibly truncated, prefix
            for index, path in enumerate(to_encode[start_index:], start=start_index + 1):
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    encoded = tokenizer.encode(handle.read())
                # Delimit files so a batch cannot silently join the final and
                # first token of adjacent files without a boundary. Skipped
                # only when the stream is still empty (nothing to delimit
                # against yet).
                if token_count > 0:
                    encoded = tokenizer.encode("\n\n") + encoded
                values = array("I", encoded)
                values.tofile(output)

                key = str(path)
                stat = _file_stat(path)
                files_record[key] = {
                    "source": path_to_source[key],
                    "size": stat["size"],
                    "mtime": stat["mtime"],
                    "token_start": token_count,
                    "token_count": len(encoded),
                }
                token_count += len(encoded)

                if index % PROGRESS_INTERVAL == 0 or index == len(to_encode):
                    output.flush()
                    progress_path.write_text(
                        json.dumps(
                            {
                                "tokenizer": tokenizer_expected,
                                "batch_size": len(to_encode),
                                "files_done": index,
                                "token_count": token_count,
                                "files_record": files_record,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"Encoded {index:,}/{len(to_encode):,} new/changed files "
                        f"| {token_count:,} tokens total in cache | checkpoint saved"
                    )
        # Batch finished cleanly; the progress file's job is done. Remove it
        # so a future run doesn't mistake a *different* future batch for a
        # resumable one.
        if progress_path.exists():
            progress_path.unlink()
    else:
        print(f"Reusing bulk token cache: {token_path} ({token_count:,} tokens) -- no new or changed files")

    source_token_counts: dict[str, int] = {}
    for record in files_record.values():
        source_token_counts[record["source"]] = source_token_counts.get(record["source"], 0) + record["token_count"]

    file_index = {"tokenizer": tokenizer_expected, "files": files_record, "token_count": token_count}
    _save_file_index(cache, file_index)

    metadata = {
        **tokenizer_expected,
        "token_count": token_count,
        "source_token_counts": source_token_counts,
        "file_count": len(files_record),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Bulk corpus ready: {len(files_record):,} files, {token_count:,} tokens "
        f"| by source: {source_token_counts}"
    )

    # Final safety net: verify the file on disk actually matches what we're
    # about to claim, regardless of which code path produced `token_count`.
    final_expected_bytes = token_count * array("I").itemsize
    final_actual_bytes = token_path.stat().st_size
    if final_actual_bytes != final_expected_bytes:
        raise RuntimeError(
            f"Bulk token cache is corrupt: {token_path} is "
            f"{final_actual_bytes:,} bytes, but the file index claims "
            f"{token_count:,} tokens ({final_expected_bytes:,} bytes). Delete this "
            f"cache_dir and re-run to force a clean re-encode."
        )

    return tokenizer, BulkTokenStore(token_path, token_count), metadata