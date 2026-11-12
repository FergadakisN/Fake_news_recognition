"""Make sure everyone in the group works on byte-identical data.

Three checks:

1. **Raw file**: the Kaggle version is pinned (``config.KAGGLE_VERSION``) and the downloaded
   CSV must have the SHA-256 in ``config.RAW_SHA256``.
2. **Cleaned data**: ``data/processed/clean_fingerprint.json`` (committed to git) stores the
   row count, the number of fakes and hashes of ``doc_id``/``is_fake``, ``title``/``text`` and
   ``content`` for both cleaned variants. Rebuilding or loading compares against it, so a
   different library version or a local change to the cleaning rules is caught immediately.
3. **Splits**: ``splits.csv`` must contain exactly the cleaned ``doc_id``s.

If the cleaning rules are changed on purpose, Person 1 rebuilds with
``python3 scripts/prepare_data.py --rebuild --update-fingerprint`` and commits the new JSON.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fake_news import config

VARIANTS = {"clean": "remove_markers=True", "clean_with_markers": "remove_markers=False"}

_HOW_TO_FIX = (
    "Your data differs from the group's reference. Try, in order:\n"
    "  1. git pull  (get the latest code, splits.csv and clean_fingerprint.json)\n"
    "  2. pip install -e .  and check  python3 -c 'import pandas, sys; print(sys.version, pandas.__version__)'\n"
    "  3. python3 scripts/prepare_data.py --rebuild\n"
    "If the cleaning rules were changed on purpose, Person 1 runs\n"
    "  python3 scripts/prepare_data.py --rebuild --update-fingerprint  and commits the new JSON."
)


class DataMismatchError(RuntimeError):
    """The local data is not identical to the group's reference data."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_file(path: Path | None = None) -> None:
    """Raise :class:`DataMismatchError` if the raw CSV is not the pinned Kaggle version."""
    path = path or config.RAW_CSV
    actual = sha256_file(path)
    if actual != config.RAW_SHA256:
        raise DataMismatchError(
            f"{path} has SHA-256 {actual}, expected {config.RAW_SHA256} "
            f"(Kaggle version {config.KAGGLE_VERSION}). Delete it and run "
            "`python3 scripts/prepare_data.py` to download the pinned version."
        )


class _Fingerprinter:
    """Incremental fingerprint: feed chunks sorted by doc_id, read the result at the end.

    Hashes only the columns that are present, so a frame loaded with a subset of columns
    can still be checked (``doc_id`` and ``is_fake`` are always required).
    """

    def __init__(self) -> None:
        self.n_rows = 0
        self.n_fake = 0
        self.hashers: dict[str, "hashlib._Hash"] = {}
        self.last_doc_id: int | None = None

    def _update(self, key: str, values) -> None:
        h = self.hashers.setdefault(key, hashlib.sha256())
        for v in values:
            h.update(str(v).encode("utf-8"))
            h.update(b"\x1e")  # record separator, so ["ab", "c"] != ["a", "bc"]

    def update(self, df: pd.DataFrame) -> None:
        ids = df["doc_id"].to_numpy()
        if len(ids) and ((self.last_doc_id is not None and ids[0] <= self.last_doc_id) or (np.diff(ids) <= 0).any()):
            raise ValueError("Fingerprint input must be sorted by increasing doc_id.")
        if len(ids):
            self.last_doc_id = int(ids[-1])
        self.n_rows += len(df)
        self.n_fake += int(df["is_fake"].sum())
        self._update("doc_ids_labels_sha256", (f"{d}:{y}" for d, y in zip(df["doc_id"], df["is_fake"])))
        if {"title", "text"} <= set(df.columns):
            self._update("title_text_sha256", (f"{t}\x1f{x}" for t, x in zip(df["title"], df["text"])))
        if "content" in df.columns:
            self._update("content_sha256", df["content"])

    def result(self) -> dict:
        out = {"n_rows": self.n_rows, "n_fake": self.n_fake}
        out.update({k: h.hexdigest() for k, h in self.hashers.items()})
        return out


def fingerprint(df: pd.DataFrame) -> dict:
    """Row count and content hashes of a cleaned frame (order-independent: sorted by doc_id)."""
    fp = _Fingerprinter()
    fp.update(df.sort_values("doc_id"))
    return fp.result()


def fingerprint_file(path: Path) -> dict:
    """Fingerprint of a cleaned CSV, read in chunks (bounded memory). Rows must be sorted by doc_id."""
    from fake_news.data.load import read_clean_csv_chunks

    fp = _Fingerprinter()
    for chunk in read_clean_csv_chunks(path):
        fp.update(chunk)
    return fp.result()


def load_expected() -> dict:
    if not config.FINGERPRINT_JSON.exists():
        return {}
    return json.loads(config.FINGERPRINT_JSON.read_text())


def save_fingerprint_values(values: dict, variant: str) -> None:
    """Store a fingerprint as the reference for ``variant`` (Person 1 only)."""
    _check_variant(variant)
    missing = {"n_rows", "n_fake", "doc_ids_labels_sha256", "title_text_sha256", "content_sha256"} - set(values)
    if missing:
        raise ValueError(f"A reference fingerprint needs all columns; missing {missing}")
    expected = load_expected()
    expected[variant] = values
    config.FINGERPRINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.FINGERPRINT_JSON.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def save_fingerprint(df: pd.DataFrame, variant: str) -> None:
    save_fingerprint_values(fingerprint(df), variant)


def check_fingerprint_values(actual: dict, variant: str) -> bool:
    """Compare a fingerprint with the stored reference (only the keys present in ``actual``).

    Returns True if it matches, False (with a warning) if no reference exists yet, and raises
    :class:`DataMismatchError` listing what differs otherwise.
    """
    _check_variant(variant)
    expected = load_expected().get(variant)
    if expected is None:
        warnings.warn(f"No reference fingerprint for {variant!r} in {config.FINGERPRINT_JSON}; data not verified.")
        return False
    diffs = [f"  {k}: expected {expected[k]}, got {actual[k]}" for k in actual if expected.get(k) != actual[k]]
    if diffs:
        raise DataMismatchError(f"Cleaned data ({variant}) does not match the reference:\n"
                                + "\n".join(diffs) + "\n" + _HOW_TO_FIX)
    return True


def check_fingerprint(df: pd.DataFrame, variant: str) -> bool:
    """:func:`check_fingerprint_values` for an in-memory frame (all or some columns)."""
    return check_fingerprint_values(fingerprint(df), variant)


def verify_clean_file(path: Path, variant: str) -> bool:
    """Check a cleaned CSV against the reference without loading it fully into memory."""
    return check_fingerprint_values(fingerprint_file(path), variant)


def check_splits_match(clean_doc_ids, splits: pd.DataFrame) -> None:
    """Raise :class:`DataMismatchError` unless ``splits.csv`` has exactly the cleaned doc_ids."""
    clean_ids, split_ids = set(clean_doc_ids), set(splits["doc_id"])
    if clean_ids != split_ids:
        raise DataMismatchError(
            f"splits.csv and the cleaned data disagree: {len(clean_ids - split_ids)} cleaned articles "
            f"have no split, {len(split_ids - clean_ids)} split rows are not in the cleaned data.\n" + _HOW_TO_FIX
        )


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
