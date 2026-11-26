"""Locality-sensitive hashing (LSH) of MinHash signatures with the banding technique.

Comparing every pair of 62,525 articles means ~2 billion comparisons. LSH avoids that: it
only proposes *candidate pairs* that are likely to be similar, and only those are checked.

Banding. Each signature of length n = bands * rows is cut into ``bands`` bands of ``rows``
values. Within a band, articles whose ``rows`` values are all identical land in the same
bucket. Two articles become a candidate pair if they share a bucket in at least one band.

If two articles have Jaccard similarity s, one band matches with probability s**rows, so

    P(candidate) = 1 - (1 - s**rows) ** bands.

With 16 bands x 8 rows this S-curve is steep around its threshold (1/bands) ** (1/rows) ≈ 0.71:

    s = 0.5 -> 0.06     s = 0.7 -> 0.61     s = 0.8 -> 0.95     s = 0.9 -> 1.00

So pairs above ~0.8 are almost never missed, and pairs below ~0.5 are rarely proposed. Candidates
are still verified with the exact Jaccard similarity afterwards (see ``groups.py``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Iterator

import numpy as np

BANDS = 16
ROWS = 8


def candidate_probability(s, bands: int = BANDS, rows: int = ROWS):
    """Probability that a pair with Jaccard similarity ``s`` becomes a candidate (the LSH S-curve)."""
    return 1 - (1 - np.asarray(s, dtype=float) ** rows) ** bands


class LSHIndex:
    """Bucket tables of MinHash signatures: one hash table per band.

    ``add`` stores a signature under a key (e.g. a ``doc_id``); ``query`` returns the keys of
    stored signatures that share at least one band bucket with a new signature. This lets us
    index the training articles only and look up validation/test articles later, without the
    new articles ever entering the index.
    """

    def __init__(self, bands: int = BANDS, rows: int = ROWS):
        self.bands, self.rows = bands, rows
        # tables[band][bucket key] -> keys of the signatures in that bucket
        self.tables: list[dict[bytes, list]] = [defaultdict(list) for _ in range(bands)]
        self.n_items = 0

    def _band_keys(self, sig: np.ndarray) -> Iterator[bytes]:
        """The bucket key of each band: the raw bytes of its ``rows`` values (equal values <=> equal key)."""
        sig = np.ascontiguousarray(sig, dtype=np.int64)
        if sig.shape != (self.bands * self.rows,):
            raise ValueError(f"Signature must have length {self.bands * self.rows}, got shape {sig.shape}")
        for band in range(self.bands):
            yield sig[band * self.rows:(band + 1) * self.rows].tobytes()

    def add(self, key: Hashable, sig: np.ndarray) -> None:
        """Store signature ``sig`` under ``key``."""
        for table, band_key in zip(self.tables, self._band_keys(sig)):
            table[band_key].append(key)
        self.n_items += 1

    def add_many(self, keys: Iterable[Hashable], sigs: np.ndarray) -> None:
        for key, sig in zip(keys, sigs):
            self.add(key, sig)

    def query(self, sig: np.ndarray) -> set:
        """Keys of all stored signatures that share at least one band bucket with ``sig``."""
        found: set = set()
        for table, band_key in zip(self.tables, self._band_keys(sig)):
            found.update(table.get(band_key, ()))  # .get: a lookup must not create empty buckets
        return found

    def buckets(self) -> Iterator[list]:
        """Every non-empty bucket of every band (lists of keys)."""
        for table in self.tables:
            yield from table.values()


def candidate_pairs(signatures: np.ndarray, bands: int = BANDS, rows: int = ROWS) -> np.ndarray:
    """All candidate pairs among the rows of ``signatures`` (shape ``(n_docs, bands * rows)``).

    Returns an int64 array of shape ``(n_pairs, 2)`` of row indices ``i < j``, each pair once,
    sorted. A pair that shares buckets in several bands is still listed once.
    """
    n = len(signatures)
    index = LSHIndex(bands, rows)
    index.add_many(range(n), signatures)

    # Every pair inside a bucket is a candidate. Pairs are encoded as one integer i * n + j so
    # duplicates across bands can be removed with np.unique (much lighter than a set of tuples).
    codes = []
    for bucket in index.buckets():
        if len(bucket) > 1:
            members = np.array(bucket, dtype=np.int64)  # ascending: rows were added in order
            i, j = np.triu_indices(len(members), k=1)
            codes.append(members[i] * n + members[j])
    if not codes:
        return np.empty((0, 2), dtype=np.int64)
    codes = np.unique(np.concatenate(codes))
    return np.column_stack([codes // n, codes % n])


def bucket_sizes(signatures: np.ndarray, bands: int = BANDS, rows: int = ROWS) -> np.ndarray:
    """Size of every bucket that holds 2+ articles, over all bands (for diagnostics)."""
    index = LSHIndex(bands, rows)
    index.add_many(range(len(signatures)), signatures)
    return np.array([len(b) for b in index.buckets() if len(b) > 1], dtype=np.int64)
