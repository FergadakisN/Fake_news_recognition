"""MinHash signatures: short fixed-length sketches whose agreement estimates Jaccard similarity.

Idea. Take a random hash function h and look at the shingle with the smallest hash value in
each of two sets A and B. Every shingle of A ∪ B is equally likely to be that minimum, and the
two minima are equal exactly when it lies in A ∩ B. Hence

    P[ min h(A) = min h(B) ] = |A ∩ B| / |A ∪ B| = J(A, B).

With n independent hash functions the *signature* of a set is the vector of its n minima, and
the share of positions where two signatures agree is an unbiased estimate of J(A, B) (standard
error sqrt(J(1-J)/n), about 0.04 for n = 128). Every article becomes n integers, whatever its
length.

Hash functions. We use the universal family h(x) = (a*x + b) mod P with P = 2**31 - 1 (a
Mersenne prime), a in [1, P) and b in [0, P) drawn with a fixed seed so signatures are
reproducible. Shingle ids are reduced mod P first, so x < 2**31 and a*x + b < 2**62 + 2**31:
everything fits in int64 without overflow.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fake_news import config

P = 2**31 - 1  # Mersenne prime; the modulus of the hash functions
N_HASHES = 128


def make_hash_params(n: int = N_HASHES, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients ``a, b`` (int64 arrays of length ``n``) of the hash functions (a*x + b) mod P."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, size=n, dtype=np.int64)  # a != 0, otherwise h would be constant
    b = rng.integers(0, P, size=n, dtype=np.int64)
    return a, b


def signature(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature of one shingle set: for each hash function, the minimum hash value.

    Returns an int64 array of length ``len(a)``. The set must not be empty.
    """
    x = np.asarray(shingles, dtype=np.int64) % P
    if x.size == 0:
        raise ValueError("Cannot compute the MinHash signature of an empty set.")
    # (n_hashes, 1) * (1, n_shingles) -> hash of every shingle under every function
    hashes = (a[:, None] * x[None, :] + b[:, None]) % P
    return hashes.min(axis=1)


def signatures(shingle_sets: Sequence[np.ndarray], a: np.ndarray, b: np.ndarray,
               batch_shingles: int = 100_000) -> np.ndarray:
    """Signatures of many shingle sets at once: an int64 array of shape ``(len(shingle_sets), len(a))``.

    Same result as calling :func:`signature` on every set, but vectorized: the shingles of a batch
    of articles are concatenated into one array, hashed in one go, and the minimum per article is
    taken with ``np.minimum.reduceat`` (minimum over each article's slice). Batches hold about
    ``batch_shingles`` shingles, which bounds memory (128 x 100,000 int64 = ~100 MB per array).
    Every set must be non-empty.
    """
    n_docs, n_hashes = len(shingle_sets), len(a)
    out = np.empty((n_docs, n_hashes), dtype=np.int64)
    sizes = np.array([len(s) for s in shingle_sets])
    if (sizes == 0).any():
        raise ValueError("Cannot compute the MinHash signature of an empty set.")

    start = 0
    while start < n_docs:
        # Grow the batch until it holds ~batch_shingles shingles (at least one article).
        end = start + 1
        total = sizes[start]
        while end < n_docs and total + sizes[end] <= batch_shingles:
            total += sizes[end]
            end += 1

        x = np.concatenate(shingle_sets[start:end]).astype(np.int64) % P
        offsets = np.concatenate([[0], np.cumsum(sizes[start:end])[:-1]])  # where each article starts
        hashes = (a[:, None] * x[None, :] + b[:, None]) % P             # (n_hashes, batch shingles)
        out[start:end] = np.minimum.reduceat(hashes, offsets, axis=1).T
        start = end
    return out


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """MinHash estimate of J(A, B): the share of signature positions where the two agree."""
    return float(np.mean(sig_a == sig_b))
