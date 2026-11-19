"""Word shingles and exact Jaccard similarity.

An article is represented by the *set* of its k-word shingles (k consecutive words). Two
articles that copy most of each other's sentences share most of their shingles, so the
Jaccard similarity of the two sets measures how much text they have in common:

    J(A, B) = |A ∩ B| / |A ∪ B|        (1 = same shingles, 0 = nothing in common)

Shingles are stored as 32-bit hashes (``zlib.crc32``) instead of strings: integers are much
smaller and can go straight into MinHash. ``crc32`` is used rather than Python's ``hash()``
because ``hash()`` of a string is randomized per process (PYTHONHASHSEED), so the same text
would get different shingle ids in every run, and cached signatures would be useless.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

K = 3  # words per shingle

# A "word" is a run of letters/digits/underscore. Punctuation and whitespace are separators, so
# "tax bill." and "tax bill," give the same shingles. (`content` is already lower-cased.)
_WORD_RE = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Lower-cased words of ``text`` (punctuation dropped)."""
    return _WORD_RE.findall(text.lower())


def shingle_ids(text: str, k: int = K) -> np.ndarray:
    """The set of k-word shingles of ``text``, as a sorted array of unique int64 hashes.

    >>> len(shingle_ids("trump signs new tax bill today"))   # 6 words -> 4 shingles of 3 words
    4

    A text with fewer than ``k`` words has no shingles (empty array).
    """
    w = words(text)
    ids = [zlib.crc32(" ".join(w[i:i + k]).encode("utf-8")) for i in range(len(w) - k + 1)]
    return np.unique(np.array(ids, dtype=np.int64))  # np.unique sorts and removes repeats


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity of two shingle sets given as arrays of unique ids.

    Two empty sets share nothing we can measure, so their similarity is defined as 0.
    """
    inter = len(np.intersect1d(a, b, assume_unique=True))
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0
