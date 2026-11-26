"""From candidate pairs to near-duplicate groups.

1. **Verify.** LSH candidates are only *likely* similar (MinHash is an estimate, and the S-curve
   lets some lower pairs through). Each candidate pair is therefore checked with the exact Jaccard
   similarity of the two shingle sets, and kept only if it reaches the threshold.
2. **Group.** Near-duplicate is a pairwise relation, but a split needs groups: if A ~ B and B ~ C,
   then A, B and C must land in the same partition, even if A and C are not similar themselves
   (otherwise B leaks information between partitions). The groups are the connected components of
   the graph whose edges are the verified pairs, found with union-find.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd

from fake_news.similarity.shingles import jaccard


class UnionFind:
    """Disjoint sets (union-find) with path compression and union by size.

    ``union(x, y)`` merges the sets of x and y; ``find(x)`` returns the representative ("root")
    of x's set. Both take almost constant time, so grouping all verified pairs is linear in their number.
    """

    def __init__(self, items: Iterable[Hashable] = ()):
        self.parent: dict = {}
        self.size: dict = {}
        for x in items:
            self.add(x)

    def add(self, x: Hashable) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: Hashable) -> Hashable:
        self.add(x)
        root = x
        while self.parent[root] != root:  # walk up to the root
            root = self.parent[root]
        while self.parent[x] != root:     # path compression: point everything on the way at the root
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: Hashable, y: Hashable) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:  # attach the smaller tree under the larger one
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]

    def components(self) -> dict:
        """item -> root of its set."""
        return {x: self.find(x) for x in self.parent}


def exact_jaccard_pairs(pairs: np.ndarray, shingles: Mapping[Hashable, np.ndarray]) -> np.ndarray:
    """Exact Jaccard similarity of each pair ``(x, y)`` in ``pairs``, looking up shingle sets in ``shingles``."""
    return np.array([jaccard(shingles[x], shingles[y]) for x, y in pairs], dtype=float)


def build_groups(doc_ids: Iterable[Hashable], pairs: np.ndarray, similarity: np.ndarray,
                 threshold: float) -> pd.Series:
    """Near-duplicate groups: connected components of the pairs with ``similarity >= threshold``.

    Returns a Series ``doc_id -> group_id`` covering every id in ``doc_ids``. The ``group_id`` is
    the smallest ``doc_id`` in the group, so it does not depend on processing order, and an
    article without near-duplicates is its own group (``group_id == doc_id``).
    """
    uf = UnionFind(doc_ids)
    for (x, y), s in zip(pairs, similarity):
        if s >= threshold:
            uf.union(x, y)

    root = pd.Series(uf.components())                    # doc_id -> root of its set
    ids = pd.Series(root.index, index=root.index)
    group_id = ids.groupby(root.to_numpy()).transform("min")  # smallest doc_id among the same root
    group_id.index.name = "doc_id"
    return group_id.rename("group_id").sort_index()
