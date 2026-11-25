"""Model B: k-nearest neighbours over shingle sets, retrieved with MinHash/LSH.

An article is classified by the labels of the most similar *training* articles:

1. **Index (train only).** Every training article becomes a MinHash signature in an LSH index.
2. **Retrieve.** A new article's signature is looked up in the index; the training articles that share
   a band bucket with it are its candidates.
3. **Score.** For each candidate the exact Jaccard similarity of the two shingle sets is computed.
4. **Vote.** Keep the ``k`` most similar candidates with similarity >= ``min_sim``.
   ``prob_fake`` = similarity-weighted share of fake neighbours:

       prob_fake = sum(sim_i * is_fake_i) / sum(sim_i)

   An article with no such neighbour is *not covered* and gets the fake rate of the training
   articles (computed in ``fit``, about 0.444), i.e. the best guess without evidence.

The banding decides which neighbours can be found at all. The 16 x 8 banding of the near-duplicate
groups only finds pairs above ~0.6 (see ``fake_news.similarity.lsh``). Retrieval wants lower
similarities too, so it uses looser bandings (32 x 4, 64 x 2) of the same 128 MinHash values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fake_news import config
from fake_news.similarity import minhash
from fake_news.similarity.lsh import LSHIndex
from fake_news.similarity.near_duplicates import MIN_SHINGLES
from fake_news.similarity.shingles import K, shingle_ids

SIM_FLOOR = 0.1       # neighbours below this similarity are never kept (lowest min_sim we tune)


@dataclass
class Encoded:
    """Shingle sets and MinHash signatures of a list of texts (computed once, reused for every banding)."""

    shingles: list[np.ndarray]   # sorted unique shingle ids per text
    signatures: np.ndarray       # (n_texts, n_hashes) int64; rows of texts without signature are -1
    has_sig: np.ndarray          # bool: text has >= MIN_SHINGLES shingles

    def __len__(self) -> int:
        return len(self.shingles)


def encode(texts: Sequence[str], n_hashes: int = minhash.N_HASHES, seed: int = config.SEED,
           shingle_k: int = K) -> Encoded:
    """Shingles + signatures of ``texts``. Texts with fewer than ``MIN_SHINGLES`` shingles get no signature."""
    shingles = [shingle_ids(t, shingle_k) for t in texts]
    has_sig = np.array([len(s) >= MIN_SHINGLES for s in shingles], dtype=bool)
    a, b = minhash.make_hash_params(n_hashes, seed)
    sigs = np.full((len(shingles), n_hashes), -1, dtype=np.int64)
    if has_sig.any():
        sigs[has_sig] = minhash.signatures([s for s, h in zip(shingles, has_sig) if h], a, b)
    return Encoded(shingles, sigs, has_sig)


@dataclass
class Neighbors:
    """Neighbours of each query: training row indices and exact Jaccard similarities, most similar first.

    Only neighbours with similarity >= ``floor`` are kept, so any ``min_sim >= floor`` can be applied later
    without querying again.
    """

    rows: list[np.ndarray]
    sims: list[np.ndarray]
    n_candidates: np.ndarray     # LSH candidates per query (before the similarity check)
    lookup_seconds: float        # total time in the LSH index
    scoring_seconds: float       # total time computing exact Jaccard

    def __len__(self) -> int:
        return len(self.rows)


class LSHNeighborIndex:
    """k-NN classifier over shingle sets with an LSH index of the training articles.

    >>> knn = LSHNeighborIndex(bands=32, rows=4).fit(train_texts, train_labels)
    >>> nb = knn.neighbors(val_texts)
    >>> prob_fake, covered = knn.predict_proba(nb, k=10, min_sim=0.2)
    """

    def __init__(self, bands: int = 32, rows: int = 4, n_hashes: int = minhash.N_HASHES,
                 seed: int = config.SEED, shingle_k: int = K):
        if bands * rows != n_hashes:
            raise ValueError(f"bands x rows must equal n_hashes ({bands} x {rows} != {n_hashes})")
        self.bands, self.rows, self.n_hashes, self.seed, self.shingle_k = bands, rows, n_hashes, seed, shingle_k

    def encode(self, texts: Sequence[str]) -> Encoded:
        return encode(texts, self.n_hashes, self.seed, self.shingle_k)

    # ---- index the training articles --------------------------------------------------------------------

    def fit(self, texts: Sequence[str] | None, labels, encoded: Encoded | None = None) -> "LSHNeighborIndex":
        """Index the training articles. Pass ``encoded`` (from :func:`encode`) to skip shingling."""
        enc = encoded if encoded is not None else self.encode(texts)
        self.labels_ = np.asarray(labels, dtype=float)
        if len(self.labels_) != len(enc):
            raise ValueError("labels and texts have different lengths")
        self.fallback_ = float(self.labels_.mean())   # prob_fake of uncovered articles: train fake rate

        # All training shingle sets in one flat array (CSR layout): article i is
        # flat[offsets[i]:offsets[i + 1]]. One array instead of 44k small ones makes scoring vectorizable.
        self.sizes_ = np.array([len(s) for s in enc.shingles], dtype=np.int64)
        self.offsets_ = np.concatenate([[0], np.cumsum(self.sizes_)])
        self.flat_ = np.concatenate(enc.shingles) if len(enc) else np.empty(0, np.int64)

        self.index_ = LSHIndex(self.bands, self.rows)
        for row in np.flatnonzero(enc.has_sig):        # articles without signature cannot be found
            self.index_.add(int(row), enc.signatures[row])
        return self

    # ---- retrieve and score -----------------------------------------------------------------------------

    def _jaccard_to_candidates(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """Exact Jaccard of query set ``q`` with each training article in ``cand`` (one vectorized pass)."""
        starts, sizes = self.offsets_[cand], self.sizes_[cand]
        # gather the candidates' shingles back to back, then mark which of them are in q
        idx = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, sizes)])
        hit = np.isin(self.flat_[idx], q, assume_unique=False)
        bounds = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        inter = np.add.reduceat(hit.astype(np.int64), bounds)   # |q ∩ c| per candidate
        return inter / (len(q) + sizes - inter)

    def _exact_jaccard_all(self, q: np.ndarray) -> np.ndarray:
        """Exact Jaccard of query set ``q`` with EVERY training article (reference without LSH).

        Uses a posting list: all training shingles sorted once, so the training articles containing
        a shingle are one contiguous slice (found with ``searchsorted``). Counting how often each
        article appears over the query's slices gives |q ∩ c| for all articles at once.
        """
        if not hasattr(self, "post_vals_"):
            order = np.argsort(self.flat_, kind="stable")
            self.post_vals_ = self.flat_[order]
            self.post_rows_ = np.repeat(np.arange(len(self.sizes_)), self.sizes_)[order]
        lo = np.searchsorted(self.post_vals_, q, side="left")
        hi = np.searchsorted(self.post_vals_, q, side="right")
        lengths = hi - lo
        # positions lo[0]..hi[0]-1, lo[1]..hi[1]-1, ... without a Python loop
        pos = np.repeat(lo - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(lengths.sum())
        inter = np.bincount(self.post_rows_[pos], minlength=len(self.sizes_))
        return inter / (len(q) + self.sizes_ - inter)

    def neighbors(self, texts: Sequence[str] | None = None, encoded: Encoded | None = None,
                  floor: float = SIM_FLOOR, exact: bool = False) -> Neighbors:
        """Candidates from the LSH index, scored with exact Jaccard; keeps those with similarity >= ``floor``.

        ``exact=True`` skips LSH and compares with every training article (the reference for what LSH
        misses); ``n_candidates`` then counts the training articles sharing at least one shingle.
        """
        enc = encoded if encoded is not None else self.encode(texts)
        rows, sims, n_cand = [], [], np.zeros(len(enc), dtype=np.int64)
        t_lookup = t_score = 0.0
        for i in range(len(enc)):
            if not enc.has_sig[i]:
                rows.append(np.empty(0, np.int64))
                sims.append(np.empty(0))
                continue
            t0 = time.perf_counter()
            if exact:
                all_sims = self._exact_jaccard_all(enc.shingles[i])
                cand = np.flatnonzero(all_sims > 0)
                t1 = time.perf_counter()
                n_cand[i] = len(cand)
                s = all_sims[cand]
            else:
                cand = np.fromiter(self.index_.query(enc.signatures[i]), dtype=np.int64)
                t1 = time.perf_counter()
                n_cand[i] = len(cand)
                s = self._jaccard_to_candidates(enc.shingles[i], cand) if len(cand) else np.empty(0)
            if len(cand):
                keep = s >= floor
                cand, s = cand[keep], s[keep]
                order = np.lexsort((cand, -s))              # most similar first; ties by row (deterministic)
                cand, s = cand[order], s[order]
            rows.append(cand)
            sims.append(s.astype(float) if len(cand) else np.empty(0))
            t_lookup += t1 - t0
            t_score += time.perf_counter() - t1
        return Neighbors(rows, sims, n_cand, t_lookup, t_score)

    # ---- vote -------------------------------------------------------------------------------------------

    def predict_proba(self, nb: Neighbors, k: int, min_sim: float) -> tuple[np.ndarray, np.ndarray]:
        """``prob_fake`` per query and ``covered`` (has at least one neighbour with similarity >= ``min_sim``).

        ``prob_fake`` is the similarity-weighted share of fake labels among the top ``k`` neighbours,
        or ``fallback_`` (the train fake rate) for uncovered queries.
        """
        prob = np.full(len(nb), self.fallback_)
        covered = np.zeros(len(nb), dtype=bool)
        for i, (r, s) in enumerate(zip(nb.rows, nb.sims)):
            keep = s >= min_sim             # sorted descending, so the kept ones come first
            r, s = r[keep][:k], s[keep][:k]
            if len(r):
                covered[i] = True
                # summed similarity of the fake neighbours / of all neighbours. (Not np.dot: it adds in a
                # different order than sum(), so all-fake neighbours could give 1.0000000000000002.)
                fake = self.labels_[r] == 1
                prob[i] = min(1.0, float(s[fake].sum() / s.sum()))
        return prob, covered


# ---- tuning (validation only) ---------------------------------------------------------------------------

K_GRID = (1, 3, 5, 10, 20, 50)
MIN_SIM_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def score(prob: np.ndarray, covered: np.ndarray, y: np.ndarray) -> dict:
    """Coverage (overall and per class) and the shared metrics on covered articles and on all articles.

    Main metrics: coverage, F1 on covered articles, ROC-AUC over all articles. "Overall" F1/accuracy
    (secondary) include the uncovered articles with the train-fake-rate fallback (< 0.5, so ``evaluate``
    counts them as predicted real).
    """
    from fake_news.evaluation import evaluate

    y = np.asarray(y).astype(int)
    out = {"coverage": covered.mean(), "coverage_fake": covered[y == 1].mean(),
           "coverage_real": covered[y == 0].mean()}
    if covered.any():
        cov = evaluate(y[covered], prob[covered])
        out.update({"f1_covered": cov["f1"], "accuracy_covered": cov["accuracy"]})
    else:
        out.update({"f1_covered": float("nan"), "accuracy_covered": float("nan")})
    overall = evaluate(y, prob)
    out.update({"f1_overall": overall["f1"], "accuracy_overall": overall["accuracy"], "roc_auc_overall": overall["roc_auc"]})
    return out


def tune(knn: LSHNeighborIndex, nb: Neighbors, y, ks=K_GRID, min_sims=MIN_SIM_GRID) -> "pd.DataFrame":
    """Score every (k, min_sim) on one set of neighbours (validation). One row per combination."""
    import pandas as pd

    rows = []
    for min_sim in min_sims:
        for k in ks:
            prob, covered = knn.predict_proba(nb, k, min_sim)
            rows.append({"k": k, "min_sim": min_sim, **score(prob, covered, y)})
    return pd.DataFrame(rows)


def best_setting(grid: "pd.DataFrame") -> "pd.Series":
    """The row with the highest ROC-AUC over all val articles; ties go to higher F1 on covered articles.

    AUC is the selection metric because it combines the two main metrics: how many articles get a
    neighbour (coverage) and how well those neighbours rank fake above real.
    """
    order = grid.sort_values(["roc_auc_overall", "f1_covered", "k", "min_sim"], ascending=[False, False, True, False])
    return order.iloc[0]
