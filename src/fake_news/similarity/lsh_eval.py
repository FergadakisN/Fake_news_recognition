"""Evaluation of MinHash/LSH retrieval against exact Jaccard similarity (ground truth), and the leakage audit.

LSH is an approximation: it only proposes *candidate pairs*. To know what it misses we need the exact
Jaccard similarity of every pair, which is what LSH is built to avoid. On a sample (3,000 articles, 4.5
million pairs) and, for the high similarities that matter, on the full data, we compute it anyway with an
**inverted index (posting lists)**: all shingles of all articles sorted once, so the articles containing a
given shingle are one contiguous slice. For one article, counting how often every other article appears over
the slices of its shingles gives all intersections ``|A ∩ B|`` at once (``np.bincount``), and

    J(A, B) = |A ∩ B| / (|A| + |B| - |A ∩ B|).

This is exact (no hashing, no sampling); it is only slow because it still touches every article that shares
any shingle, including very common ones ("one of the").

Measures (per banding, per threshold t):

- **recall at J >= t**: share of the true pairs with J >= t that LSH proposes as candidates;
- **false-positive rate at t**: share of the candidates whose true J is below t. Candidates are verified with
  exact Jaccard afterwards, so false positives cost time, not quality;
- **candidates per article**: 2 x candidate pairs / articles (every pair involves two articles);
- **measured S-curve**: share of the pairs that are candidates, per bin of true J, next to the theoretical
  P(candidate | s) = 1 - (1 - s^rows)^bands.

Everything here uses NumPy and the standard library only, like the rest of the MinHash/LSH code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fake_news.similarity import lsh

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------------------------------------
# Exact Jaccard with an inverted index
# ---------------------------------------------------------------------------------------------------------

class PostingLists:
    """Inverted index of shingle sets: ``intersections(q)`` = ``|q ∩ S_i|`` for every set ``S_i`` at once."""

    def __init__(self, shingle_sets: list[np.ndarray]):
        self.sizes = np.array([len(s) for s in shingle_sets], dtype=np.int64)
        flat = np.concatenate(shingle_sets) if len(shingle_sets) else np.empty(0, np.int64)
        owner = np.repeat(np.arange(len(shingle_sets)), self.sizes)    # which set each shingle belongs to
        order = np.argsort(flat, kind="stable")
        self.values, self.owners = flat[order], owner[order]           # sorted shingles and their sets

    def intersections(self, q: np.ndarray) -> np.ndarray:
        """``|q ∩ S_i|`` for every indexed set ``S_i`` (``q``: sorted unique shingle ids)."""
        lo = np.searchsorted(self.values, q, side="left")
        hi = np.searchsorted(self.values, q, side="right")
        lengths = hi - lo
        # all positions lo[0]..hi[0]-1, lo[1]..hi[1]-1, ... (the slices of q's shingles) without a loop
        pos = np.repeat(lo - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(lengths.sum())
        return np.bincount(self.owners[pos], minlength=len(self.sizes))

    def jaccard(self, q: np.ndarray) -> np.ndarray:
        """Exact Jaccard of ``q`` with every indexed set."""
        inter = self.intersections(q)
        union = len(q) + self.sizes - inter
        return np.divide(inter, union, out=np.zeros(len(union)), where=union > 0)


def all_pairs_jaccard(shingle_sets: list[np.ndarray]) -> np.ndarray:
    """Exact Jaccard of every pair, as a dense ``(n, n)`` float32 matrix (small samples only: n^2 values)."""
    index = PostingLists(shingle_sets)
    out = np.empty((len(shingle_sets), len(shingle_sets)), dtype=np.float32)
    for i, q in enumerate(shingle_sets):
        out[i] = index.jaccard(q)
    return out


def pairs_above(shingle_sets: list[np.ndarray], min_j: float, verbose_every: int = 0) -> pd.DataFrame:
    """All pairs ``i < j`` with exact Jaccard >= ``min_j``: columns ``i, j, jaccard`` (for the full data)."""
    index = PostingLists(shingle_sets)
    parts = []
    for i, q in enumerate(shingle_sets):
        j_all = index.jaccard(q)
        j = np.flatnonzero(j_all >= min_j)
        j = j[j > i]                                 # each pair once
        if len(j):
            parts.append(np.column_stack([np.full(len(j), i), j, j_all[j]]))
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    {i + 1:,} / {len(shingle_sets):,}", flush=True)
    arr = np.concatenate(parts) if parts else np.empty((0, 3))
    return pd.DataFrame({"i": arr[:, 0].astype(np.int64), "j": arr[:, 1].astype(np.int64), "jaccard": arr[:, 2]})


# ---------------------------------------------------------------------------------------------------------
# Recall, false-positive rate, S-curve
# ---------------------------------------------------------------------------------------------------------

def pair_codes(i, j, n: int) -> np.ndarray:
    """One integer per unordered pair (min * n + max), so pair sets can be compared with ``np.isin``."""
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    return np.minimum(i, j) * n + np.maximum(i, j)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval of a proportion k/n (sensible also for k = n and small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def recall_table(truth_codes: np.ndarray, truth_j: np.ndarray, cand_codes: np.ndarray,
                 thresholds=THRESHOLDS) -> pd.DataFrame:
    """Per threshold t: true pairs with J >= t, how many LSH proposes, recall with 95% CI.

    ``truth_*``: codes and exact J of the true pairs (at least all pairs with J >= min(thresholds));
    ``cand_codes``: codes of the LSH candidate pairs.
    """
    found = np.isin(truth_codes, cand_codes)
    rows = []
    for t in thresholds:
        above = truth_j >= t
        n, k = int(above.sum()), int((found & above).sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"threshold": t, "true_pairs": n, "found": k, "recall": k / n if n else float("nan"),
                     "recall_ci_low": lo, "recall_ci_high": hi})
    return pd.DataFrame(rows)


def false_positive_table(cand_j: np.ndarray, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Per threshold t: share of the candidates whose exact J is below t."""
    n = len(cand_j)
    return pd.DataFrame([{"threshold": t, "candidates": n, "below_threshold": int((cand_j < t).sum()),
                          "false_positive_rate": float((cand_j < t).mean()) if n else float("nan")}
                         for t in thresholds])


def measured_s_curve(pair_j: np.ndarray, is_candidate: np.ndarray, bands: int, rows: int,
                     width: float = 0.05, min_pairs: int = 5) -> pd.DataFrame:
    """Share of pairs that are candidates, per bin of exact J, next to what the S-curve predicts for them.

    The prediction is the *mean* of P(candidate | J) over the pairs in the bin, not P at the bin centre: in
    the lowest bin almost all pairs have J close to 0, so the centre would overstate it. ``mean_j`` (the x
    position for plots) is the mean J of the pairs in the bin. Bins with fewer than ``min_pairs`` pairs are
    dropped (one or two pairs say nothing about a probability).
    """
    pair_j = np.asarray(pair_j, dtype=float)
    b = np.floor(np.clip(pair_j, 0, 1 - 1e-9) / width).astype(int)
    d = pd.DataFrame({"bin": b, "cand": is_candidate, "j": pair_j,
                      "p": lsh.candidate_probability(pair_j, bands, rows)})
    out = d.groupby("bin").agg(pairs=("cand", "size"), candidates=("cand", "sum"), mean_j=("j", "mean"),
                               theoretical=("p", "mean")).reset_index()
    out = out[out["pairs"] >= min_pairs].copy()
    out["j_low"] = out["bin"] * width
    out["center"] = out["j_low"] + width / 2
    out["measured"] = out["candidates"] / out["pairs"]
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(out["candidates"], out["pairs"])]
    out["ci_low"], out["ci_high"] = zip(*ci) if ci else ((), ())
    return out.drop(columns="bin").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------------------------------------

def subset_scores(feats: pd.DataFrame, y: np.ndarray, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """How ``fake_neighbor_share`` alone performs on subsets of articles (rows of ``feats``).

    Per subset: size, fake share, coverage (has a neighbour), F1 and accuracy on the covered articles
    (fake if share >= 0.5), and ROC-AUC over all articles of the subset (uncovered ones keep the
    train-fake-rate fallback). ROC-AUC is NaN if a subset holds only one class.
    """
    from fake_news.evaluation import evaluate

    y = np.asarray(y).astype(int)
    share = feats["fake_neighbor_share"].to_numpy()
    covered = feats["neighbor_count"].to_numpy() > 0
    rows = []
    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        row = {"subset": name, "n": int(m.sum()), "share_of_val": float(m.mean()),
               "fake_rate": float(y[m].mean()) if m.any() else float("nan"),
               "coverage": float(covered[m].mean()) if m.any() else float("nan")}
        c = m & covered
        if c.any():
            ev = evaluate(y[c], share[c])
            row.update({"f1_covered": ev["f1"], "accuracy_covered": ev["accuracy"]})
        if m.any() and len(np.unique(y[m])) == 2:
            row["roc_auc"] = evaluate(y[m], share[m])["roc_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def share_above(max_j: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """For each t in ``grid``: share of articles whose closest train neighbour has J >= t."""
    max_j = np.asarray(max_j)
    return np.array([(max_j >= t).mean() for t in grid])
