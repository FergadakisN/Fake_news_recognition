"""LSH neighbour features: what the most similar *training* articles say about an article.

For every article (train, val and test) we look up similar training articles and summarize their labels
in three numbers that the final classifier uses as input features:

    fake_neighbor_share   similarity-weighted share of fake neighbours: sum(J_i * is_fake_i) / sum(J_i)
    max_jaccard           similarity of the closest neighbour (0 if none)
    neighbor_count        number of neighbours with J >= MIN_SIM

An article without neighbours gets ``fake_neighbor_share`` = fake rate of the training articles (the best
guess without evidence), ``max_jaccard`` = 0 and ``neighbor_count`` = 0.

How neighbours are found (the design chosen on random val; reasons in reports/minhash/LSH_FEATURES_DESIGN.md):

1. **Shingles.** Each article becomes the set of its 2-word shingles (crc32-hashed).
2. **Index (train only).** Every training article's 128-value MinHash signature goes into an LSH index with
   64 bands x 2 rows, which proposes pairs down to Jaccard ~0.1.
3. **Retrieve.** An article's signature is looked up: training articles sharing a band bucket are candidates.
4. **Prefilter.** Candidates whose signatures agree in fewer than ``MIN_AGREE`` (5) of the 128 positions are
   dropped. A pair with Jaccard J agrees in Binomial(128, J) positions, so a pair at J = 0.1 (expected 12.8)
   is dropped with probability < 1%; 99.9% of the 64 x 2 candidates are below 0.1 and this skips most of them.
5. **Score.** Exact Jaccard of the shingle sets; neighbours are the candidates with J >= ``MIN_SIM`` (0.1).

**Exclusion rule (``exclude``).** Training articles are themselves in the index, so a training article
would find itself (J = 1) and its near-duplicate copies. Its features must look like those of a val/test
article, which can never find itself, so:

- ``exclude="self"`` (random split): a training article never counts itself as a neighbour. Its copies in
  train stay, just as a val article in the random split can have copies in train.
- ``exclude="group"`` (grouped and cluster splits): a training article also ignores every article of its
  near-duplicate group, because in these splits a val/test article never has a group member in train.

The rule only applies to training articles; val/test articles are not in the index and always see the whole
training set. Val/test features use training labels only.

The second half of the module is the *neighbour voting analysis* (``predict_proba``, ``tune``): the
neighbour vote used on its own as a classifier, to measure how informative the neighbours are
(scripts/run_model_b.py). It is an analysis tool, not one of the pipeline's models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.similarity import minhash
from fake_news.similarity.lsh import LSHIndex
from fake_news.similarity.near_duplicates import MIN_SHINGLES
from fake_news.similarity.shingles import K, shingle_ids

# ---- the fixed design of the neighbour features -------------------------------------------------------------
SHINGLE_WORDS = 2
BANDS, ROWS = 64, 2
MIN_SIM = 0.1         # neighbours need exact Jaccard >= MIN_SIM
MIN_AGREE = 5         # prefilter: agreeing MinHash positions (of 128) needed before the exact check
SIM_FLOOR = MIN_SIM   # neighbours below this are never kept (also the lowest min_sim of the voting analysis)

FEATURE_COLUMNS = ["fake_neighbor_share", "max_jaccard", "neighbor_count"]
FEATURES_DIR = config.PROCESSED_DIR / "features"
EXCLUSION_RULES = ("self", "group")


def exclusion_for_regime(regime: str) -> str:
    """The exclusion rule a split regime needs: "self" for random, "group" for grouped and cluster."""
    return "self" if regime == "random" else "group"


# ---------------------------------------------------------------------------------------------------------
# Shingles + signatures
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Encoded:
    """Shingle sets and MinHash signatures of a list of texts (computed once, reused for every index)."""

    shingles: list[np.ndarray]   # sorted unique shingle ids per text
    signatures: np.ndarray       # (n_texts, n_hashes) int64; rows of texts without signature are -1
    has_sig: np.ndarray          # bool: text has >= MIN_SHINGLES shingles

    def __len__(self) -> int:
        return len(self.shingles)

    def take(self, rows: np.ndarray) -> "Encoded":
        """The texts at positions ``rows`` (e.g. only the training articles)."""
        return Encoded([self.shingles[i] for i in rows], self.signatures[rows], self.has_sig[rows])


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


# ---------------------------------------------------------------------------------------------------------
# The train-only neighbour index
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Neighbors:
    """Neighbours of each query: training row indices and exact Jaccard similarities, most similar first.

    Only neighbours with similarity >= ``floor`` are kept, so any threshold >= floor can be applied later
    without querying again.
    """

    rows: list[np.ndarray]
    sims: list[np.ndarray]
    n_candidates: np.ndarray     # LSH candidates per query (before prefilter and similarity check)
    lookup_seconds: float        # total time in the LSH index
    scoring_seconds: float       # total time for prefilter + exact Jaccard

    def __len__(self) -> int:
        return len(self.rows)


class LSHNeighborIndex:
    """LSH index of the training articles that returns, for any text, its similar training articles.

    >>> index = LSHNeighborIndex(64, 2, shingle_k=2).fit(train_texts, train_labels)
    >>> nb = index.neighbors(val_texts)                  # neighbours with exact Jaccard >= 0.1
    """

    def __init__(self, bands: int = BANDS, rows: int = ROWS, n_hashes: int = minhash.N_HASHES,
                 seed: int = config.SEED, shingle_k: int = SHINGLE_WORDS):
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
        self.fallback_ = float(self.labels_.mean())   # fake rate of the training articles

        # All training shingle sets in one flat array (CSR layout): article i is
        # flat[offsets[i]:offsets[i + 1]]. One array instead of 44k small ones makes scoring vectorizable.
        self.sizes_ = np.array([len(s) for s in enc.shingles], dtype=np.int64)
        self.offsets_ = np.concatenate([[0], np.cumsum(self.sizes_)])
        self.flat_ = np.concatenate(enc.shingles) if len(enc) else np.empty(0, np.int64)
        self.signatures_ = enc.signatures              # for the prefilter

        self.index_ = LSHIndex(self.bands, self.rows)
        for row in np.flatnonzero(enc.has_sig):        # articles without signature cannot be found
            self.index_.add(int(row), enc.signatures[row])
        return self

    # ---- retrieve and score -----------------------------------------------------------------------------

    def _jaccard_to_candidates(self, q: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """Exact Jaccard of the sorted query set ``q`` with each training article in ``cand``.

        The candidates' shingles are gathered back to back into one array ``g``; ``searchsorted`` finds
        where each of them would sit in ``q``, and it is in ``q`` if the value there is equal. Summing the
        hits per candidate gives |q ∩ c|.
        """
        starts, sizes = self.offsets_[cand], self.sizes_[cand]
        bounds = np.concatenate([[0], np.cumsum(sizes)[:-1]])            # where each candidate starts in g
        g = self.flat_[np.repeat(starts - bounds, sizes) + np.arange(sizes.sum())]
        pos = np.searchsorted(q, g).clip(max=len(q) - 1)
        inter = np.add.reduceat((q[pos] == g).astype(np.int64), bounds)  # |q ∩ c| per candidate
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
                  floor: float = SIM_FLOOR, exact: bool = False, min_agree: int = 0,
                  query_keys: np.ndarray | None = None, index_keys: np.ndarray | None = None) -> Neighbors:
        """Similar training articles of each query: LSH candidates scored with exact Jaccard (>= ``floor``).

        - ``min_agree``: drop candidates whose MinHash signatures agree in fewer positions before the
          exact check (0 = no prefilter).
        - ``query_keys`` / ``index_keys``: exclusion. A candidate is dropped when its key (``index_keys``,
          one per training article) equals the query's key. Keys are doc_ids for ``exclude="self"`` and
          group ids for ``exclude="group"``; a query with key -1 excludes nothing.
        - ``exact=True`` skips LSH and compares with every training article (the reference for what LSH
          misses); ``n_candidates`` then counts the training articles sharing at least one shingle.
        """
        enc = encoded if encoded is not None else self.encode(texts)
        rows, sims, n_cand = [], [], np.zeros(len(enc), dtype=np.int64)
        t_lookup = t_score = 0.0
        empty = (np.empty(0, np.int64), np.empty(0))
        for i in range(len(enc)):
            if not enc.has_sig[i]:          # too short for a signature: no neighbours
                rows.append(empty[0])
                sims.append(empty[1])
                continue
            t0 = time.perf_counter()
            if exact:
                all_sims = self._exact_jaccard_all(enc.shingles[i])
                cand = np.flatnonzero(all_sims > 0)
            else:
                cand = np.fromiter(self.index_.query(enc.signatures[i]), dtype=np.int64)
            t1 = time.perf_counter()
            n_cand[i] = len(cand)
            if query_keys is not None and query_keys[i] != -1 and len(cand):
                cand = cand[index_keys[cand] != query_keys[i]]                        # exclusion rule
            if min_agree and not exact and len(cand):
                agree = (self.signatures_[cand] == enc.signatures[i]).sum(axis=1)
                cand = cand[agree >= min_agree]                                        # prefilter
            if len(cand):
                s = all_sims[cand] if exact else self._jaccard_to_candidates(enc.shingles[i], cand)
                keep = s >= floor
                cand, s = cand[keep], s[keep]
                order = np.lexsort((cand, -s))              # most similar first; ties by row (deterministic)
                rows.append(cand[order])
                sims.append(s[order].astype(float))
            else:
                rows.append(empty[0])
                sims.append(empty[1])
            t_lookup += t1 - t0
            t_score += time.perf_counter() - t1
        return Neighbors(rows, sims, n_cand, t_lookup, t_score)

    # ---- features ---------------------------------------------------------------------------------------

    def features(self, nb: Neighbors, min_sim: float = MIN_SIM) -> pd.DataFrame:
        """The three neighbour features per query (see the module docstring), from neighbours >= ``min_sim``."""
        share = np.full(len(nb), self.fallback_)
        max_j = np.zeros(len(nb))
        count = np.zeros(len(nb), dtype=np.int64)
        for i, (r, s) in enumerate(zip(nb.rows, nb.sims)):
            keep = s >= min_sim
            r, s = r[keep], s[keep]
            if len(r):
                share[i] = _weighted_fake_share(s, self.labels_[r])
                max_j[i] = s[0]                     # sorted descending
                count[i] = len(r)
        return pd.DataFrame({"fake_neighbor_share": share, "max_jaccard": max_j, "neighbor_count": count})

    # ---- vote (neighbour voting analysis) ---------------------------------------------------------------

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
                prob[i] = _weighted_fake_share(s, self.labels_[r])
        return prob, covered


def _weighted_fake_share(sims: np.ndarray, labels: np.ndarray) -> float:
    """sum(sim * is_fake) / sum(sim). (Not np.dot: it adds in a different order than sum(), so all-fake
    neighbours could give 1.0000000000000002.)"""
    return min(1.0, float(sims[labels == 1].sum() / sims.sum()))


# ---------------------------------------------------------------------------------------------------------
# Building, saving and loading the features
# ---------------------------------------------------------------------------------------------------------

def build_lsh_features(df: pd.DataFrame, split: pd.Series, exclude: str, groups: pd.Series | None = None,
                       encoded: Encoded | None = None, min_agree: int = MIN_AGREE,
                       verbose: bool = False) -> pd.DataFrame:
    """Neighbour features of every article in ``df`` for one train/val/test split.

    ``df`` has ``doc_id, content, is_fake``; ``split`` maps doc_id -> train/val/test. The index holds the
    training articles only, so val/test features use training labels only. ``exclude`` is the exclusion
    rule for training articles ("self" or "group", see the module docstring); "group" needs ``groups``
    (doc_id -> near-duplicate group id; an article missing from it is its own group). ``encoded`` (all of
    ``df``, 2-word shingles) can be passed to skip shingling when building several splits.

    Returns ``doc_id, split`` and the three ``FEATURE_COLUMNS``, in the order of ``df``.
    """
    if exclude not in EXCLUSION_RULES:
        raise ValueError(f"exclude must be one of {EXCLUSION_RULES}, got {exclude!r}")
    if exclude == "group" and groups is None:
        raise ValueError('exclude="group" needs the near-duplicate groups')
    doc_ids = df["doc_id"].to_numpy()
    part = split.reindex(doc_ids).to_numpy()
    if pd.isna(part).any():
        raise ValueError("split has no partition for some articles")
    is_train = part == "train"
    train_rows = np.flatnonzero(is_train)

    t0 = time.perf_counter()
    enc = encoded if encoded is not None else encode(df["content"].tolist(), shingle_k=SHINGLE_WORDS)
    index = LSHNeighborIndex(BANDS, ROWS, shingle_k=SHINGLE_WORDS)
    index.fit(None, df["is_fake"].to_numpy()[train_rows], encoded=enc.take(train_rows))

    # Exclusion keys: doc_id ("self") or group id ("group"). Only training articles exclude anything (-1).
    if exclude == "self":
        keys = doc_ids.astype(np.int64)
    else:
        keys = pd.Series(doc_ids).map(groups).fillna(pd.Series(doc_ids)).to_numpy().astype(np.int64)
    query_keys = np.where(is_train, keys, -1)
    if verbose:
        print(f"  index of {len(train_rows):,} training articles built in {time.perf_counter() - t0:.0f} s; "
              f"querying {len(df):,} articles ...", flush=True)

    nb = index.neighbors(encoded=enc, min_agree=min_agree, query_keys=query_keys, index_keys=keys[train_rows])
    feats = index.features(nb)
    feats.insert(0, "split", part)
    feats.insert(0, "doc_id", doc_ids)
    feats.attrs.update({"fallback": index.fallback_, "seconds": round(time.perf_counter() - t0, 1),
                        "candidates_per_query": float(nb.n_candidates.mean()),
                        "ms_per_query": 1000 * (nb.lookup_seconds + nb.scoring_seconds) / max(len(nb), 1)})
    return feats


def features_path(regime: str, seed: int | None = None) -> Path:
    """data/processed/features/lsh__<regime>[__seed<N>].csv"""
    return FEATURES_DIR / (f"lsh__{regime}" + (f"__seed{seed}" if seed is not None else "") + ".csv")


def save_lsh_features(feats: pd.DataFrame, regime: str, seed: int | None = None) -> Path:
    path = features_path(regime, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")     # write-then-rename: never a half-written file
    feats.sort_values("doc_id").to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(path)
    return path


def load_lsh_features(regime: str, seed: int | None = None, check: bool = True) -> pd.DataFrame:
    """The LSH neighbour features of one split: ``doc_id, split, fake_neighbor_share, max_jaccard, neighbor_count``.

    For the shared splits (``seed=None``) the ``split`` column is checked against ``split_<regime>`` in
    splits.csv, so features built for an older version of the split are never used by mistake.
    Rebuild with ``python scripts/build_lsh_features.py --regime <regime>``.
    """
    path = features_path(regime, seed)
    rebuild = f"python scripts/build_lsh_features.py --regime {regime}" + (f" --seed {seed}" if seed is not None else "")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it with: {rebuild}")
    feats = pd.read_csv(path)
    if check and seed is None:
        from fake_news.data import load_splits

        splits = load_splits().set_index("doc_id")
        col = f"split_{regime}"
        if col not in splits.columns:
            raise KeyError(f"{col} is not in splits.csv")
        current = splits[col].reindex(feats["doc_id"]).to_numpy()
        if len(feats) != len(splits) or (current != feats["split"].to_numpy()).any():
            raise ValueError(f"{path.name} was built for a different version of {col}. Rebuild: {rebuild}")
    return feats


def sanity_report(feats: pd.DataFrame, labels: pd.Series, groups: pd.Series | None = None) -> pd.DataFrame:
    """Per partition: share without neighbours, max_jaccard, and fake_neighbor_share for fake vs real articles.

    Train and val should look alike (that is what the exclusion rule is for). With ``groups``, also the
    number of articles that have a member of their near-duplicate group in train (0 for val/test in the
    grouped split; for train articles, other members of their group are excluded under "group").
    """
    d = feats.assign(is_fake=labels.reindex(feats["doc_id"]).to_numpy())
    rows = []
    for part in config.SPLIT_NAMES:
        p = d[d["split"] == part]
        has = p["neighbor_count"] > 0
        row = {"split": part, "n": len(p), "share_no_neighbor": (~has).mean(),
               "mean_max_jaccard": p["max_jaccard"].mean(), "median_max_jaccard": p["max_jaccard"].median(),
               "mean_max_jaccard_if_any": p.loc[has, "max_jaccard"].mean(),
               "mean_neighbor_count": p["neighbor_count"].mean(),
               "share_fake_if_fake": p.loc[p.is_fake == 1, "fake_neighbor_share"].mean(),
               "share_fake_if_real": p.loc[p.is_fake == 0, "fake_neighbor_share"].mean(),
               "share_fake_if_fake_with_nb": p.loc[has & (p.is_fake == 1), "fake_neighbor_share"].mean(),
               "share_fake_if_real_with_nb": p.loc[has & (p.is_fake == 0), "fake_neighbor_share"].mean()}
        if groups is not None:
            g = groups.reindex(d["doc_id"]).to_numpy()
            train_groups = pd.Series(g[d["split"].to_numpy() == "train"]).value_counts()
            pg = groups.reindex(p["doc_id"]).to_numpy()
            # members of the article's group in train, not counting the article itself
            in_train = pd.Series(pg).map(train_groups).fillna(0).to_numpy() - (part == "train")
            row["with_group_member_in_train"] = int((in_train > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------------------------------------
# Neighbour voting analysis (validation only; not a pipeline model)
# ---------------------------------------------------------------------------------------------------------

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


def tune(index: LSHNeighborIndex, nb: Neighbors, y, ks=K_GRID, min_sims=MIN_SIM_GRID) -> pd.DataFrame:
    """Score every (k, min_sim) of the neighbour vote on one set of neighbours (validation)."""
    rows = []
    for min_sim in min_sims:
        for k in ks:
            prob, covered = index.predict_proba(nb, k, min_sim)
            rows.append({"k": k, "min_sim": min_sim, **score(prob, covered, y)})
    return pd.DataFrame(rows)


def best_setting(grid: pd.DataFrame) -> pd.Series:
    """The row with the highest ROC-AUC over all val articles; ties go to higher F1 on covered articles.

    AUC is the selection metric because it combines the two main metrics: how many articles get a
    neighbour (coverage) and how well those neighbours rank fake above real.
    """
    order = grid.sort_values(["roc_auc_overall", "f1_covered", "k", "min_sim"], ascending=[False, False, True, False])
    return order.iloc[0]
