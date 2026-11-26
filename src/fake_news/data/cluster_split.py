"""The ``cluster`` split regime: whole topic clusters held out, so val and test contain topics never seen in train.

Steps (used by ``scripts/build_cluster_split.py``):

1. **Topic clusters.** K-means on the sentence embeddings of *all* cleaned articles. This uses no labels, so it is
   the same situation as the near-duplicate groups: the groups exist before the split. (The K-means *inside* a model
   is still fitted on train only.)
2. **Keep near-duplicates together.** A near-duplicate group whose copies fall into different clusters is moved
   entirely into the cluster that most of its members belong to. Otherwise copies of one story would end up in
   different partitions and the hardest split would leak more than ``grouped``.
3. **Balanced assignment.** Whole clusters go to train / val / test with ``make_group_split``. Because a few dozen
   large groups can be filled in many orders, several seeds are tried and the split closest to the target fractions
   and to the overall fake share in every partition is kept. This uses only sizes and the overall label balance,
   never a model result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.data.split import make_group_split


def merge_near_duplicates(clusters: pd.Series, dup_groups: pd.Series) -> pd.Series:
    """Move every near-duplicate group into the cluster most of its members are in (ties: lowest cluster id).

    ``clusters`` and ``dup_groups`` map doc_id -> id. Articles without a group (or in a singleton group) keep their
    cluster. Returns a new Series doc_id -> cluster.
    """
    groups = dup_groups.reindex(clusters.index)
    sized = groups.dropna()
    sized = sized[sized.groupby(sized).transform("size") > 1]
    if sized.empty:
        return clusters.copy()
    members = pd.DataFrame({"group": sized, "cluster": clusters.loc[sized.index]})
    counts = members.groupby(["group", "cluster"]).size().rename("n").reset_index()
    counts = counts.sort_values(["group", "n", "cluster"], ascending=[True, False, True])
    majority = counts.drop_duplicates("group").set_index("group")["cluster"]
    out = clusters.copy()
    out.loc[sized.index] = sized.map(majority).to_numpy()
    return out


def balance_score(split: pd.Series, labels: pd.Series, fractions=config.SPLIT_FRACTIONS) -> float:
    """Largest deviation from the target fractions plus largest deviation of a partition's fake share from overall.

    0 is a perfect split; e.g. 0.03 means some partition is ~3 points off in size or fake share.
    """
    labels = labels.reindex(split.index)
    frac = split.value_counts(normalize=True).reindex(config.SPLIT_NAMES).fillna(0).to_numpy()
    fake = labels.groupby(split).mean().reindex(config.SPLIT_NAMES).to_numpy()
    return float(np.max(np.abs(frac - np.array(fractions))) + np.max(np.abs(fake - labels.mean())))


def best_cluster_split(clean: pd.DataFrame, clusters: pd.Series, n_seeds: int = 500,
                       min_clusters: int = 3) -> tuple[pd.Series, int, float]:
    """Try ``n_seeds`` assignment orders and keep the most balanced split.

    Splits where val or test hold fewer than ``min_clusters`` clusters are skipped (a test set of one or two topics
    says little). Returns (split, seed, balance score).
    """
    labels = clean.set_index("doc_id")["is_fake"]
    best = (None, -1, np.inf)
    for seed in range(n_seeds):
        split = make_group_split(clean, clusters, stratify=False, seed=seed)
        held_out = clusters.reindex(split.index)
        if min(held_out[split == p].nunique() for p in ("val", "test")) < min_clusters:
            continue
        score = balance_score(split, labels)
        if score < best[2]:
            best = (split, seed, score)
    if best[0] is None:
        raise ValueError(f"No seed gave >= {min_clusters} clusters in both val and test; use a larger K.")
    return best
