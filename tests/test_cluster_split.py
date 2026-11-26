"""Unit tests for the cluster split regime (fake_news.data.cluster_split)."""

import numpy as np
import pandas as pd
import pytest

from fake_news.data.cluster_split import balance_score, best_cluster_split, merge_near_duplicates


def test_near_duplicate_group_moves_to_its_majority_cluster():
    clusters = pd.Series([0, 0, 1, 1, 2], index=[10, 11, 12, 13, 14])
    dup = pd.Series([10, 10, 10, 13, 14], index=[10, 11, 12, 13, 14])   # group 10 = {10, 11, 12}; 13, 14 alone
    out = merge_near_duplicates(clusters, dup)
    assert out.tolist() == [0, 0, 0, 1, 2]                               # doc 12 joins cluster 0 (2 of 3 members)


def test_tie_goes_to_the_lowest_cluster_id():
    clusters = pd.Series([3, 1], index=[1, 2])
    dup = pd.Series([1, 1], index=[1, 2])
    assert merge_near_duplicates(clusters, dup).tolist() == [1, 1]


def test_articles_without_groups_keep_their_cluster():
    clusters = pd.Series([0, 1, 2], index=[1, 2, 3])
    assert merge_near_duplicates(clusters, pd.Series([1, 2, 3], index=[1, 2, 3])).equals(clusters)


def test_balance_score_is_zero_for_a_perfect_split():
    # 70 / 15 / 15 articles, 40% fake in every partition
    split = pd.Series(["train"] * 70 + ["val"] * 15 + ["test"] * 15)
    labels = pd.Series([1] * 28 + [0] * 42 + [1] * 6 + [0] * 9 + [1] * 6 + [0] * 9)
    assert balance_score(split, labels) == pytest.approx(0)


def test_balance_score_adds_size_and_label_deviations():
    # sizes 60 / 20 / 20 (largest size deviation 0.10); test is 60% fake vs 40% overall (largest label deviation 0.20)
    split = pd.Series(["train"] * 60 + ["val"] * 20 + ["test"] * 20)
    labels = pd.Series([1] * 23 + [0] * 37 + [1] * 5 + [0] * 15 + [1] * 12 + [0] * 8)
    assert labels.mean() == pytest.approx(0.4)
    assert balance_score(split, labels) == pytest.approx(0.10 + 0.20)


def test_best_split_keeps_clusters_whole_and_is_balanced():
    rng = np.random.default_rng(0)
    n, k = 4000, 40
    clusters = pd.Series(rng.integers(0, k, n), index=np.arange(n))
    fake_rate = rng.uniform(0.1, 0.9, k)
    clean = pd.DataFrame({"doc_id": np.arange(n), "is_fake": (rng.random(n) < fake_rate[clusters]).astype(int)})
    split, seed, score = best_cluster_split(clean, clusters, n_seeds=50)
    assert (split.groupby(clusters).nunique() == 1).all()                # no cluster spread over partitions
    assert score < 0.05
    frac = split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.05 and abs(frac["test"] - 0.15) < 0.05
    assert min(clusters[split == p].nunique() for p in ("val", "test")) >= 3


def test_too_few_clusters_raises():
    clean = pd.DataFrame({"doc_id": range(100), "is_fake": [0, 1] * 50})
    clusters = pd.Series([0] * 50 + [1] * 50, index=range(100))
    with pytest.raises(ValueError):
        best_cluster_split(clean, clusters, n_seeds=5)
