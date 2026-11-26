"""Tests for the LSH evaluation helpers (src/fake_news/similarity/lsh_eval.py)."""

import numpy as np
import pandas as pd
import pytest

from fake_news.similarity import lsh, lsh_eval
from fake_news.similarity.shingles import jaccard


def _random_sets(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 500, 60)
    sets = []
    for _ in range(n):
        # half of the sets copy part of a shared base, so there are pairs at many similarities
        own = rng.integers(0, 500, rng.integers(3, 40))
        take = base[: rng.integers(0, 60)] if rng.random() < 0.5 else np.empty(0, np.int64)
        sets.append(np.unique(np.concatenate([own, take]).astype(np.int64)))
    sets.append(np.empty(0, np.int64))   # an empty set: J = 0 with everything
    return sets


def test_posting_lists_equal_pairwise_jaccard():
    sets = _random_sets()
    J = lsh_eval.all_pairs_jaccard(sets)
    for i in range(len(sets)):
        for j in range(len(sets)):
            assert J[i, j] == pytest.approx(jaccard(sets[i], sets[j]) if (i != j or len(sets[i])) else 0.0, abs=1e-6)


def test_pairs_above_matches_dense_matrix():
    sets = _random_sets(seed=1)
    J = lsh_eval.all_pairs_jaccard(sets)
    got = lsh_eval.pairs_above(sets, 0.2)
    i, j = np.triu_indices(len(sets), k=1)
    want = {(a, b) for a, b in zip(i, j) if J[a, b] >= 0.2}
    assert set(zip(got["i"], got["j"])) == want
    assert (got["i"] < got["j"]).all()
    assert np.allclose(got["jaccard"], J[got["i"], got["j"]], atol=1e-6)


def test_pair_codes_are_order_free():
    assert lsh_eval.pair_codes([3], [1], 10)[0] == lsh_eval.pair_codes([1], [3], 10)[0] == 13


def test_recall_and_false_positive_tables():
    truth_codes = np.array([1, 2, 3, 4])
    truth_j = np.array([0.95, 0.75, 0.65, 0.55])
    cands = np.array([1, 3, 99])          # finds the 0.95 and 0.65 pairs, plus one pair not in the truth
    rec = lsh_eval.recall_table(truth_codes, truth_j, cands, thresholds=(0.5, 0.7, 0.9)).set_index("threshold")
    assert rec.loc[0.5, ["true_pairs", "found"]].tolist() == [4, 2]
    assert rec.loc[0.7, ["true_pairs", "found"]].tolist() == [2, 1]
    assert rec.loc[0.9, "recall"] == 1.0
    fp = lsh_eval.false_positive_table(np.array([0.95, 0.65, 0.1]), thresholds=(0.5, 0.7)).set_index("threshold")
    assert fp.loc[0.5, "false_positive_rate"] == pytest.approx(1 / 3)
    assert fp.loc[0.7, "false_positive_rate"] == pytest.approx(2 / 3)


def test_wilson_interval_contains_estimate():
    lo, hi = lsh_eval.wilson_interval(9, 10)
    assert lo < 0.9 < hi <= 1
    lo, hi = lsh_eval.wilson_interval(5, 5)
    assert hi == 1.0 and 0.4 < lo < 0.6
    assert np.isnan(lsh_eval.wilson_interval(0, 0)[0])


def test_measured_s_curve_uses_mean_prediction_of_the_pairs():
    j = np.array([0.01] * 50 + [0.92] * 10)
    cand = np.array([False] * 50 + [True] * 10)
    sc = lsh_eval.measured_s_curve(j, cand, bands=16, rows=8, min_pairs=5).set_index("j_low")
    assert sc.loc[0.0, "measured"] == 0 and sc.loc[0.9, "measured"] == 1
    assert sc.loc[0.0, "theoretical"] == pytest.approx(float(lsh.candidate_probability(0.01, 16, 8)))
    assert sc.loc[0.0, "mean_j"] == pytest.approx(0.01)


def test_subset_scores_coverage_and_auc():
    feats = pd.DataFrame({"fake_neighbor_share": [0.9, 0.8, 0.1, 0.2, 0.44, 0.44],
                          "neighbor_count": [2, 1, 1, 3, 0, 0]})
    y = np.array([1, 1, 0, 0, 1, 0])
    out = lsh_eval.subset_scores(feats, y, {"all": np.ones(6, bool), "covered": feats["neighbor_count"] > 0,
                                            "only fake": y == 1}).set_index("subset")
    assert out.loc["all", "coverage"] == pytest.approx(4 / 6)
    assert out.loc["covered", "f1_covered"] == 1.0
    assert out.loc["covered", "roc_auc"] == 1.0
    assert np.isnan(out.loc["only fake", "roc_auc"])        # one class: no AUC


def test_share_above():
    assert lsh_eval.share_above(np.array([0.0, 0.2, 0.6, 0.9]), np.array([0.1, 0.5, 1.0])).tolist() == [0.75, 0.5, 0.0]
