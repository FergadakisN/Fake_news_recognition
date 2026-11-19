"""Unit tests for Model B (MinHash/LSH k-NN): the query path and the vote, on toy texts."""

import numpy as np
import pytest

from fake_news.models import model_b as mb

TRAIN = [
    "the senate passed the new tax bill late on tuesday after a long debate about the deficit and the cuts",
    "heavy rain flooded several streets in the old town on sunday and the fire brigade pumped out basements",
    "the central bank kept interest rates unchanged on thursday and said inflation was slowly coming down",
    "scientists found a new species of frog in the rain forest during a survey of the remote mountain region",
]
LABELS = [1, 0, 0, 1]
UNRELATED = "a recipe for apple pie needs flour butter sugar cinnamon and six large apples from the orchard"


@pytest.fixture(params=[(16, 8), (32, 4), (64, 2)])
def knn(request):
    bands, rows = request.param
    return mb.MinHashKNN(bands, rows).fit(TRAIN, LABELS)


def test_indexed_article_finds_itself(knn):
    nb = knn.neighbors([TRAIN[2]])
    assert nb.rows[0][0] == 2 and nb.sims[0][0] == 1.0   # itself, first, with similarity 1
    prob, covered = knn.predict_proba(nb, k=1, min_sim=0.5)
    assert covered[0] and prob[0] == LABELS[2]


def test_unrelated_text_finds_nothing(knn):
    nb = knn.neighbors([UNRELATED])
    assert len(nb.rows[0]) == 0
    prob, covered = knn.predict_proba(nb, k=10, min_sim=0.1)
    assert not covered[0] and prob[0] == knn.fallback_


def test_short_text_is_not_covered(knn):
    nb = knn.neighbors(["too short"])                      # < 5 shingles: no signature, no lookup
    assert nb.n_candidates[0] == 0
    assert not knn.predict_proba(nb, k=5, min_sim=0.1)[1][0]


def test_exact_search_agrees_with_lsh_on_near_copy():
    knn = mb.MinHashKNN(64, 2).fit(TRAIN, LABELS)
    edited = TRAIN[0].replace("late on tuesday", "early on wednesday")
    lsh_nb, exact_nb = knn.neighbors([edited]), knn.neighbors([edited], exact=True)
    assert lsh_nb.rows[0][0] == exact_nb.rows[0][0] == 0
    assert np.isclose(lsh_nb.sims[0][0], exact_nb.sims[0][0])


def test_vote_is_similarity_weighted_and_respects_k_and_min_sim():
    knn = mb.MinHashKNN(64, 2).fit(TRAIN, LABELS)
    nb = mb.Neighbors(rows=[np.array([0, 1, 3])], sims=[np.array([0.6, 0.3, 0.15])],
                      n_candidates=np.array([3]), lookup_seconds=0.0, scoring_seconds=0.0)
    prob, _ = knn.predict_proba(nb, k=10, min_sim=0.1)      # fake, real, fake
    assert np.isclose(prob[0], (0.6 + 0.15) / (0.6 + 0.3 + 0.15))
    prob, _ = knn.predict_proba(nb, k=2, min_sim=0.1)       # top 2 only
    assert np.isclose(prob[0], 0.6 / 0.9)
    prob, covered = knn.predict_proba(nb, k=10, min_sim=0.7)  # nothing similar enough
    assert not covered[0] and prob[0] == knn.fallback_


def test_all_fake_neighbours_give_exactly_one():
    knn = mb.MinHashKNN(64, 2).fit(TRAIN, [1, 1, 1, 1])
    sims = np.array([0.7, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # rounding would give > 1 here
    nb = mb.Neighbors(rows=[np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])], sims=[sims],
                      n_candidates=np.array([10]), lookup_seconds=0.0, scoring_seconds=0.0)
    prob, _ = knn.predict_proba(nb, k=10, min_sim=0.1)
    assert prob[0] == 1.0


def test_fallback_is_train_fake_rate():
    knn = mb.MinHashKNN(64, 2).fit(TRAIN, LABELS)
    assert knn.fallback_ == np.mean(LABELS)
    prob, covered = knn.predict_proba(knn.neighbors([UNRELATED]), k=5, min_sim=0.1)
    assert not covered[0] and prob[0] == 0.5   # 2 of 4 toy training articles are fake


def test_shingle_size_is_used():
    knn = mb.MinHashKNN(64, 2, shingle_k=2).fit(TRAIN, LABELS)
    assert len(knn.encode(["one two three four"]).shingles[0]) == 3   # 2-word shingles


def test_bands_times_rows_must_match_signature_length():
    with pytest.raises(ValueError):
        mb.MinHashKNN(bands=10, rows=10)


def test_tune_and_best_setting():
    knn = mb.MinHashKNN(64, 2).fit(TRAIN, LABELS)
    nb = knn.neighbors(TRAIN + [UNRELATED])
    grid = mb.tune(knn, nb, LABELS + [0], ks=(1, 3), min_sims=(0.1, 0.5))
    assert len(grid) == 4 and grid["coverage"].max() == 0.8   # 4 of 5 queries are in the index
    best = mb.best_setting(grid)
    assert best["roc_auc_overall"] == grid["roc_auc_overall"].max()
