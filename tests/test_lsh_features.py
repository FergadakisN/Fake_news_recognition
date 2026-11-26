"""Unit tests for fake_news.similarity.lsh_features: the LSH neighbour query path and the vote, on toy texts."""

import numpy as np
import pytest

from fake_news.similarity import lsh_features as lf

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
    return lf.LSHNeighborIndex(bands, rows).fit(TRAIN, LABELS)


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
    knn = lf.LSHNeighborIndex(64, 2).fit(TRAIN, LABELS)
    edited = TRAIN[0].replace("late on tuesday", "early on wednesday")
    lsh_nb, exact_nb = knn.neighbors([edited]), knn.neighbors([edited], exact=True)
    assert lsh_nb.rows[0][0] == exact_nb.rows[0][0] == 0
    assert np.isclose(lsh_nb.sims[0][0], exact_nb.sims[0][0])


def test_vote_is_similarity_weighted_and_respects_k_and_min_sim():
    knn = lf.LSHNeighborIndex(64, 2).fit(TRAIN, LABELS)
    nb = lf.Neighbors(rows=[np.array([0, 1, 3])], sims=[np.array([0.6, 0.3, 0.15])],
                      n_candidates=np.array([3]), lookup_seconds=0.0, scoring_seconds=0.0)
    prob, _ = knn.predict_proba(nb, k=10, min_sim=0.1)      # fake, real, fake
    assert np.isclose(prob[0], (0.6 + 0.15) / (0.6 + 0.3 + 0.15))
    prob, _ = knn.predict_proba(nb, k=2, min_sim=0.1)       # top 2 only
    assert np.isclose(prob[0], 0.6 / 0.9)
    prob, covered = knn.predict_proba(nb, k=10, min_sim=0.7)  # nothing similar enough
    assert not covered[0] and prob[0] == knn.fallback_


def test_all_fake_neighbours_give_exactly_one():
    knn = lf.LSHNeighborIndex(64, 2).fit(TRAIN, [1, 1, 1, 1])
    sims = np.array([0.7, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # rounding would give > 1 here
    nb = lf.Neighbors(rows=[np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])], sims=[sims],
                      n_candidates=np.array([10]), lookup_seconds=0.0, scoring_seconds=0.0)
    prob, _ = knn.predict_proba(nb, k=10, min_sim=0.1)
    assert prob[0] == 1.0


def test_fallback_is_train_fake_rate():
    knn = lf.LSHNeighborIndex(64, 2).fit(TRAIN, LABELS)
    assert knn.fallback_ == np.mean(LABELS)
    prob, covered = knn.predict_proba(knn.neighbors([UNRELATED]), k=5, min_sim=0.1)
    assert not covered[0] and prob[0] == 0.5   # 2 of 4 toy training articles are fake


def test_shingle_size_is_used():
    knn = lf.LSHNeighborIndex(64, 2, shingle_k=2).fit(TRAIN, LABELS)
    assert len(knn.encode(["one two three four"]).shingles[0]) == 3   # 2-word shingles


def test_bands_times_rows_must_match_signature_length():
    with pytest.raises(ValueError):
        lf.LSHNeighborIndex(bands=10, rows=10)


def test_tune_and_best_setting():
    knn = lf.LSHNeighborIndex(64, 2).fit(TRAIN, LABELS)
    nb = knn.neighbors(TRAIN + [UNRELATED])
    grid = lf.tune(knn, nb, LABELS + [0], ks=(1, 3), min_sims=(0.1, 0.5))
    assert len(grid) == 4 and grid["coverage"].max() == 0.8   # 4 of 5 queries are in the index
    best = lf.best_setting(grid)
    assert best["roc_auc_overall"] == grid["roc_auc_overall"].max()


# ---- neighbour features ---------------------------------------------------------------------------------

import pandas as pd  # noqa: E402

STORY = TRAIN[0]
STORY_COPY = STORY + " and the vote was close"            # near-duplicate of STORY (same group)
STORY_VAL = STORY.replace("late on tuesday", "on tuesday night")


def _toy():
    """6 articles: a story and its copy in train (one group), two other train articles, two val articles."""
    df = pd.DataFrame({
        "doc_id": [10, 11, 12, 13, 20, 21],
        "content": [STORY, STORY_COPY, TRAIN[1], TRAIN[2], STORY_VAL, UNRELATED],
        "is_fake": [1, 1, 0, 0, 1, 0],
    })
    split = pd.Series(["train", "train", "train", "train", "val", "test"], index=df["doc_id"])
    groups = pd.Series([10, 10, 12, 13, 20, 21], index=df["doc_id"])  # 10 and 11 are one group
    return df, split, groups


def test_article_is_never_its_own_neighbour():
    df, split, groups = _toy()
    enc = lf.encode(df["content"].tolist(), shingle_k=lf.SHINGLE_WORDS)
    train_rows = np.flatnonzero(split.to_numpy() == "train")
    index = lf.LSHNeighborIndex().fit(None, df["is_fake"].to_numpy()[train_rows], encoded=enc.take(train_rows))
    keys = df["doc_id"].to_numpy()
    nb = index.neighbors(encoded=enc.take(train_rows), query_keys=keys[train_rows], index_keys=keys[train_rows])
    for query_row, found in enumerate(nb.rows):
        assert query_row not in found                      # train row i never finds train row i
    assert 1 in nb.rows[0] and 0 in nb.rows[1]             # but the copy is still found (exclude="self")


def test_self_exclusion_keeps_copies_group_exclusion_removes_them():
    df, split, groups = _toy()
    by_id = lambda f: f.set_index("doc_id")
    self_ = by_id(lf.build_lsh_features(df, split, exclude="self"))
    group = by_id(lf.build_lsh_features(df, split, exclude="group", groups=groups))
    assert self_.at[10, "neighbor_count"] == 1 and self_.at[10, "max_jaccard"] > 0.5   # finds its copy 11
    assert group.at[10, "neighbor_count"] == 0 and group.at[11, "neighbor_count"] == 0  # copy is in its group
    # val/test articles are not affected by the rule (they are never in the index)
    assert self_.loc[[20, 21]].equals(group.loc[[20, 21]])


def test_group_mode_never_returns_a_group_member():
    df, split, groups = _toy()
    enc = lf.encode(df["content"].tolist(), shingle_k=lf.SHINGLE_WORDS)
    train_rows = np.flatnonzero(split.to_numpy() == "train")
    index = lf.LSHNeighborIndex().fit(None, df["is_fake"].to_numpy()[train_rows], encoded=enc.take(train_rows))
    g = groups.reindex(df["doc_id"]).to_numpy()[train_rows]
    nb = index.neighbors(encoded=enc.take(train_rows), query_keys=g, index_keys=g)
    for query_row, found in enumerate(nb.rows):
        assert not (g[found] == g[query_row]).any()


def test_no_neighbour_gets_fallback_values():
    df, split, groups = _toy()
    f = lf.build_lsh_features(df, split, exclude="self").set_index("doc_id")
    train_rate = df.loc[split.to_numpy() == "train", "is_fake"].mean()      # 0.5 in the toy data
    assert f.at[21, "neighbor_count"] == 0 and f.at[21, "max_jaccard"] == 0
    assert f.at[21, "fake_neighbor_share"] == train_rate
    assert f.at[20, "neighbor_count"] >= 1 and f.at[20, "fake_neighbor_share"] == 1.0   # finds the fake story


def test_val_test_features_use_train_labels_only():
    df, split, groups = _toy()
    a = lf.build_lsh_features(df, split, exclude="group", groups=groups)
    flipped = df.assign(is_fake=np.where(split.to_numpy() == "train", df["is_fake"], 1 - df["is_fake"]))
    b = lf.build_lsh_features(flipped, split, exclude="group", groups=groups)
    assert a.equals(b)                                      # changing val/test labels changes nothing


def test_prefilter_keeps_true_neighbours():
    df, split, groups = _toy()
    a = lf.build_lsh_features(df, split, exclude="self", min_agree=0)
    b = lf.build_lsh_features(df, split, exclude="self", min_agree=lf.MIN_AGREE)
    assert a.equals(b)


def test_exclusion_rule_per_regime_and_bad_rule():
    assert lf.exclusion_for_regime("random") == "self"
    assert lf.exclusion_for_regime("grouped") == lf.exclusion_for_regime("cluster") == "group"
    df, split, _ = _toy()
    with pytest.raises(ValueError):
        lf.build_lsh_features(df, split, exclude="group")          # groups missing
    with pytest.raises(ValueError):
        lf.build_lsh_features(df, split, exclude="nothing")


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(lf, "FEATURES_DIR", tmp_path)
    df, split, _ = _toy()
    f = lf.build_lsh_features(df, split, exclude="self")
    lf.save_lsh_features(f, "random", seed=7)
    back = lf.load_lsh_features("random", seed=7)
    assert list(back.columns) == ["doc_id", "split", *lf.FEATURE_COLUMNS]
    assert np.allclose(back[lf.FEATURE_COLUMNS], f.sort_values("doc_id")[lf.FEATURE_COLUMNS])
    with pytest.raises(FileNotFoundError):
        lf.load_lsh_features("grouped", seed=7)
