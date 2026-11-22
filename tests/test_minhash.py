"""Unit tests for fake_news.similarity: shingles, Jaccard, MinHash, LSH and grouping (toy data only)."""

import numpy as np
import pandas as pd
import pytest

from fake_news.similarity import lsh, minhash
from fake_news.similarity import near_duplicates as nd
from fake_news.similarity.groups import UnionFind, build_groups
from fake_news.similarity.shingles import jaccard, shingle_ids

A, B = minhash.make_hash_params()

# Two texts that share long stretches (an article and an edited copy) and an unrelated one.
BASE = ("the senate passed the new tax bill late on tuesday after a long debate about the deficit "
        "and the effect of the cuts on middle class families across the country while critics said "
        "the plan would mostly help large companies and the richest households in the long run")
EDITED = BASE.replace("late on tuesday", "early on wednesday").replace("critics said", "opponents argued")
OTHER = ("heavy rain flooded several streets in the old town on sunday and the fire brigade pumped water "
         "out of dozens of basements while the river reached its highest level in ten years")


# ---- shingles and Jaccard -------------------------------------------------------------------------------

def test_shingles_toy_sentence():
    s = shingle_ids("trump signs new tax bill today")  # 6 words -> 4 shingles of 3 words
    assert len(s) == 4
    assert s.dtype == np.int64


def test_shingles_are_deterministic_and_ignore_punctuation():
    assert np.array_equal(shingle_ids("Tax bill, signed today."), shingle_ids("tax bill signed today"))
    assert len(shingle_ids("two words")) == 0  # fewer than k words -> no shingles


def test_repeated_shingles_counted_once():
    assert len(shingle_ids("a b c a b c a b c")) == 3  # "a b c", "b c a", "c a b"


def test_jaccard_known_pair():
    assert jaccard(np.array([1, 2, 3]), np.array([2, 3, 4])) == 0.5  # |{2,3}| / |{1,2,3,4}|
    assert jaccard(np.array([1, 2]), np.array([1, 2])) == 1.0
    assert jaccard(np.array([1, 2]), np.array([3, 4])) == 0.0
    assert jaccard(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) == 0.0


# ---- MinHash --------------------------------------------------------------------------------------------

def test_hash_params_reproducible():
    a1, b1 = minhash.make_hash_params(64, seed=1)
    a2, b2 = minhash.make_hash_params(64, seed=1)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert (a1 > 0).all() and (a1 < minhash.P).all() and (b1 >= 0).all() and (b1 < minhash.P).all()


def test_no_overflow_with_large_shingle_ids():
    # Ids up to 2**32 - 1 (crc32 range) are reduced mod P first; the result must stay in [0, P).
    sig = minhash.signature(np.array([2**32 - 1, 2**31, 12345], dtype=np.int64), A, B)
    assert (sig >= 0).all() and (sig < minhash.P).all()


def test_minhash_estimate_close_to_exact_jaccard():
    universe = np.arange(10_000)
    for overlap in (100, 400, 700, 900):          # J = overlap / (2000 - overlap)
        a = universe[:1000]
        b = universe[1000 - overlap:2000 - overlap]
        exact = jaccard(a, b)
        est = minhash.estimate_jaccard(minhash.signature(a, A, B), minhash.signature(b, A, B))
        # standard error sqrt(J(1-J)/128) <= 0.045; allow 3 standard errors
        assert abs(est - exact) < 0.135, (overlap, exact, est)
    # and on real-looking text
    sa, sb = shingle_ids(BASE), shingle_ids(EDITED)
    est = minhash.estimate_jaccard(minhash.signature(sa, A, B), minhash.signature(sb, A, B))
    assert abs(est - jaccard(sa, sb)) < 0.135


def test_batch_signatures_equal_single_signatures():
    rng = np.random.default_rng(1)
    sets = [np.unique(rng.integers(0, 2**32, size=rng.integers(1, 300))) for _ in range(50)]
    batch = minhash.signatures(sets, A, B, batch_shingles=500)  # small batches: exercises the batching
    single = np.array([minhash.signature(s, A, B) for s in sets])
    assert np.array_equal(batch, single)


def test_empty_set_has_no_signature():
    with pytest.raises(ValueError):
        minhash.signature(np.array([], dtype=np.int64), A, B)


# ---- LSH ------------------------------------------------------------------------------------------------

def _sigs(texts):
    return minhash.signatures([shingle_ids(t) for t in texts], A, B)


def test_identical_texts_always_candidates():
    sigs = _sigs([BASE, OTHER, BASE, OTHER + " and more rain is expected", BASE])
    pairs = {tuple(p) for p in lsh.candidate_pairs(sigs)}
    assert {(0, 2), (0, 4), (2, 4)} <= pairs     # identical signatures share every bucket
    assert (0, 1) not in pairs                   # unrelated texts


def test_candidate_pairs_sorted_unique():
    sigs = _sigs([BASE, BASE, BASE])
    pairs = lsh.candidate_pairs(sigs)
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]  # once each, although they collide in all 16 bands


def test_index_query_finds_near_duplicate_only():
    index = lsh.LSHIndex()
    index.add("base", _sigs([BASE])[0])
    index.add("other", _sigs([OTHER])[0])
    assert index.query(_sigs([BASE])[0]) == {"base"}
    assert "other" not in index.query(_sigs([EDITED])[0])
    assert index.n_items == 2


def test_index_rejects_wrong_signature_length():
    with pytest.raises(ValueError):
        lsh.LSHIndex(bands=16, rows=8).add("x", np.zeros(100, dtype=np.int64))


def test_s_curve():
    p = lsh.candidate_probability([0.0, 0.5, 0.9, 1.0])
    assert p[0] == 0 and p[-1] == 1 and p[1] < 0.1 and p[2] > 0.99


# ---- union-find and groups ------------------------------------------------------------------------------

def test_union_find_chaining():
    uf = UnionFind(["A", "B", "C", "D"])
    uf.union("A", "B")
    uf.union("B", "C")                           # A ~ B and B ~ C -> one group, although A and C never met
    assert uf.find("A") == uf.find("C")
    assert uf.find("D") != uf.find("A")


def test_build_groups_threshold_and_ids():
    pairs = np.array([[10, 11], [11, 12], [13, 14]])
    groups = build_groups([10, 11, 12, 13, 14, 15], pairs, np.array([0.9, 0.85, 0.5]), threshold=0.8)
    assert groups.to_dict() == {10: 10, 11: 10, 12: 10, 13: 13, 14: 14, 15: 15}  # group id = smallest doc_id


def test_pipeline_on_toy_frame(tmp_path):
    df = pd.DataFrame({
        "doc_id": [5, 7, 9, 11, 13],
        "content": [BASE, EDITED, OTHER, BASE + " the end", "too short"],
        "is_fake": [1, 1, 0, 1, 0],
    })
    table = nd.compute_signatures(df, cache=tmp_path / "sigs.npz")
    assert list(table.short_doc_ids) == [13]     # 2 words -> no signature, stays a singleton
    cached = nd.compute_signatures(df, cache=tmp_path / "sigs.npz")
    assert cached.from_cache and np.array_equal(cached.signatures, table.signatures)

    cands = nd.verify_candidates(nd.find_candidates(table), df)
    groups = nd.make_groups(df["doc_id"], cands, threshold=0.7)
    assert groups[5] == groups[11] and groups[9] == 9 and groups[13] == 13
    summary = nd.group_summary(groups, df.set_index("doc_id")["is_fake"])
    assert summary["mixed_label_groups"] == 0 and summary["largest_group"] >= 2
