"""Unit tests for fake_news.similarity.paraphrases (embedding tier of the duplicate groups), on toy vectors."""

import numpy as np
import pandas as pd

from fake_news.similarity import paraphrases as pp


def _unit(rows):
    X = np.asarray(rows, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_topk_matches_brute_force_and_excludes_self():
    rng = np.random.default_rng(0)
    X = _unit(rng.normal(size=(300, 16)))
    idx, sim = pp.embedding_topk(X, k=4, chunk=64)          # several blocks, last one partial
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    assert np.allclose(np.sort(S, axis=1)[:, ::-1][:, :4], sim, atol=1e-6)
    assert not (idx == np.arange(300)[:, None]).any()       # never its own neighbour
    assert (np.diff(sim, axis=1) <= 0).all()                # most similar first


def test_pairs_threshold_and_dedup():
    X = _unit([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0, 1]])
    ids = np.array([10, 11, 12, 13])
    idx, sim = pp.embedding_topk(X, k=1)
    pairs = pp.embedding_pairs(ids, idx, sim, threshold=0.9)
    assert pairs[["doc_a", "doc_b"]].values.tolist() == [[10, 11]]   # found from both sides, kept once


def test_merge_groups_joins_tiers_by_chaining():
    minhash = pd.Series({1: 1, 2: 1, 3: 3, 4: 4, 5: 5})                  # MinHash: {1, 2}
    pairs = pd.DataFrame({"doc_a": [2, 4], "doc_b": [3, 5]})            # embeddings: 2~3, 4~5
    merged = pp.merge_groups([1, 2, 3, 4, 5], minhash, pairs)
    assert merged.to_dict() == {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}            # {1,2,3} via 2~3, {4,5}


def test_two_tier_groups_record_the_linking_tier():
    # MinHash: {1, 2} and {6, 7}; embeddings: 2~3 (enlarges {1,2}), 4~5 (new group), 6~7 (already MinHash)
    minhash = pd.Series({1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 8})
    pairs = pd.DataFrame({"doc_a": [2, 4, 6], "doc_b": [3, 5, 7]})
    g = pp.two_tier_groups([1, 2, 3, 4, 5, 6, 7, 8], minhash, pairs).set_index("doc_id")
    assert g.loc[[1, 2, 3], "tier"].eq("both").all() and g.loc[[1, 2, 3], "group_id"].eq(1).all()
    assert g.loc[[4, 5], "tier"].eq("embedding").all()
    assert g.loc[[6, 7], "tier"].eq("minhash").all()      # embedding pair inside a MinHash group changes nothing
    assert g.at[8, "tier"] == "none" and g.at[8, "group_id"] == 8


def test_two_tier_joining_two_minhash_groups_is_both():
    minhash = pd.Series({1: 1, 2: 1, 3: 3, 4: 3})
    pairs = pd.DataFrame({"doc_a": [2], "doc_b": [3]})
    g = pp.two_tier_groups([1, 2, 3, 4], minhash, pairs)
    assert g["tier"].eq("both").all() and g["group_id"].eq(1).all()


def test_link_type_separates_copies_from_paraphrases():
    pairs = pd.DataFrame({
        "same_minhash_group": [True, False, False, False, False],
        "containment":        [0.95, 0.95, 0.90, 0.50, 0.10],
        "n_words_a":          [300, 1600, 300, 300, 300],
        "n_words_b":          [310, 220, 320, 280, 290],
    })
    assert pp.link_type(pairs).tolist() == ["minhash", "truncated_copy", "edited_copy", "partial_overlap", "paraphrase"]


def test_containment_of_a_truncated_copy():
    from fake_news.similarity.shingles import jaccard, shingle_ids

    long = " ".join(f"word{i}" for i in range(200))          # 198 shingles
    short = " ".join(f"word{i}" for i in range(50))           # its first 50 words: 48 shingles, all in `long`
    other = " ".join(f"other{i}" for i in range(50))
    df = pd.DataFrame({"doc_id": [1, 2, 3], "content": [long, short, other], "n_words": [200, 50, 50],
                       "is_fake": [1, 1, 0]})
    pairs = pd.DataFrame({"doc_a": [1, 1], "doc_b": [2, 3], "cosine": [0.91, 0.91]})
    info = pp.add_pair_info(pairs, df, pd.Series({1: 1, 2: 2, 3: 3})).set_index("doc_b")
    assert info.at[2, "containment"] == 1.0                    # the short text is fully contained
    assert info.at[2, "jaccard"] == jaccard(shingle_ids(long), shingle_ids(short)) < 0.3   # but Jaccard is low
    assert info.at[3, "containment"] == 0.0
    assert pp.link_type(info.reset_index()).tolist() == ["truncated_copy", "paraphrase"]


def test_select_links_containment_rule():
    pairs = pd.DataFrame({"doc_a": [1, 2, 3, 4], "doc_b": [5, 6, 7, 8],
                          "cosine": [0.95, 0.91, 0.91, 0.89], "containment": [0.1, 0.75, 0.5, 0.99]})
    links = pp.select_links(pairs)
    # 0.95: cosine rule; 0.91 with containment 0.75: containment rule; 0.91 / 0.5 and 0.89 / 0.99: not linked
    assert links[["doc_a", "rule"]].values.tolist() == [[1, "embedding"], [2, "containment"]]


def test_tier_records_the_containment_rule():
    minhash = pd.Series({1: 1, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7})
    pairs = pd.DataFrame({"doc_a": [2, 4, 6, 1], "doc_b": [3, 5, 7, 2],
                          "rule": ["containment", "containment", "embedding", "containment"]})
    g = pp.two_tier_groups(range(1, 8), minhash, pairs).set_index("doc_id")
    assert g.at[3, "tier"] == "both+containment"               # MinHash group {1,2} enlarged by a containment link
    assert g.at[4, "tier"] == g.at[5, "tier"] == "embedding+containment"
    assert g.at[6, "tier"] == "embedding"
