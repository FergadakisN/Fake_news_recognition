"""Embedding tier of the near-duplicate groups: paraphrases that MinHash cannot see.

MinHash compares *wording* (shared 3-word shingles). A story rewritten in other words (a paraphrase, or the
same agency report rewritten by another site) shares few shingles, so MinHash misses it. Sentence embeddings
(MiniLM, see ``fake_news.features.embeddings``) compare *meaning*: two texts about the same event in different
words get vectors with high cosine similarity.

Method:

1. **Exact nearest neighbours.** The vectors are L2-normalized, so cosine similarity is a dot product. For
   every article, compute its similarity to all 62,525 articles, block by block (``chunk`` rows of X @ X.T at
   a time, so memory stays bounded), and keep its ``k`` most similar other articles. No approximation and no
   clustering: 62,525 x 62,525 dot products of 384 numbers take well under a minute with NumPy/BLAS.
2. **Pairs.** Every (article, neighbour) with cosine >= a threshold is an embedding pair. Each pair is stored
   once (smaller doc_id first).
3. **Merge (later step).** Embedding pairs and MinHash pairs (J >= 0.7) are joined with union-find into the
   two-tier groups.

Only the top-k neighbours are considered, so an article with more than k near-identical articles (e.g. a
widget-text group of 40) links to k of them; union-find still connects the whole group through them.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from fake_news import config

TOP_K = 10
TOPK_CACHE = config.INTERIM_DIR / f"embedding_top{TOP_K}.npz"   # not in git (data/interim/)


def embedding_topk(X: np.ndarray, k: int = TOP_K, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-``k`` cosine neighbours of every row of ``X`` (rows L2-normalized), excluding the row itself.

    Returns ``(idx, sim)``, both of shape ``(n, k)``: row positions of the neighbours and their cosine
    similarity, most similar first. Computed in blocks of ``chunk`` rows (chunk x n float32 at a time).
    """
    n = len(X)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        block = X[start:start + chunk] @ X.T                       # (rows, n) cosine similarities
        rows = np.arange(len(block))
        block[rows, start + rows] = -np.inf                         # an article is not its own neighbour
        top = np.argpartition(-block, k, axis=1)[:, :k]             # k largest, unordered (fast)
        top_sim = block[rows[:, None], top]
        order = np.argsort(-top_sim, axis=1, kind="stable")         # sort those k, most similar first
        idx[start:start + chunk] = top[rows[:, None], order]
        sim[start:start + chunk] = top_sim[rows[:, None], order]
    return idx, sim


def load_embeddings_checked(doc_ids: np.ndarray) -> np.ndarray:
    """The MiniLM vectors of ``doc_ids`` (rows aligned), after checking they belong to the current cleaned data.

    ``embeddings_for`` compares a hash of the raw bytes of clean_fingerprint.json with the hash stored when the
    embeddings were built. They were built on Windows, where git checks the file out with CRLF line endings,
    so on macOS/Linux (LF) the bytes differ although the content is the same, and ``embeddings_for`` raises
    StaleEmbeddingsError. Here the stored hash is accepted if it matches the current file with either line
    ending; any real change of the cleaned data still fails. The shared ``embeddings`` module is not modified.
    """
    from fake_news.features.embeddings import StaleEmbeddingsError, load_embeddings, load_meta

    text = config.FINGERPRINT_JSON.read_bytes().replace(b"\r\n", b"\n")
    accepted = {hashlib.sha256(v).hexdigest()[:16] for v in (text, text.replace(b"\n", b"\r\n"))}
    stored = load_meta()["data_fingerprint"]
    if stored not in accepted:
        raise StaleEmbeddingsError(f"Embeddings were built from another version of the cleaned data ({stored}).")
    E, ids = load_embeddings(check=False)
    pos = pd.Index(ids).get_indexer(np.asarray(doc_ids, dtype=np.int64))
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} doc_ids have no embedding")
    return E[pos]


def load_or_compute_topk(doc_ids: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """``embedding_topk`` for the embeddings of ``doc_ids``, cached in data/interim/ (rebuilt when the
    embeddings or the doc_ids change). Neighbour indices are positions in ``doc_ids``."""
    from fake_news.features.embeddings import load_meta

    meta = {"k": k, "embeddings": load_meta().get("data_fingerprint"),
            "doc_ids": hashlib.sha256(np.asarray(doc_ids, np.int64).tobytes()).hexdigest()[:16]}
    path = config.INTERIM_DIR / f"embedding_top{k}.npz"
    if path.exists():
        with np.load(path) as z:
            if json.loads(str(z["meta"])) == meta:
                return z["idx"], z["sim"]
    idx, sim = embedding_topk(load_embeddings_checked(doc_ids), k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, idx=idx, sim=sim, meta=json.dumps(meta))
    tmp.replace(path)
    return idx, sim


def embedding_pairs(doc_ids: np.ndarray, idx: np.ndarray, sim: np.ndarray, threshold: float) -> pd.DataFrame:
    """All (article, top-k neighbour) pairs with cosine >= ``threshold``, each pair once.

    Returns ``doc_a, doc_b`` (doc_a < doc_b) and ``cosine``. A pair found from both sides is kept once.
    """
    rows, cols = np.nonzero(sim >= threshold)
    a, b = doc_ids[rows], doc_ids[idx[rows, cols]]
    pairs = pd.DataFrame({"doc_a": np.minimum(a, b), "doc_b": np.maximum(a, b), "cosine": sim[rows, cols]})
    return pairs.drop_duplicates(["doc_a", "doc_b"]).sort_values(["doc_a", "doc_b"]).reset_index(drop=True)


def merge_groups(doc_ids, minhash_groups: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    """Two-tier groups: connected components of the MinHash groups plus the embedding ``pairs``.

    ``minhash_groups`` maps doc_id -> MinHash group id; every member is linked to its group, then every
    embedding pair is added as an edge (union-find). Returns doc_id -> group id (smallest doc_id of the group).
    """
    from fake_news.similarity.groups import UnionFind

    uf = UnionFind(doc_ids)
    for doc, gid in minhash_groups.items():
        if doc != gid:
            uf.union(doc, gid)
    for a, b in zip(pairs["doc_a"], pairs["doc_b"]):
        uf.union(a, b)
    root = pd.Series(uf.components())
    merged = pd.Series(root.index, index=root.index).groupby(root.to_numpy()).transform("min")
    merged.index.name = "doc_id"
    return merged.rename("group_id").sort_index()


def add_pair_info(pairs: pd.DataFrame, df: pd.DataFrame, minhash_groups: pd.Series) -> pd.DataFrame:
    """Add the 3-word exact Jaccard (the MinHash tier's measure), the containment, labels, lengths and whether
    MinHash already put the two articles in one group. ``df`` needs ``doc_id, content, n_words, is_fake``."""
    from fake_news.similarity.shingles import jaccard, shingle_ids

    d = df.set_index("doc_id")
    need = np.union1d(pairs["doc_a"], pairs["doc_b"])
    sh = {i: shingle_ids(t) for i, t in d["content"].loc[need].items()}
    out = pairs.copy()
    out["jaccard"] = [jaccard(sh[a], sh[b]) for a, b in zip(out["doc_a"], out["doc_b"])]
    # containment: share of the SHORTER text's shingles that are also in the other text. High containment with
    # low Jaccard = one text contains the other (truncated/extended copy); Jaccard is low only because of length.
    inter = [len(np.intersect1d(sh[a], sh[b], assume_unique=True)) for a, b in zip(out["doc_a"], out["doc_b"])]
    smaller = [max(min(len(sh[a]), len(sh[b])), 1) for a, b in zip(out["doc_a"], out["doc_b"])]
    out["containment"] = np.array(inter) / np.array(smaller)
    out["same_minhash_group"] = minhash_groups.loc[out["doc_a"]].to_numpy() == minhash_groups.loc[out["doc_b"]].to_numpy()
    for col, src in (("a", "doc_a"), ("b", "doc_b")):
        out[f"is_fake_{col}"] = d["is_fake"].loc[out[src]].to_numpy()
        out[f"n_words_{col}"] = d["n_words"].loc[out[src]].to_numpy()
    out["mixed_labels"] = out["is_fake_a"] != out["is_fake_b"]
    return out


COSINE_THRESHOLD = 0.92      # an embedding pair is a link at cosine >= 0.92 ...
CONTAINMENT_BAND_LOW = 0.90  # ... or at 0.90 <= cosine < 0.92 if
CONTAINMENT_MIN = 0.7        # ... its 3-word shingle containment is >= 0.7 (a copy, not a topic neighbour)
TIERS = ("none", "minhash", "embedding", "embedding+containment", "both", "both+containment")


def select_links(pairs: pd.DataFrame, threshold: float = COSINE_THRESHOLD, band_low: float = CONTAINMENT_BAND_LOW,
                 containment_min: float = CONTAINMENT_MIN) -> pd.DataFrame:
    """The embedding pairs that become links, with the rule that admitted each (column ``rule``).

    - ``"embedding"``: cosine >= ``threshold`` (0.92);
    - ``"containment"``: ``band_low`` <= cosine < ``threshold`` (0.90-0.92) and 3-word shingle containment
      >= ``containment_min`` (0.7). Below 0.92 most pairs are different articles on one news topic, which would
      chain topics together; requiring that one text largely contains the other keeps only the copies
      (truncated, extended or edited) that sit just below the cosine threshold.

    ``pairs`` needs ``cosine`` and ``containment`` (from :func:`add_pair_info`).
    """
    by_cosine = pairs["cosine"] >= threshold
    by_containment = (pairs["cosine"] >= band_low) & ~by_cosine & (pairs["containment"] >= containment_min)
    out = pairs[by_cosine | by_containment].copy()
    out["rule"] = np.where(by_cosine[by_cosine | by_containment], "embedding", "containment")
    return out


def two_tier_groups(doc_ids, minhash_groups: pd.Series, pairs: pd.DataFrame) -> pd.DataFrame:
    """Final groups with the tier(s) that linked each one: ``doc_id, group_id, tier``.

    ``tier`` describes the whole group (the same for all its members):

    - ``"none"``: a singleton (no near-duplicate);
    - ``"minhash"``: exactly one MinHash group, not changed by the embedding links;
    - ``"embedding"``: linked only by embedding links (copies and paraphrases MinHash did not see);
    - ``"both"``: one or more MinHash groups enlarged or joined by embedding links.

    ``"+containment"`` is appended (``"embedding+containment"``, ``"both+containment"``) when at least one link
    admitted by the containment rule (``pairs["rule"] == "containment"``, see :func:`select_links`) connects
    articles from different MinHash groups inside the group, i.e. the rule changed the group.
    """
    merged = merge_groups(doc_ids, minhash_groups, pairs)
    mh = minhash_groups.reindex(merged.index)
    mh_size = mh.map(mh.value_counts())
    d = pd.DataFrame({"group_id": merged, "mh": mh, "in_mh_group": mh_size >= 2})
    per = d.groupby("group_id").agg(size=("mh", "size"),
                                    n_mh_groups=("mh", lambda s: s[d.loc[s.index, "in_mh_group"]].nunique()),
                                    n_in_mh=("in_mh_group", "sum"))
    tier = pd.Series(np.select([per["size"] == 1, per["n_mh_groups"] == 0,
                                (per["n_mh_groups"] == 1) & (per["n_in_mh"] == per["size"])],
                               ["none", "embedding", "minhash"], default="both"), index=per.index)

    if "rule" in pairs.columns:
        c = pairs[pairs["rule"] == "containment"]
        changing = c[mh.loc[c["doc_a"]].to_numpy() != mh.loc[c["doc_b"]].to_numpy()]  # not inside one MinHash group
        marked = merged.loc[changing["doc_a"]].unique()
        tier.loc[marked] = tier.loc[marked] + "+containment"

    out = merged.to_frame()
    out["tier"] = out["group_id"].map(tier)
    return out.reset_index()


def save_two_tier_groups(groups: pd.DataFrame, path=None) -> "Path":
    """Write ``doc_id, group_id, tier`` to data/processed/near_dup_groups.csv (the final groups)."""
    from fake_news.similarity.near_duplicates import GROUPS_CSV

    path = path or GROUPS_CSV
    groups[["doc_id", "group_id", "tier"]].sort_values("doc_id").to_csv(path, index=False)
    return path


# ---- what kind of link did the embedding tier add? (descriptive only; grouping uses cosine alone) ----------
COPY_CONTAINMENT = 0.8        # >= 0.8 of the shorter text's shingles are in the other text: a copy
PARAPHRASE_CONTAINMENT = 0.3  # < 0.3: little shared wording: a paraphrase
TRUNCATION_RATIO = 1.5        # a copy whose texts differ in length by >= 1.5x: truncated / extended copy
LINK_TYPES = ("minhash", "truncated_copy", "edited_copy", "partial_overlap", "paraphrase")


def link_type(pairs: pd.DataFrame) -> pd.Series:
    """Classify embedding pairs (needs ``same_minhash_group, containment, n_words_a, n_words_b`` from
    :func:`add_pair_info`):

    - ``minhash``: MinHash already put the two in one group (the embedding link adds nothing);
    - ``truncated_copy``: one text contains the other (containment >= 0.8) but is >= 1.5x longer or shorter.
      MinHash misses these because Jaccard divides by the union, which the extra length inflates;
    - ``edited_copy``: containment >= 0.8 at similar length (Jaccard just below MinHash's 0.7);
    - ``partial_overlap``: 0.3 <= containment < 0.8 (shared passages: updated versions, long quotes);
    - ``paraphrase``: containment < 0.3: the same story in other words.
    """
    ratio = (np.maximum(pairs["n_words_a"], pairs["n_words_b"])
             / np.maximum(np.minimum(pairs["n_words_a"], pairs["n_words_b"]), 1))
    c = pairs["containment"]
    kind = np.select(
        [pairs["same_minhash_group"], (c >= COPY_CONTAINMENT) & (ratio >= TRUNCATION_RATIO),
         c >= COPY_CONTAINMENT, c >= PARAPHRASE_CONTAINMENT],
        ["minhash", "truncated_copy", "edited_copy", "partial_overlap"], default="paraphrase")
    return pd.Series(kind, index=pairs.index, name="link_type")
