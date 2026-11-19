"""Near-duplicate pipeline on the cleaned articles: shingles -> MinHash -> LSH -> exact Jaccard -> groups.

Used by ``scripts/find_near_duplicates.py`` (end-to-end run) and ``notebooks/04-minhash-lsh.ipynb``.
The building blocks live in ``shingles``, ``minhash``, ``lsh`` and ``groups``; this module only
connects them, caches the signatures and summarizes the results.

Short texts: articles with fewer than ``MIN_SHINGLES`` distinct shingles get no signature and
always stay singletons. They are kept in the data on purpose (cleaning decision G7), but with a
handful of shingles two unrelated teasers can reach a high Jaccard similarity by chance.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.similarity import lsh, minhash
from fake_news.similarity.groups import build_groups, exact_jaccard_pairs
from fake_news.similarity.shingles import K, shingle_ids, words

MIN_SHINGLES = 5
SIGNATURES_CACHE = config.INTERIM_DIR / "minhash_signatures.npz"  # not in git (data/interim/)
GROUPS_CSV = config.PROCESSED_DIR / "near_dup_groups.csv"          # in git
REPORT_DIR = config.PROJECT_ROOT / "reports" / "minhash"


@dataclass
class SignatureTable:
    """MinHash signatures of the cleaned articles."""

    doc_ids: np.ndarray      # doc_ids of the articles WITH a signature (>= MIN_SHINGLES shingles)
    signatures: np.ndarray   # (len(doc_ids), n_hashes) int64, row i belongs to doc_ids[i]
    n_shingles: pd.Series    # doc_id -> number of distinct shingles, for ALL articles
    from_cache: bool = False  # True if read from SIGNATURES_CACHE instead of computed

    @property
    def short_doc_ids(self) -> np.ndarray:
        """Articles too short for a signature (always singletons)."""
        return self.n_shingles.index[self.n_shingles < MIN_SHINGLES].to_numpy()


# ---------------------------------------------------------------------------------------------------------
# Stage 1: shingles + signatures (cached)
# ---------------------------------------------------------------------------------------------------------

def _data_fingerprint() -> str:
    """Short hash of clean_fingerprint.json: changes whenever the cleaned data changes."""
    f = config.FINGERPRINT_JSON
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "missing"


def _ids_hash(doc_ids) -> str:
    return hashlib.sha256(np.asarray(doc_ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def compute_signatures(df: pd.DataFrame, k: int = K, n_hashes: int = minhash.N_HASHES,
                       seed: int = config.SEED, cache: Path | None = SIGNATURES_CACHE,
                       rebuild: bool = False) -> SignatureTable:
    """Signatures of the articles in ``df`` (columns ``doc_id, content``), read from ``cache`` if up to date.

    The cache is reused only if it was built from the same cleaned data, the same doc_ids and the
    same settings (k, number of hashes, seed, minimum shingles). ``rebuild=True`` recomputes and
    overwrites it; ``cache=None`` neither reads nor writes a cache.
    """
    meta = {"data_fingerprint": _data_fingerprint(), "doc_ids_hash": _ids_hash(df["doc_id"]),
            "k": k, "n_hashes": n_hashes, "seed": seed, "min_shingles": MIN_SHINGLES}

    if cache is not None and cache.exists() and not rebuild:
        with np.load(cache) as z:
            if json.loads(str(z["meta"])) == meta:
                return SignatureTable(z["doc_ids"], z["signatures"],
                                      pd.Series(z["n_shingles"], index=pd.Index(z["all_doc_ids"], name="doc_id")),
                                      from_cache=True)

    sets = [shingle_ids(t, k) for t in df["content"]]
    n_shingles = np.array([len(s) for s in sets])
    has_sig = n_shingles >= MIN_SHINGLES
    a, b = minhash.make_hash_params(n_hashes, seed)
    sigs = minhash.signatures([s for s, h in zip(sets, has_sig) if h], a, b)
    table = SignatureTable(df["doc_id"].to_numpy()[has_sig], sigs,
                           pd.Series(n_shingles, index=pd.Index(df["doc_id"].to_numpy(), name="doc_id")))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")  # write-then-rename: never a half-written cache
        np.savez(tmp, doc_ids=table.doc_ids, signatures=table.signatures, all_doc_ids=table.n_shingles.index,
                 n_shingles=table.n_shingles.to_numpy(), meta=json.dumps(meta))
        tmp.replace(cache)
    return table


# ---------------------------------------------------------------------------------------------------------
# Stage 2: LSH candidate pairs
# ---------------------------------------------------------------------------------------------------------

def find_candidates(table: SignatureTable, bands: int = lsh.BANDS, rows: int = lsh.ROWS) -> pd.DataFrame:
    """Candidate pairs from LSH banding: ``doc_a, doc_b`` (doc_a < doc_b) and ``minhash`` (estimated Jaccard)."""
    rows_ij = lsh.candidate_pairs(table.signatures, bands, rows)
    i, j = rows_ij[:, 0], rows_ij[:, 1]
    est = (table.signatures[i] == table.signatures[j]).mean(axis=1)
    a, b = table.doc_ids[i], table.doc_ids[j]
    return pd.DataFrame({"doc_a": np.minimum(a, b), "doc_b": np.maximum(a, b), "minhash": est})


def largest_buckets(table: SignatureTable, top: int = 5, bands: int = lsh.BANDS,
                    rows: int = lsh.ROWS) -> list[np.ndarray]:
    """doc_ids of the ``top`` largest LSH buckets (over all bands), largest first."""
    index = lsh.LSHIndex(bands, rows)
    index.add_many(table.doc_ids, table.signatures)
    biggest = sorted(index.buckets(), key=len, reverse=True)[:top]
    return [np.array(b) for b in biggest]


# ---------------------------------------------------------------------------------------------------------
# Stage 3: exact Jaccard of the candidates
# ---------------------------------------------------------------------------------------------------------

def verify_candidates(cands: pd.DataFrame, df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """Add the exact Jaccard similarity (column ``jaccard``) to every candidate pair.

    Shingles are recomputed only for the articles that appear in a candidate pair.
    """
    involved = np.union1d(cands["doc_a"], cands["doc_b"])
    texts = df.set_index("doc_id")["content"].loc[involved]
    shingles = {doc_id: shingle_ids(t, k) for doc_id, t in texts.items()}
    out = cands.copy()
    out["jaccard"] = exact_jaccard_pairs(out[["doc_a", "doc_b"]].to_numpy(), shingles)
    return out


# ---------------------------------------------------------------------------------------------------------
# Stage 4: groups
# ---------------------------------------------------------------------------------------------------------

def make_groups(doc_ids, cands: pd.DataFrame, threshold: float) -> pd.Series:
    """``doc_id -> group_id`` for all ``doc_ids``, joining verified pairs with ``jaccard >= threshold``."""
    return build_groups(doc_ids, cands[["doc_a", "doc_b"]].to_numpy(), cands["jaccard"].to_numpy(), threshold)


def group_table(groups: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """One row per group with 2+ articles: ``size``, ``n_fake``, ``mixed`` (both labels), largest first."""
    df = pd.DataFrame({"group_id": groups, "is_fake": labels.reindex(groups.index)})
    t = df.groupby("group_id")["is_fake"].agg(size="size", n_fake="sum")
    t = t[t["size"] >= 2].copy()
    t["mixed"] = (t["n_fake"] > 0) & (t["n_fake"] < t["size"])
    return t.sort_values("size", ascending=False)


def group_summary(groups: pd.Series, labels: pd.Series) -> dict:
    """Headline numbers for the report."""
    t = group_table(groups, labels)
    return {
        "n_articles": int(len(groups)),
        "groups_2plus": int(len(t)),
        "articles_in_groups_2plus": int(t["size"].sum()),
        "share_articles_in_groups": round(float(t["size"].sum() / len(groups)), 4),
        "largest_group": int(t["size"].max()) if len(t) else 1,
        "mixed_label_groups": int(t["mixed"].sum()),
        "articles_in_mixed_groups": int(t.loc[t["mixed"], "size"].sum()),
        "group_size_counts": {int(s): int(c) for s, c in t["size"].value_counts().sort_index().items()},
    }


def save_groups(groups: pd.Series, path: Path = GROUPS_CSV) -> Path:
    """Write ``doc_id, group_id`` (every cleaned article; singletons have ``group_id == doc_id``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    groups.rename_axis("doc_id").rename("group_id").reset_index().sort_values("doc_id").to_csv(path, index=False)
    return path


def load_groups(path: Path = GROUPS_CSV) -> pd.Series:
    """``doc_id -> group_id`` as written by :func:`save_groups`."""
    return pd.read_csv(path).set_index("doc_id")["group_id"]


# ---------------------------------------------------------------------------------------------------------
# Diagnostics for choosing the threshold
# ---------------------------------------------------------------------------------------------------------

def lsh_recall(cands: pd.DataFrame, reference: pd.DataFrame,
               thresholds=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)) -> pd.DataFrame:
    """How many of the pairs found by a looser banding (``reference``, e.g. 32 x 4) our banding also finds.

    Both tables need ``doc_a, doc_b, jaccard``. Per threshold t: pairs with J >= t in the reference,
    how many of them are in ``cands``, and the share (an estimate of LSH recall above t).
    """
    ours = set(zip(cands["doc_a"], cands["doc_b"]))
    rows = []
    for t in thresholds:
        ref = reference[reference["jaccard"] >= t]
        found = sum(pair in ours for pair in zip(ref["doc_a"], ref["doc_b"]))
        rows.append({"threshold": t, "reference_pairs": len(ref), "found": found,
                     "recall": round(found / len(ref), 4) if len(ref) else float("nan")})
    return pd.DataFrame(rows)


def threshold_table(doc_ids, cands: pd.DataFrame, labels: pd.Series,
                    thresholds=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)) -> pd.DataFrame:
    """Group statistics for several thresholds side by side."""
    rows = []
    for t in thresholds:
        s = group_summary(make_groups(doc_ids, cands, t), labels)
        s.pop("group_size_counts")
        rows.append({"threshold": t, "verified_pairs": int((cands["jaccard"] >= t).sum()), **s})
    return pd.DataFrame(rows).set_index("threshold")


def _shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 2] + " …"


def word_diff(text_a: str, text_b: str, max_chars: int = 300) -> tuple[str, str]:
    """The words that differ between two texts (from difflib), as ``(only in A, only in B)``.

    Differing stretches are joined with " | ". Shows at a glance *why* a pair is not identical
    (e.g. a different headline, an added "share this" footer).
    """
    wa, wb = words(text_a), words(text_b)
    only_a, only_b = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if op != "equal":
            if i2 > i1:
                only_a.append(" ".join(wa[i1:i2]))
            if j2 > j1:
                only_b.append(" ".join(wb[j1:j2]))
    return _shorten(" | ".join(only_a), max_chars), _shorten(" | ".join(only_b), max_chars)


def threshold_examples(cands: pd.DataFrame, df: pd.DataFrame, around: float = 0.8, n: int = 10) -> pd.DataFrame:
    """The ``n`` candidate pairs just above and the ``n`` just below ``around`` (exact Jaccard), side by side.

    ``df`` needs ``doc_id, title, content, n_words, is_fake``.
    """
    above = cands[cands["jaccard"] >= around].nsmallest(n, "jaccard").assign(side="above")
    below = cands[cands["jaccard"] < around].nlargest(n, "jaccard").assign(side="below")
    ex = pd.concat([above.sort_values("jaccard", ascending=False), below])
    d = df.set_index("doc_id")
    for col, src in (("a", "doc_a"), ("b", "doc_b")):
        ex[f"is_fake_{col}"] = d["is_fake"].loc[ex[src]].to_numpy()
        ex[f"n_words_{col}"] = d["n_words"].loc[ex[src]].to_numpy()
        ex[f"title_{col}"] = [_shorten(t, 100) for t in d["title"].loc[ex[src]]]
    diffs = [word_diff(d.at[a, "content"], d.at[b, "content"]) for a, b in zip(ex["doc_a"], ex["doc_b"])]
    ex["only_in_a"], ex["only_in_b"] = zip(*diffs) if diffs else ((), ())
    cols = ["side", "jaccard", "minhash", "doc_a", "doc_b", "is_fake_a", "is_fake_b", "n_words_a", "n_words_b",
            "title_a", "title_b", "only_in_a", "only_in_b"]
    return ex[cols].round({"jaccard": 3, "minhash": 3}).reset_index(drop=True)


def describe_group(groups: pd.Series, group_id, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """A few members of one group: ``doc_id, is_fake, n_words, title`` and the start of ``content``."""
    members = groups.index[groups == group_id]
    d = df.set_index("doc_id").loc[members[:n]]
    return pd.DataFrame({"is_fake": d["is_fake"], "n_words": d["n_words"],
                         "title": [_shorten(t, 70) for t in d["title"]],
                         "content_start": [_shorten(t, 140) for t in d["content"]]})


def binned_recall(cands: pd.DataFrame, reference: pd.DataFrame, width: float = 0.05,
                  min_pairs: int = 20) -> pd.DataFrame:
    """Share of the ``reference`` pairs found by ``cands``, per Jaccard bin of ``width``.

    Directly comparable with the LSH S-curve, which gives P(candidate) at one similarity value.
    Bins with fewer than ``min_pairs`` reference pairs are left out (too noisy). The reference is a
    looser banding, not the full set of pairs, so at low similarity it misses pairs itself.
    """
    ours = set(zip(cands["doc_a"], cands["doc_b"]))
    ref = reference.assign(found=[p in ours for p in zip(reference["doc_a"], reference["doc_b"])],
                           bin=np.floor(reference["jaccard"].clip(upper=1 - 1e-9) / width) * width)
    out = ref.groupby("bin").agg(reference_pairs=("found", "size"), found=("found", "sum"))
    out = out[out["reference_pairs"] >= min_pairs]
    out["recall"] = out["found"] / out["reference_pairs"]
    out["center"] = out.index + width / 2
    return out.reset_index()
