"""Evaluation of the MinHash/LSH part for the report: retrieval quality against exact Jaccard, and a leakage audit.

    python scripts/evaluate_lsh.py                 # everything (full-data checks are cached in data/interim/)
    python scripts/evaluate_lsh.py --sample-only   # only the 3,000-article sample (about a minute)

Stages
    1. sample    3,000 random articles (seed 42), 3-word shingles. Exact Jaccard of all 4.5 million pairs is
                 the ground truth. For the bandings 16x8, 32x4 and 64x2: recall at J >= 0.5 ... 0.9,
                 false-positive rate, candidates per article, runtime vs brute force, measured S-curve.
    2. sample2   the same sample with 2-word shingles: the feature index (64x2, prefilter, min_sim 0.1)
                 against exact search.
    3. full      the same retrieval check on all 62,525 articles for the pairs that matter (J >= 0.2): a random
                 sample of 3,000 holds only a handful of near-duplicate pairs, so recall at high J needs the
                 full data. Exact all-pairs search takes ~30 min the first time (cached afterwards).
    4. leakage   grouped regime: the feature index (train only) against exact search for every val article,
                 then how many val articles still have a train neighbour at 2-word J >= 0.5 / 0.3, and how
                 fake_neighbor_share performs on them vs the rest. Random regime alongside for comparison.

Only train and val are used: test is never scored. Nothing is tuned here.

Outputs
    reports/minhash/lsh_eval_retrieval.csv        recall / false positives / candidates per banding and threshold
    reports/minhash/lsh_eval_runtime.csv          runtime of LSH stages vs brute force
    reports/minhash/lsh_eval_s_curve.csv          measured vs theoretical P(candidate) per Jaccard bin
    reports/minhash/lsh_eval_feature_index.csv    2-word feature index vs exact search (sample and val)
    reports/minhash/lsh_eval_missed_pairs.csv     full data: pairs with J >= 0.7 that 16x8 missed, and where they ended up
    reports/minhash/leakage_audit_counts.csv      val articles with a train neighbour >= 0.3 / 0.5
    reports/minhash/leakage_audit_scores.csv      fake_neighbor_share on those vs the rest
    reports/minhash/leakage_audit_pairs.csv       grouped val articles with a train neighbour >= 0.5: the pair
    reports/minhash/lsh_eval_summary.json         settings, timings and headline numbers
    reports/figures/lsh_eval_s_curves.png
    reports/figures/leakage_max_jaccard.png
"""

import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.data import load_clean, load_splits
from fake_news.similarity import lsh, lsh_eval, minhash, plots
from fake_news.similarity import lsh_features as lf
from fake_news.similarity import near_duplicates as nd
from fake_news.similarity.groups import exact_jaccard_pairs
from fake_news.similarity.shingles import jaccard, shingle_ids

SAMPLE_SIZE = 3000
BANDINGS = [(16, 8), (32, 4), (64, 2)]          # 3-word shingles; 16x8 is the one used for the groups
FULL_MIN_J = 0.2                                 # full data: keep exact pairs from this similarity up
FEATURE_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)   # 2-word feature index
LEAK_THRESHOLDS = (0.3, 0.5)
NAIVE_TIMING_PAIRS = 20_000
SHORT_WORDS = 50                                 # leak pairs: "short text" if one article has fewer words
SAME_STORY_COSINE = 0.8                          # leak pairs: "same story" from this embedding cosine

OUT = nd.REPORT_DIR
CACHE_FULL = config.INTERIM_DIR / "lsh_eval_full_pairs_3w.npz"
CACHE_VAL = config.INTERIM_DIR / "lsh_eval_val_neighbors_grouped.npz"


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def naive_seconds_per_pair(sets: list[np.ndarray], rng: np.random.Generator) -> float:
    """Time of one exact Jaccard with ``np.intersect1d`` (the naive brute force), from random pairs."""
    i = rng.integers(0, len(sets), NAIVE_TIMING_PAIRS)
    j = rng.integers(0, len(sets), NAIVE_TIMING_PAIRS)
    t0 = time.perf_counter()
    for a, b in zip(i, j):
        jaccard(sets[a], sets[b])
    return (time.perf_counter() - t0) / NAIVE_TIMING_PAIRS


def lsh_on_sets(sets: list[np.ndarray], label: str, truth_codes, truth_j, all_pairs_j=None,
                all_codes=None):
    """Signatures + banding + verification for every banding; recall, false positives, S-curve, timings.

    Also returns the candidate pair codes per banding ("16x8" -> codes)."""
    n = len(sets)
    a, b = minhash.make_hash_params()
    sigs, t_sig = timed(minhash.signatures, sets, a, b)
    retrieval, s_curves, runtime, cand_codes = [], [], [], {}
    for bands, rows in BANDINGS:
        pairs, t_band = timed(lsh.candidate_pairs, sigs, bands, rows)
        codes = lsh_eval.pair_codes(pairs[:, 0], pairs[:, 1], n)
        cand_codes[f"{bands}x{rows}"] = codes
        shingle_map = dict(enumerate(sets))
        cand_j, t_verify = timed(exact_jaccard_pairs, pairs, shingle_map)   # what the pipeline does next
        rec = lsh_eval.recall_table(truth_codes, truth_j, codes)
        fp = lsh_eval.false_positive_table(cand_j)
        tab = rec.merge(fp, on="threshold")
        tab.insert(0, "banding", f"{bands}x{rows}")
        tab.insert(0, "data", label)
        tab["candidates_per_article"] = 2 * len(pairs) / n
        retrieval.append(tab)
        if all_pairs_j is not None:     # sample: every pair, all similarities
            sc = lsh_eval.measured_s_curve(all_pairs_j, np.isin(all_codes, codes), bands, rows)
        else:                           # full data: the stored pairs (J >= FULL_MIN_J)
            sc = lsh_eval.measured_s_curve(truth_j, np.isin(truth_codes, codes), bands, rows)
            sc = sc[sc["j_low"] >= FULL_MIN_J - 1e-9]
        s_curves.append(sc.assign(data=label, banding=f"{bands}x{rows}", shingle_words=3))
        runtime.append({"data": label, "method": f"LSH {bands}x{rows}", "articles": n, "signatures_s": t_sig,
                        "banding_s": t_band, "verify_s": t_verify, "total_s": t_sig + t_band + t_verify,
                        "pairs_checked": len(pairs)})
        print(f"  {bands}x{rows}: {len(pairs):,} candidates ({2 * len(pairs) / n:.2f} per article), "
              f"recall at 0.7 = {rec.set_index('threshold').at[0.7, 'recall']:.3f} "
              f"({rec.set_index('threshold').at[0.7, 'true_pairs']} pairs), "
              f"banding {t_band:.2f} s, verify {t_verify:.2f} s", flush=True)
    return retrieval, s_curves, runtime, cand_codes


# ---------------------------------------------------------------------------------------------------------
# 1. Sample, 3-word shingles
# ---------------------------------------------------------------------------------------------------------

def stage_sample(df: pd.DataFrame, rng: np.random.Generator, summary: dict):
    print(f"\n[sample] {SAMPLE_SIZE:,} random articles (seed {config.SEED}), 3-word shingles", flush=True)
    rows = np.sort(np.random.default_rng(config.SEED).choice(len(df), SAMPLE_SIZE, replace=False))
    sets_all, t_sh = timed(lambda: [shingle_ids(t, 3) for t in df["content"].to_numpy()[rows]])
    keep = np.array([len(s) >= nd.MIN_SHINGLES for s in sets_all])
    sets = [s for s, k in zip(sets_all, keep) if k]
    n = len(sets)
    print(f"  {n:,} articles with a signature ({(~keep).sum()} too short, left out); shingling {t_sh:.1f} s")

    J, t_exact = timed(lsh_eval.all_pairs_jaccard, sets)
    iu = np.triu_indices(n, k=1)
    pair_j = J[iu].astype(float)
    codes = lsh_eval.pair_codes(iu[0], iu[1], n)
    print(f"  exact Jaccard of all {len(pair_j):,} pairs (posting lists): {t_exact:.1f} s")
    per_pair = naive_seconds_per_pair(sets, rng)

    retrieval, s_curves, runtime, _ = lsh_on_sets(sets, "sample", codes, pair_j, pair_j, codes)
    runtime += [{"data": "sample", "method": "brute force, posting lists (measured)", "articles": n,
                 "total_s": t_exact, "pairs_checked": len(pair_j)},
                {"data": "sample", "method": "brute force, pair by pair (extrapolated)", "articles": n,
                 "total_s": per_pair * len(pair_j), "pairs_checked": len(pair_j)}]
    hist = np.histogram(pair_j, bins=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001])[0]
    summary["sample"] = {"articles": n, "too_short": int((~keep).sum()), "pairs": len(pair_j),
                         "pairs_by_jaccard": dict(zip(["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", ">=0.7"],
                                                      map(int, hist))),
                         "shingling_s": round(t_sh, 2), "naive_us_per_pair": round(1e6 * per_pair, 2)}
    return rows[keep], retrieval, s_curves, runtime


# ---------------------------------------------------------------------------------------------------------
# 2. Sample, 2-word shingles: the feature index
# ---------------------------------------------------------------------------------------------------------

def feature_index_check(index: lf.LSHNeighborIndex, enc: lf.Encoded, label: str, query_keys=None,
                        index_keys=None):
    """LSH + prefilter (the feature setting), LSH without prefilter, and exact search, for the same queries."""
    out, nbs = [], {}
    for method, kw in (("lsh_prefilter", {"min_agree": lf.MIN_AGREE}), ("lsh_no_prefilter", {}),
                       ("exact", {"exact": True})):
        nb, t = timed(index.neighbors, encoded=enc, query_keys=query_keys, index_keys=index_keys, **kw)
        nbs[method] = nb
        print(f"  {label} {method}: {1000 * t / len(enc):.1f} ms per article", flush=True)
        out.append({"method": method, "seconds": t, "ms_per_article": 1000 * t / len(enc),
                    "candidates_per_article": float(nb.n_candidates.mean())})
    # directed pairs (query, training article) with exact J >= t, and which of them LSH returned
    n_index = len(index.sizes_)
    q_ex = np.repeat(np.arange(len(enc)), [len(r) for r in nbs["exact"].rows])
    codes_ex = q_ex * n_index + np.concatenate(nbs["exact"].rows)
    j_ex = np.concatenate(nbs["exact"].sims)
    rows = []
    for m in out:
        nb = nbs[m["method"]]
        q = np.repeat(np.arange(len(enc)), [len(r) for r in nb.rows])
        found = np.isin(codes_ex, q * n_index + np.concatenate(nb.rows))
        has_nb = np.array([len(s) > 0 for s in nb.sims])
        has_nb_ex = np.array([len(s) > 0 for s in nbs["exact"].sims])
        for t in FEATURE_THRESHOLDS:
            above = j_ex >= t
            lo, hi = lsh_eval.wilson_interval(int((found & above).sum()), int(above.sum()))
            rows.append({"data": label, **m, "threshold": t, "true_pairs": int(above.sum()),
                         "found": int((found & above).sum()),
                         "recall": (found & above).sum() / above.sum() if above.any() else float("nan"),
                         "recall_ci_low": lo, "recall_ci_high": hi,
                         "articles_with_neighbor": float(has_nb.mean()),
                         "articles_with_neighbor_exact": float(has_nb_ex.mean())})
    return pd.DataFrame(rows), nbs


def stage_sample_2word(df: pd.DataFrame, rows: np.ndarray, summary: dict):
    print("\n[sample2] the same articles, 2-word shingles: the feature index (64x2) vs exact search", flush=True)
    enc = lf.encode(df["content"].to_numpy()[rows].tolist(), shingle_k=lf.SHINGLE_WORDS)
    index = lf.LSHNeighborIndex(lf.BANDS, lf.ROWS, shingle_k=lf.SHINGLE_WORDS)
    index.fit(None, df["is_fake"].to_numpy()[rows], encoded=enc)
    keys = np.arange(len(enc))           # every article is in the index: never count itself
    table, _ = feature_index_check(index, enc, "sample", keys, keys)

    # measured S-curve of the 64x2 banding on 2-word shingles (all pairs of the sample)
    sets = [s for s, h in zip(enc.shingles, enc.has_sig) if h]
    J = lsh_eval.all_pairs_jaccard(sets)
    iu = np.triu_indices(len(sets), k=1)
    pairs = lsh.candidate_pairs(enc.signatures[enc.has_sig], lf.BANDS, lf.ROWS)
    is_cand = np.isin(lsh_eval.pair_codes(iu[0], iu[1], len(sets)), lsh_eval.pair_codes(pairs[:, 0], pairs[:, 1], len(sets)))
    sc = lsh_eval.measured_s_curve(J[iu].astype(float), is_cand, lf.BANDS, lf.ROWS)
    summary["sample_2word"] = {"pairs_j_ge_0.1": int((J[iu] >= 0.1).sum()), "candidate_pairs": len(pairs)}
    return table, sc.assign(data="sample", banding=f"{lf.BANDS}x{lf.ROWS}", shingle_words=2)


# ---------------------------------------------------------------------------------------------------------
# 3. Full data, 3-word shingles, pairs with J >= 0.2
# ---------------------------------------------------------------------------------------------------------

def _key(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def stage_full(df: pd.DataFrame, rng: np.random.Generator, summary: dict):
    print(f"\n[full] all articles, 3-word shingles, exact pairs with J >= {FULL_MIN_J}", flush=True)
    sets_all, t_sh = timed(lambda: [shingle_ids(t, 3) for t in df["content"]])
    keep = np.array([len(s) >= nd.MIN_SHINGLES for s in sets_all])
    sets = [s for s, k in zip(sets_all, keep) if k]
    n = len(sets)
    key = _key(nd._data_fingerprint(), nd._ids_hash(df["doc_id"].to_numpy()[keep]), FULL_MIN_J)
    if CACHE_FULL.exists() and str(np.load(CACHE_FULL)["key"]) == key:
        with np.load(CACHE_FULL) as z:
            truth = pd.DataFrame({"i": z["i"], "j": z["j"], "jaccard": z["jaccard"]})
            t_exact = float(z["seconds"])
        print(f"  exact pairs from cache ({len(truth):,} pairs; computing them took {t_exact / 60:.0f} min)")
    else:
        print(f"  exact search over {n:,} articles (about 30 min) ...", flush=True)
        truth, t_exact = timed(lsh_eval.pairs_above, sets, FULL_MIN_J, verbose_every=5000)
        CACHE_FULL.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_FULL, i=truth["i"].to_numpy(), j=truth["j"].to_numpy(), jaccard=truth["jaccard"].to_numpy(),
                 seconds=t_exact, key=key)
    codes = lsh_eval.pair_codes(truth["i"], truth["j"], n)
    per_pair = naive_seconds_per_pair(sets, rng)
    retrieval, s_curves, runtime, cand_codes = lsh_on_sets(sets, "full", codes, truth["jaccard"].to_numpy())
    missed = missed_pairs(df["doc_id"].to_numpy()[keep], truth, codes, cand_codes[f"{lsh.BANDS}x{lsh.ROWS}"])
    missed.to_csv(OUT / "lsh_eval_missed_pairs.csv", index=False)
    total_pairs = n * (n - 1) // 2
    runtime += [{"data": "full", "method": "brute force, posting lists (measured)", "articles": n,
                 "total_s": t_exact, "pairs_checked": total_pairs},
                {"data": "full", "method": "brute force, pair by pair (extrapolated)", "articles": n,
                 "total_s": per_pair * total_pairs, "pairs_checked": total_pairs}]
    summary["full"] = {"articles": n, "too_short": int((~keep).sum()), "pairs": total_pairs,
                       "pairs_j_ge_0.2": len(truth), "shingling_s": round(t_sh, 1),
                       "exact_search_s": round(t_exact, 1), "naive_us_per_pair": round(1e6 * per_pair, 2),
                       "missed_by_16x8_at_0.7": {"pairs": len(missed),
                                                 "same_minhash_group": int(missed["same_minhash_group"].sum()),
                                                 "same_final_group": int(missed["same_final_group"].sum()),
                                                 "across_split_grouped": int(missed["across_split_grouped"].sum())}}
    return retrieval, s_curves, runtime


def missed_pairs(doc_ids: np.ndarray, truth: pd.DataFrame, codes: np.ndarray, cand_codes: np.ndarray) -> pd.DataFrame:
    """Pairs with exact 3-word J >= 0.7 that the group banding (16x8) did not propose, and whether they still
    ended up together: in one MinHash group or one final group (via other links), or across split_grouped."""
    m = truth[(truth["jaccard"] >= 0.7) & ~np.isin(codes, cand_codes)]
    a, b = doc_ids[m["i"].to_numpy()], doc_ids[m["j"].to_numpy()]
    final, mh = nd.load_groups(), nd.load_minhash_groups()
    part = load_splits().set_index("doc_id")["split_grouped"]
    return pd.DataFrame({"doc_a": a, "doc_b": b, "jaccard": m["jaccard"].round(4).to_numpy(),
                         "same_minhash_group": mh.loc[a].to_numpy() == mh.loc[b].to_numpy(),
                         "same_final_group": final.loc[a].to_numpy() == final.loc[b].to_numpy(),
                         "partition_a": part.loc[a].to_numpy(), "partition_b": part.loc[b].to_numpy(),
                         "across_split_grouped": part.loc[a].to_numpy() != part.loc[b].to_numpy()})


# ---------------------------------------------------------------------------------------------------------
# 4. Leakage audit (grouped), with the feature index vs exact search on val
# ---------------------------------------------------------------------------------------------------------

def _pack(nb: lf.Neighbors) -> dict:
    lengths = np.array([len(r) for r in nb.rows])
    return {"lengths": lengths, "rows": np.concatenate(nb.rows) if len(nb.rows) else np.empty(0, np.int64),
            "sims": np.concatenate(nb.sims) if len(nb.sims) else np.empty(0)}


def _unpack(z, n_cand, t) -> lf.Neighbors:
    bounds = np.concatenate([[0], np.cumsum(z["lengths"])])
    rows = [z["rows"][bounds[i]:bounds[i + 1]] for i in range(len(z["lengths"]))]
    sims = [z["sims"][bounds[i]:bounds[i + 1]] for i in range(len(z["lengths"]))]
    return lf.Neighbors(rows, sims, n_cand, t, 0.0)


def stage_leakage(df: pd.DataFrame, summary: dict):
    print("\n[leakage] grouped regime: val articles vs the train-only feature index", flush=True)
    splits = load_splits().set_index("doc_id").reindex(df["doc_id"])
    part = splits["split_grouped"].to_numpy()
    train, val = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    enc = lf.encode(df["content"].tolist(), shingle_k=lf.SHINGLE_WORDS)
    index = lf.LSHNeighborIndex(lf.BANDS, lf.ROWS, shingle_k=lf.SHINGLE_WORDS)
    index.fit(None, df["is_fake"].to_numpy()[train], encoded=enc.take(train))
    q = enc.take(val)

    # LSH as used for the features; exact search (cached: ~10 min) as the reference
    nb_lsh, t_lsh = timed(index.neighbors, encoded=q, min_agree=lf.MIN_AGREE)
    key = _key(nd._data_fingerprint(), nd._ids_hash(df["doc_id"].to_numpy()[val]),
               nd._ids_hash(df["doc_id"].to_numpy()[train]))
    if CACHE_VAL.exists() and str(np.load(CACHE_VAL)["key"]) == key:
        with np.load(CACHE_VAL) as z:
            nb_ex, t_ex = _unpack(z, z["n_cand"], float(z["seconds"])), float(z["seconds"])
        print(f"  exact val neighbours from cache (took {t_ex / 60:.1f} min)")
    else:
        print(f"  exact search for {len(val):,} val articles (about 10 min) ...", flush=True)
        nb_ex, t_ex = timed(index.neighbors, encoded=q, exact=True)
        np.savez(CACHE_VAL, **_pack(nb_ex), n_cand=nb_ex.n_candidates, seconds=t_ex, key=key)

    # the features from this run must equal the saved lsh__grouped.csv (same code, same split)
    saved = lf.load_lsh_features("grouped").set_index("doc_id").loc[df["doc_id"].to_numpy()[val]]
    mine = index.features(nb_lsh)
    for c in lf.FEATURE_COLUMNS:
        assert np.allclose(mine[c].to_numpy(), saved[c].to_numpy(), atol=1e-5), f"{c} differs from lsh__grouped.csv"
    print("  features of this run match data/processed/features/lsh__grouped.csv")

    # recall of the feature index on the val queries (directed pairs val -> train)
    n_index = len(train)

    def directed(nb):
        qq = np.repeat(np.arange(len(nb.rows)), [len(r) for r in nb.rows])
        return qq * n_index + np.concatenate(nb.rows), np.concatenate(nb.sims)

    codes_ex, j_ex = directed(nb_ex)
    codes_lsh, _ = directed(nb_lsh)
    found = np.isin(codes_ex, codes_lsh)
    feat_rows = []
    for t in FEATURE_THRESHOLDS:
        above = j_ex >= t
        lo, hi = lsh_eval.wilson_interval(int((found & above).sum()), int(above.sum()))
        feat_rows.append({"data": "val (grouped) -> train", "method": "lsh_prefilter", "seconds": t_lsh,
                          "ms_per_article": 1000 * t_lsh / len(val),
                          "candidates_per_article": float(nb_lsh.n_candidates.mean()), "threshold": t,
                          "true_pairs": int(above.sum()), "found": int((found & above).sum()),
                          "recall": (found & above).sum() / above.sum(), "recall_ci_low": lo, "recall_ci_high": hi,
                          "articles_with_neighbor": float(np.mean([len(s) > 0 for s in nb_lsh.sims])),
                          "articles_with_neighbor_exact": float(np.mean([len(s) > 0 for s in nb_ex.sims]))})
    feat_rows.append({"data": "val (grouped) -> train", "method": "exact", "seconds": t_ex,
                      "ms_per_article": 1000 * t_ex / len(val),
                      "candidates_per_article": float(np.mean(nb_ex.n_candidates))})

    # ---- leakage counts: exact (ground truth) and LSH, grouped; LSH for random -------------------------
    y_val = df["is_fake"].to_numpy()[val]
    max_ex = np.array([s[0] if len(s) else 0.0 for s in nb_ex.sims])
    max_lsh = saved["max_jaccard"].to_numpy()
    feats_r = lf.load_lsh_features("random").set_index("doc_id")
    val_r = feats_r.index[feats_r["split"] == "val"]
    y_r = df.set_index("doc_id")["is_fake"].loc[val_r].to_numpy()
    max_r = feats_r.loc[val_r, "max_jaccard"].to_numpy()
    counts = []
    for regime, source, mj, y in (("grouped", "exact search", max_ex, y_val), ("grouped", "LSH features", max_lsh, y_val),
                                  ("random", "LSH features", max_r, y_r)):
        for t in LEAK_THRESHOLDS:
            m = mj >= t
            counts.append({"regime": regime, "source": source, "threshold": t, "val_articles": int(m.sum()),
                           "share_of_val": m.mean(), "fake": int((m & (y == 1)).sum()),
                           "share_of_fake_val": m[y == 1].mean(), "real": int((m & (y == 0)).sum()),
                           "share_of_real_val": m[y == 0].mean()})
    counts = pd.DataFrame(counts)

    # ---- fake_neighbor_share on the leak subsets vs the rest (the feature as the model sees it) --------
    def masks(mj):
        return {"max J >= 0.5": mj >= 0.5, "0.3 <= max J < 0.5": (mj >= 0.3) & (mj < 0.5), "max J >= 0.3": mj >= 0.3,
                "rest: max J < 0.3": mj < 0.3, "rest with a neighbour: 0.1 <= max J < 0.3": (mj >= 0.1) & (mj < 0.3),
                "all val": np.ones(len(mj), bool)}

    scores = pd.concat([lsh_eval.subset_scores(saved.reset_index(), y_val, masks(max_lsh)).assign(regime="grouped"),
                        lsh_eval.subset_scores(feats_r.loc[val_r].reset_index(), y_r, masks(max_r)).assign(regime="random")])

    # ---- what the remaining grouped leaks are: the closest pair of each val article with max J >= 0.5 ----
    pairs = leak_pairs(df, val, train, nb_ex, max_ex)
    summary["leakage"] = {
        "grouped_exact": {f"ge_{t}": int((max_ex >= t).sum()) for t in LEAK_THRESHOLDS},
        "grouped_lsh": {f"ge_{t}": int((max_lsh >= t).sum()) for t in LEAK_THRESHOLDS},
        "random_lsh": {f"ge_{t}": int((max_r >= t).sum()) for t in LEAK_THRESHOLDS},
        "val_articles": len(val), "val_lsh_ms_per_article": round(1000 * t_lsh / len(val), 2),
        "val_exact_ms_per_article": round(1000 * t_ex / len(val), 2),
        "leak_pairs_ge_0.5": {"n": len(pairs), "same_label": float((pairs["is_fake_val"] == pairs["is_fake_train"]).mean()),
                              "containment3_ge_0.7": float((pairs["containment_3w"] >= 0.7).mean()),
                              "median_jaccard_3w": float(pairs["jaccard_3w"].median()),
                              "median_cosine": float(pairs["cosine"].median()) if "cosine" in pairs else None,
                              "by_kind": pairs["kind"].value_counts().to_dict() if "kind" in pairs else None},
    }
    grid = np.round(np.arange(0.1, 1.0001, 0.02), 2)
    curves = pd.DataFrame({"threshold": grid, "grouped": lsh_eval.share_above(max_lsh, grid),
                           "random": lsh_eval.share_above(max_r, grid)})
    return pd.DataFrame(feat_rows), counts, scores, pairs, curves


def leak_pairs(df: pd.DataFrame, val, train, nb_ex: lf.Neighbors, max_ex) -> pd.DataFrame:
    """For every grouped val article with an exact 2-word train neighbour at J >= 0.5: the closest pair, its
    3-word Jaccard and containment, embedding cosine and labels (why it is not one near-duplicate group)."""
    from fake_news.similarity.paraphrases import load_embeddings_checked

    sel = np.flatnonzero(max_ex >= 0.5)
    va = val[sel]
    tr = train[[nb_ex.rows[i][0] for i in sel]]
    content, ids = df["content"].to_numpy(), df["doc_id"].to_numpy()
    j3, c3 = [], []
    for a, b in zip(va, tr):
        sa, sb = shingle_ids(content[a], 3), shingle_ids(content[b], 3)
        inter = len(np.intersect1d(sa, sb, assume_unique=True))
        j3.append(inter / (len(sa) + len(sb) - inter) if len(sa) + len(sb) else 0.0)
        c3.append(inter / min(len(sa), len(sb)) if min(len(sa), len(sb)) else 0.0)
    out = pd.DataFrame({"doc_val": ids[va], "doc_train": ids[tr], "jaccard_2w": max_ex[sel], "jaccard_3w": j3,
                        "containment_3w": c3, "is_fake_val": df["is_fake"].to_numpy()[va],
                        "is_fake_train": df["is_fake"].to_numpy()[tr],
                        "n_words_val": df["n_words"].to_numpy()[va], "n_words_train": df["n_words"].to_numpy()[tr]})
    try:
        X = load_embeddings_checked(ids)
        out["cosine"] = np.einsum("ij,ij->i", X[va], X[tr])
        # Descriptive kind of each pair (a heuristic, for reading the table; not used anywhere else):
        # short teasers share most words by construction; a high cosine means the same story; otherwise the two
        # articles are about different things and share template text (footers, sidebars, site boilerplate).
        short = np.minimum(out["n_words_val"], out["n_words_train"]) < SHORT_WORDS
        out["kind"] = np.select([short, out["cosine"] >= SAME_STORY_COSINE],
                                [f"short text (< {SHORT_WORDS} words)", f"same story (cosine >= {SAME_STORY_COSINE})"],
                                f"shared template text (cosine < {SAME_STORY_COSINE})")
    except Exception as e:   # embeddings missing: the audit still works without cosine
        print(f"  (no embedding cosine: {e})")
    out["title_val"] = [nd._shorten(t, 90) for t in df["title"].to_numpy()[va]]
    out["title_train"] = [nd._shorten(t, 90) for t in df["title"].to_numpy()[tr]]
    return out.sort_values("jaccard_2w", ascending=False).round(4).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-only", action="store_true", help="only the 3,000-article sample (stages 1-2)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = load_clean()
    rng = np.random.default_rng(config.SEED)
    summary = {"settings": {"sample_size": SAMPLE_SIZE, "seed": config.SEED, "n_hashes": minhash.N_HASHES,
                            "bandings_3word": [f"{b}x{r}" for b, r in BANDINGS],
                            "feature_index": f"{lf.BANDS}x{lf.ROWS}, 2-word, min_sim {lf.MIN_SIM}, "
                                             f"prefilter {lf.MIN_AGREE}/128",
                            "min_shingles": nd.MIN_SHINGLES, "full_min_j": FULL_MIN_J}}

    rows, retrieval, s_curves, runtime = stage_sample(df, rng, summary)
    feat_table, sc2 = stage_sample_2word(df, rows, summary)
    s_curves.append(sc2)
    feat_tables = [feat_table]
    if not args.sample_only:
        r, s, t = stage_full(df, rng, summary)
        retrieval += r
        s_curves += s
        runtime += t
        ft, counts, scores, pairs, curves = stage_leakage(df, summary)
        feat_tables.append(ft)
        counts.round(4).to_csv(OUT / "leakage_audit_counts.csv", index=False)
        scores.round(4).to_csv(OUT / "leakage_audit_scores.csv", index=False)
        pairs.to_csv(OUT / "leakage_audit_pairs.csv", index=False)
        fig = plots.leak_share_curves(curves)
        print("  saved", plots.save(fig, "leakage_max_jaccard"))
        print(counts.round(3).to_string(index=False))
        print(scores.round(3).to_string(index=False))

    retrieval = pd.concat(retrieval).round(4)
    s_curves = pd.concat(s_curves).round(4)
    pd.DataFrame(runtime).round(3).to_csv(OUT / "lsh_eval_runtime.csv", index=False)
    retrieval.to_csv(OUT / "lsh_eval_retrieval.csv", index=False)
    s_curves.to_csv(OUT / "lsh_eval_s_curve.csv", index=False)
    pd.concat(feat_tables).round(4).to_csv(OUT / "lsh_eval_feature_index.csv", index=False)
    fig = plots.s_curves(s_curves)
    print("\nsaved", plots.save(fig, "lsh_eval_s_curves"))

    summary["seconds"] = round(time.perf_counter() - t0, 1)
    if not args.sample_only:
        (OUT / "lsh_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(retrieval.to_string(index=False))
    print(pd.DataFrame(runtime).round(2).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
