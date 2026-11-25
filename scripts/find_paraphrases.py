"""Embedding tier of the near-duplicate groups: find paraphrases (same story, other words) that MinHash misses.

    python scripts/find_paraphrases.py                     # proposed cosine threshold 0.92
    python scripts/find_paraphrases.py --threshold 0.94    # examples and statistics around another threshold

Steps:
    1. exact top-10 cosine neighbours of every article (MiniLM embeddings; cached in data/interim/, ~1 min)
    2. profile of the pairs by cosine: how many MinHash already groups, their 3-word Jaccard, mixed labels
    3. for several thresholds: the two-tier groups (MinHash groups + embedding pairs, union-find): number,
       size, largest group (chaining), mixed fake/real groups
    4. 10 pairs just above and 10 just below --threshold, among pairs MinHash has NOT already grouped
       (only those are decided by the threshold), with their 3-word Jaccard

Nothing is written to data/processed/ yet: the threshold is chosen first.

Outputs (reports/minhash/):
    embedding_cosine_profile.csv         per cosine bin (pairs with cosine >= 0.80)
    embedding_threshold_comparison.csv   two-tier group statistics per threshold
    embedding_threshold_examples.csv     the 20 pairs around --threshold
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from fake_news.data import load_clean
from fake_news.similarity import paraphrases as pp
from fake_news.similarity.near_duplicates import REPORT_DIR, group_table, load_groups

PROPOSED_THRESHOLD = 0.92
PROFILE_BINS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0001]
COMPARE = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=PROPOSED_THRESHOLD)
    args = ap.parse_args()
    th = args.threshold

    df = load_clean(columns=["title", "content", "n_words"])
    labels = df.set_index("doc_id")["is_fake"]
    ids = df["doc_id"].to_numpy()
    minhash = load_groups()

    t0 = time.perf_counter()
    idx, sim = pp.load_or_compute_topk(ids)
    print(f"top-{pp.TOP_K} cosine neighbours of {len(ids):,} articles: {time.perf_counter() - t0:.0f} s")
    print("nearest-neighbour cosine, quantiles 10/25/50/75/90/99%:",
          np.quantile(sim[:, 0], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).tolist())

    # ---- profile by cosine ----
    pairs = pp.add_pair_info(pp.embedding_pairs(ids, idx, sim, PROFILE_BINS[0]), df, minhash)
    pairs["bin"] = pd.cut(pairs["cosine"], PROFILE_BINS, right=False)
    profile = pairs.groupby("bin", observed=True).agg(
        pairs=("cosine", "size"), share_already_minhash=("same_minhash_group", "mean"),
        median_jaccard=("jaccard", "median"), share_jaccard_below_0_3=("jaccard", lambda s: (s < 0.3).mean()),
        share_mixed_labels=("mixed_labels", "mean"))
    profile.round(4).to_csv(REPORT_DIR / "embedding_cosine_profile.csv")
    print("\nEmbedding pairs by cosine (3-word Jaccard = the MinHash tier's measure):")
    print(profile.round(3).to_string())

    # ---- two-tier groups per threshold ----
    base = group_table(minhash, labels)
    rows = []
    for t in COMPARE:
        p = pairs[pairs["cosine"] >= t]
        gt = group_table(pp.merge_groups(ids, minhash, p), labels)
        rows.append({"threshold": t, "embedding_pairs": len(p), "new_links": int((~p["same_minhash_group"]).sum()),
                     "groups_2plus": len(gt), "articles_in_groups": int(gt["size"].sum()),
                     "articles_added": int(gt["size"].sum() - base["size"].sum()),
                     "largest_group": int(gt["size"].max()), "mixed_label_groups": int(gt["mixed"].sum())})
    comparison = pd.DataFrame(rows).set_index("threshold")
    comparison.to_csv(REPORT_DIR / "embedding_threshold_comparison.csv")
    print(f"\nTwo-tier groups (MinHash only: {len(base)} groups, {int(base['size'].sum())} articles, "
          f"largest {int(base['size'].max())}, {int(base['mixed'].sum())} mixed):")
    print(comparison.to_string())

    # ---- examples around the threshold (pairs MinHash has not already grouped) ----
    new = pairs[~pairs["same_minhash_group"]]
    above = new[new["cosine"] >= th].nsmallest(10, "cosine").assign(side="above")
    below = new[new["cosine"] < th].nlargest(10, "cosine").assign(side="below")
    ex = pd.concat([above.sort_values("cosine", ascending=False), below])
    d = df.set_index("doc_id")
    short = lambda s, n: s if len(s) <= n else s[:n - 2] + " …"
    for c, src in (("a", "doc_a"), ("b", "doc_b")):
        ex[f"text_{c}"] = [short(t, 160) for t in d["content"].loc[ex[src]]]
    ex = ex[["side", "cosine", "jaccard", "doc_a", "doc_b", "is_fake_a", "is_fake_b", "n_words_a", "n_words_b",
             "text_a", "text_b"]].round({"cosine": 3, "jaccard": 3})
    ex.to_csv(REPORT_DIR / "embedding_threshold_examples.csv", index=False)
    print(f"\n10 pairs just above and 10 just below cosine {th} (not already grouped by MinHash):")
    for _, r in ex.iterrows():
        lab = lambda f: "fake" if f else "real"
        print(f"\n  [{r.side}] cosine {r.cosine:.3f}  Jaccard {r.jaccard:.3f}  "
              f"{r.doc_a} ({lab(r.is_fake_a)}, {r.n_words_a}w) / {r.doc_b} ({lab(r.is_fake_b)}, {r.n_words_b}w)")
        print(f"     A: {r.text_a}\n     B: {r.text_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
