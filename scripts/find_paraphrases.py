"""Embedding tier of the near-duplicate groups: find paraphrases (same story, other words) that MinHash misses.

    python scripts/find_paraphrases.py --threshold 0.90 --write   # the chosen threshold: write the final groups
    python scripts/find_paraphrases.py --threshold 0.92           # analysis only: statistics and examples

Steps:
    1. exact top-10 cosine neighbours of every article (MiniLM embeddings; cached in data/interim/, ~1 min)
    2. profile of the pairs by cosine: how many MinHash already groups, their 3-word Jaccard, mixed labels
    3. for several thresholds: the two-tier groups (MinHash groups + embedding pairs, union-find): number,
       size, largest group (chaining), mixed fake/real groups
    4. 10 pairs just above and 10 just below --threshold, among pairs MinHash has NOT already grouped
       (only those are decided by the threshold), with their 3-word Jaccard

    5. with --write: the final two-tier groups (MinHash groups + embedding pairs >= --threshold, union-find)
       with the tier that linked each group, and the statistics of the merge

The MinHash tier is read from data/processed/near_dup_groups_minhash.csv (scripts/find_near_duplicates.py).

Outputs (reports/minhash/):
    embedding_cosine_profile.csv         per cosine bin (pairs with cosine >= 0.80)
    embedding_threshold_comparison.csv   two-tier group statistics per threshold
    embedding_threshold_examples.csv     the 20 pairs around --threshold
  with --write:
    data/processed/near_dup_groups.csv   doc_id, group_id, tier (none / minhash / embedding / both)
    two_tier_summary.json                groups and articles per tier, largest group, mixed groups
    two_tier_largest_groups.csv          the 25 largest groups and all mixed-label groups, with tier and an example
    paraphrase_examples.csv              5 paraphrases MinHash misses (cosine >= threshold, 3-word Jaccard < 0.3)
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from fake_news.data import load_clean
from fake_news.similarity import paraphrases as pp
from fake_news.similarity.near_duplicates import REPORT_DIR, group_table, load_minhash_groups

PROPOSED_THRESHOLD = 0.92
PROFILE_BINS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0001]
COMPARE = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=PROPOSED_THRESHOLD)
    ap.add_argument("--write", action="store_true", help="write the final groups to data/processed/near_dup_groups.csv")
    args = ap.parse_args()
    th = args.threshold

    df = load_clean(columns=["title", "content", "n_words"])
    labels = df.set_index("doc_id")["is_fake"]
    ids = df["doc_id"].to_numpy()
    minhash = load_minhash_groups()

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

    if not args.write:
        return 0

    # ---- final two-tier groups -----------------------------------------------------------------------------
    chosen = pairs[pairs["cosine"] >= th]
    groups = pp.two_tier_groups(ids, minhash, chosen)
    path = pp.save_two_tier_groups(groups)
    final = groups.set_index("doc_id")["group_id"]
    gt = group_table(final, labels)
    gt["tier"] = groups.drop_duplicates("group_id").set_index("group_id")["tier"].reindex(gt.index)
    by_tier = gt.groupby("tier").agg(groups=("size", "size"), articles=("size", "sum"), mixed=("mixed", "sum"))

    # MinHash groups (2+ articles) that the embedding tier enlarged or joined: their final group has tier "both"
    mh_ids = base.index                                   # MinHash group ids = smallest doc_id of the group
    mh_enlarged = int((gt["tier"].reindex(final.loc[mh_ids].to_numpy()) == "both").sum())

    summary = {
        "threshold_cosine": th, "minhash_threshold_jaccard": 0.7, "top_k": pp.TOP_K,
        "embedding_pairs": len(chosen), "embedding_pairs_new_links": int((~chosen["same_minhash_group"]).sum()),
        "minhash_tier_alone": {"groups_2plus": len(base), "articles": int(base["size"].sum()),
                               "largest_group": int(base["size"].max()), "mixed_label_groups": int(base["mixed"].sum())},
        "final": {"groups_2plus": len(gt), "articles": int(gt["size"].sum()),
                  "share_of_all_articles": round(float(gt["size"].sum() / len(ids)), 4),
                  "largest_group": int(gt["size"].max()), "mixed_label_groups": int(gt["mixed"].sum()),
                  "articles_in_mixed_groups": int(gt.loc[gt["mixed"], "size"].sum())},
        "by_tier": {t: {k: int(v) for k, v in r.items()} for t, r in by_tier.iterrows()},
        "minhash_groups_enlarged_or_joined": mh_enlarged,
    }
    (REPORT_DIR / "two_tier_summary.json").write_text(json.dumps(summary, indent=2))

    d = df.set_index("doc_id")
    show = gt.loc[gt.head(25).index.union(gt.index[gt["mixed"]])].sort_values("size", ascending=False).copy()
    show["example_title"] = [str(d.at[g, "title"])[:100] for g in show.index]
    show.to_csv(REPORT_DIR / "two_tier_largest_groups.csv")

    # 5 paraphrases MinHash misses: clearly different wording (Jaccard < 0.15) at similar length (at most 2x),
    # highest cosine first; 2 real/real, 2 fake/fake and 1 mixed pair, so both labels are represented.
    # Pairs with very different lengths are mostly truncated copies (same opening, one text much longer):
    # MiniLM only reads the opening, while Jaccard is low because one set is much bigger. Counted separately.
    ratio = np.maximum(chosen["n_words_a"], chosen["n_words_b"]) / np.maximum(np.minimum(chosen["n_words_a"], chosen["n_words_b"]), 1)
    not_mh = ~chosen["same_minhash_group"]
    summary["embedding_links_low_jaccard_truncated"] = int((not_mh & (chosen["jaccard"] < 0.3) & (ratio > 2)).sum())
    summary["embedding_links_low_jaccard_similar_length"] = int((not_mh & (chosen["jaccard"] < 0.3) & (ratio <= 2)).sum())
    (REPORT_DIR / "two_tier_summary.json").write_text(json.dumps(summary, indent=2))
    miss = chosen[not_mh & (chosen["jaccard"] < 0.15) & (ratio <= 2)].sort_values("cosine", ascending=False)
    kinds = [((miss.is_fake_a == 0) & (miss.is_fake_b == 0), 2), ((miss.is_fake_a == 1) & (miss.is_fake_b == 1), 2),
             (miss["mixed_labels"], 1)]
    ex = pd.concat([miss[m].head(n) for m, n in kinds])
    for c, src in (("a", "doc_a"), ("b", "doc_b")):
        ex[f"text_{c}"] = [short(t, 220) for t in d["content"].loc[ex[src]]]
    ex[["cosine", "jaccard", "doc_a", "doc_b", "is_fake_a", "is_fake_b", "text_a", "text_b"]].round(3).to_csv(
        REPORT_DIR / "paraphrase_examples.csv", index=False)

    print(f"\nwritten {path.name}: {summary['final']['groups_2plus']:,} groups with 2+ articles, "
          f"{summary['final']['articles']:,} articles ({summary['final']['share_of_all_articles']:.1%})")
    print(by_tier.to_string())
    print(json.dumps(summary, indent=2))
    print("\nlargest groups:")
    print(show.head(8)[["size", "n_fake", "mixed", "tier", "example_title"]].to_string())
    print("\nparaphrases MinHash misses:")
    for _, r in ex.iterrows():
        lab = lambda f: "fake" if f else "real"
        print(f"\n  cosine {r.cosine:.3f}  Jaccard {r.jaccard:.3f}  {r.doc_a} ({lab(r.is_fake_a)}) / {r.doc_b} ({lab(r.is_fake_b)})")
        print(f"     A: {r.text_a}\n     B: {r.text_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
