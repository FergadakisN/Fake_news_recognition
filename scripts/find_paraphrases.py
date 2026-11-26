"""Embedding tier of the near-duplicate groups: find paraphrases (same story, other words) that MinHash misses.

    python scripts/find_paraphrases.py --write                    # chosen rules: write the final groups
    python scripts/find_paraphrases.py --threshold 0.90           # analysis only: statistics and examples

Steps:
    1. exact top-10 cosine neighbours of every article (MiniLM embeddings; cached in data/interim/, ~1 min)
    2. profile of the pairs by cosine: how many MinHash already groups, their 3-word Jaccard, mixed labels
    3. for several thresholds: the two-tier groups (MinHash groups + embedding pairs, union-find): number,
       size, largest group (chaining), mixed fake/real groups
    4. 10 pairs just above and 10 just below --threshold, among pairs MinHash has NOT already grouped
       (only those are decided by the threshold), with their 3-word Jaccard

    5. with --write: the final two-tier groups, merged with union-find from the MinHash groups and the embedding
       links: cosine >= --threshold (0.92), or 0.90 <= cosine < 0.92 with 3-word shingle containment >= 0.7
       (paraphrases.select_links; recovers copies just below the threshold without chaining topics; turn it
       off with --no-containment). Each group records the tier(s) that linked it. Each embedding link MinHash did
       not have is described by its containment (share of the shorter text's shingles found in the other):
       truncated copy / edited copy / partial overlap / paraphrase (paraphrases.link_type)

The MinHash tier is read from data/processed/near_dup_groups_minhash.csv (scripts/find_near_duplicates.py).

Outputs (reports/minhash/):
    embedding_cosine_profile.csv         per cosine bin (pairs with cosine >= 0.80)
    embedding_threshold_comparison.csv   two-tier group statistics per threshold
    embedding_threshold_examples.csv     the 20 pairs around --threshold
  with --write:
    data/processed/near_dup_groups.csv   doc_id, group_id, tier (none / minhash / embedding / both)
    two_tier_summary.json                groups and articles per tier, largest group, mixed groups
    two_tier_largest_groups.csv          the 25 largest groups and all mixed-label groups, with tier and an example
    paraphrase_examples.csv              5 paraphrases MinHash misses (containment < 0.3), both labels
    truncated_copy_examples.csv          5 truncated copies MinHash misses because of length
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

CHOSEN_THRESHOLD = 0.92  # 0.90 chained whole news topics (reports/minhash/LSH_FEATURES_DESIGN.md)
PROFILE_BINS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0001]
COMPARE = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=CHOSEN_THRESHOLD)
    ap.add_argument("--write", action="store_true", help="write the final groups to data/processed/near_dup_groups.csv")
    ap.add_argument("--no-containment", action="store_true",
                    help="do not add the 0.90-0.92 pairs with containment >= 0.7 (cosine threshold only)")
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
    if args.no_containment:
        chosen = pairs[pairs["cosine"] >= th].assign(rule="embedding")
    else:
        chosen = pp.select_links(pairs, threshold=th)
    groups = pp.two_tier_groups(ids, minhash, chosen)
    path = pp.save_two_tier_groups(groups)
    final = groups.set_index("doc_id")["group_id"]
    gt = group_table(final, labels)
    gt["tier"] = groups.drop_duplicates("group_id").set_index("group_id")["tier"].reindex(gt.index)
    by_tier = gt.groupby("tier").agg(groups=("size", "size"), articles=("size", "sum"), mixed=("mixed", "sum"))

    # MinHash groups (2+ articles) that the embedding tier enlarged or joined: final tier "both" or "both+containment"
    mh_ids = base.index                                   # MinHash group ids = smallest doc_id of the group
    mh_enlarged = int(gt["tier"].reindex(final.loc[mh_ids].to_numpy()).str.startswith("both").sum())

    summary = {
        "threshold_cosine": th, "minhash_threshold_jaccard": 0.7, "top_k": pp.TOP_K,
        "containment_rule": None if args.no_containment else
            {"cosine_from": pp.CONTAINMENT_BAND_LOW, "cosine_below": th, "containment_min": pp.CONTAINMENT_MIN},
        "embedding_pairs": len(chosen), "embedding_pairs_new_links": int((~chosen["same_minhash_group"]).sum()),
        "links_by_rule": {k: int(v) for k, v in chosen["rule"].value_counts().items()},
        "minhash_tier_alone": {"groups_2plus": len(base), "articles": int(base["size"].sum()),
                               "largest_group": int(base["size"].max()), "mixed_label_groups": int(base["mixed"].sum())},
        "final": {"groups_2plus": len(gt), "articles": int(gt["size"].sum()),
                  "share_of_all_articles": round(float(gt["size"].sum() / len(ids)), 4),
                  "largest_group": int(gt["size"].max()), "mixed_label_groups": int(gt["mixed"].sum()),
                  "articles_in_mixed_groups": int(gt.loc[gt["mixed"], "size"].sum())},
        "by_tier": {t: {k: int(v) for k, v in r.items()} for t, r in by_tier.iterrows()},
        "minhash_groups_enlarged_or_joined": mh_enlarged,
    }

    d = df.set_index("doc_id")
    show = gt.loc[gt.head(25).index.union(gt.index[gt["mixed"]])].sort_values("size", ascending=False).copy()
    show["example_title"] = [str(d.at[g, "title"])[:100] for g in show.index]
    show.to_csv(REPORT_DIR / "two_tier_largest_groups.csv")

    # What the new embedding links are (descriptive; the grouping itself only uses the cosine threshold)
    chosen = chosen.assign(link_type=pp.link_type(chosen))
    counts = chosen["link_type"].value_counts()
    summary["embedding_links_by_type"] = {t: int(counts.get(t, 0)) for t in pp.LINK_TYPES}
    rec = chosen.loc[chosen["rule"] == "containment", "link_type"].value_counts()
    summary["containment_rule_links_by_type"] = {t: int(rec.get(t, 0)) for t in pp.LINK_TYPES}
    summary["link_type_rules"] = {"copy_containment": pp.COPY_CONTAINMENT, "paraphrase_containment":
                                  pp.PARAPHRASE_CONTAINMENT, "truncation_length_ratio": pp.TRUNCATION_RATIO}
    (REPORT_DIR / "two_tier_summary.json").write_text(json.dumps(summary, indent=2))

    def examples(rows: pd.DataFrame, name: str) -> pd.DataFrame:
        """Highest cosine first; 2 real/real, 2 fake/fake and 1 mixed pair, so both labels are represented."""
        rows = rows.sort_values("cosine", ascending=False)
        kinds = [((rows.is_fake_a == 0) & (rows.is_fake_b == 0), 2), ((rows.is_fake_a == 1) & (rows.is_fake_b == 1), 2),
                 (rows["mixed_labels"], 1)]
        ex = pd.concat([rows[m].head(n) for m, n in kinds])
        for c, src in (("a", "doc_a"), ("b", "doc_b")):
            ex[f"text_{c}"] = [short(t, 220) for t in d["content"].loc[ex[src]]]
        ex = ex[["cosine", "jaccard", "containment", "doc_a", "doc_b", "is_fake_a", "is_fake_b",
                 "n_words_a", "n_words_b", "text_a", "text_b"]].round(3)
        ex.to_csv(REPORT_DIR / name, index=False)
        return ex

    para = examples(chosen[chosen["link_type"] == "paraphrase"], "paraphrase_examples.csv")
    trunc = examples(chosen[chosen["link_type"] == "truncated_copy"], "truncated_copy_examples.csv")

    print(f"\nwritten {path.name}: {summary['final']['groups_2plus']:,} groups with 2+ articles, "
          f"{summary['final']['articles']:,} articles ({summary['final']['share_of_all_articles']:.1%})")
    print(by_tier.to_string())
    print(json.dumps(summary, indent=2))
    print("\nlargest groups:")
    print(show.head(8)[["size", "n_fake", "mixed", "tier", "example_title"]].to_string())
    for title, ex in (("paraphrases MinHash misses", para), ("truncated copies MinHash misses (length)", trunc)):
        print(f"\n{title}:")
        for _, r in ex.iterrows():
            lab = lambda f: "fake" if f else "real"
            print(f"\n  cosine {r.cosine:.3f}  Jaccard {r.jaccard:.3f}  containment {r.containment:.2f}  "
                  f"{r.doc_a} ({lab(r.is_fake_a)}, {r.n_words_a}w) / {r.doc_b} ({lab(r.is_fake_b)}, {r.n_words_b}w)")
            print(f"     A: {r.text_a}\n     B: {r.text_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
