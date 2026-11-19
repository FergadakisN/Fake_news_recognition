"""Find near-duplicate articles with MinHash + LSH and write data/processed/near_dup_groups.csv.

    python scripts/find_near_duplicates.py                    # proposed threshold (0.7)
    python scripts/find_near_duplicates.py --threshold 0.8    # another threshold
    python scripts/find_near_duplicates.py --rebuild          # recompute the signatures (ignore the cache)

Stages (each one timed):
    1. load      cleaned articles (load_clean)
    2. minhash   3-word shingles -> 128 MinHash values per article (cached in data/interim/)
    3. lsh       16 bands x 8 rows -> candidate pairs
    4. verify    exact Jaccard of every candidate pair
    5. recall    the same signatures with a looser banding (32 x 4) to measure what 16 x 8 misses
    6. groups    pairs with Jaccard >= threshold -> union-find -> doc_id -> group_id

Outputs:
    data/processed/near_dup_groups.csv          doc_id, group_id (all articles; singletons: group_id = doc_id)
    reports/figures/minhash_candidate_jaccard.png
    reports/minhash/near_dup_summary.json       settings, timings, statistics
    reports/minhash/threshold_examples.csv      10 pairs just above and 10 just below 0.8
    reports/minhash/threshold_comparison.csv    group statistics for several thresholds
    reports/minhash/largest_groups.csv          the 25 largest groups and all mixed-label groups

This script never touches data/processed/splits.csv. The grouped split is added by Person 1.
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.data import load_clean
from fake_news.similarity import lsh, minhash, plots
from fake_news.similarity import near_duplicates as nd
from fake_news.similarity.shingles import K

PROPOSED_THRESHOLD = 0.7
EXAMPLES_AROUND = 0.8


@contextmanager
def stage(name: str, seconds: dict):
    """Print the stage name, run the block, then print and record how long it took."""
    print(f"\n[{name}]", flush=True)
    t0 = time.perf_counter()
    yield
    seconds[name] = round(time.perf_counter() - t0, 2)
    print(f"  ({seconds[name]:.1f} s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=PROPOSED_THRESHOLD,
                    help=f"exact Jaccard needed to join two articles (default {PROPOSED_THRESHOLD})")
    ap.add_argument("--rebuild", action="store_true", help="recompute signatures even if the cache is up to date")
    args = ap.parse_args()
    th = args.threshold
    seconds: dict[str, float] = {}
    nd.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with stage("load", seconds):
        df = load_clean(columns=["title", "content", "n_words"])
        labels = df.set_index("doc_id")["is_fake"]
        print(f"  {len(df):,} cleaned articles, {labels.mean():.1%} fake")

    with stage("minhash", seconds):
        table = nd.compute_signatures(df, rebuild=args.rebuild)
        n_short = len(table.short_doc_ids)
        print(f"  {'read from cache' if table.from_cache else 'computed'}: signatures {table.signatures.shape}")
        print(f"  {table.n_shingles.sum():,} shingles in total; {n_short} articles with < {nd.MIN_SHINGLES} "
              f"shingles get no signature (stay singletons)")

    with stage("lsh", seconds):
        cands = nd.find_candidates(table)
        sizes = lsh.bucket_sizes(table.signatures)
        print(f"  {len(cands):,} candidate pairs (of {len(table.doc_ids) * (len(table.doc_ids) - 1) // 2:,} possible)")
        print(f"  {len(sizes):,} buckets with 2+ articles; largest bucket: {sizes.max()} articles")

    with stage("verify", seconds):
        cands = nd.verify_candidates(cands, df)
        err = (cands["minhash"] - cands["jaccard"]).abs()
        print(f"  exact Jaccard: median {cands.jaccard.median():.3f}; MinHash estimate mean |error| {err.mean():.3f}")
        cands.to_csv(config.INTERIM_DIR / "minhash_candidates.csv", index=False)  # for the notebook, not in git

    with stage("recall", seconds):
        loose = nd.verify_candidates(nd.find_candidates(table, bands=32, rows=4), df)
        recall = nd.lsh_recall(cands, loose)
        print(f"  32 x 4 banding gives {len(loose):,} candidates. Share of its pairs that 16 x 8 also finds:")
        print("  " + recall.to_string(index=False).replace("\n", "\n  "))

    # ---- threshold diagnostics: histogram + examples around 0.8 ----
    counts, edges = np.histogram(cands["jaccard"], bins=np.arange(0, 1.0001, 0.05))
    print("\nExact Jaccard of candidate pairs:")
    for lo, c in zip(edges[:-1], counts):
        if c:
            print(f"  {lo:.2f}-{lo + 0.05:.2f} {c:6,} {'#' * max(1, int(60 * c / counts.max()))}")
    fig = plots.jaccard_histogram(cands, threshold=th, reference=EXAMPLES_AROUND,
                                  recall=nd.binned_recall(cands, loose))
    print(f"  figure: {plots.save(fig, 'minhash_candidate_jaccard')}")

    examples = nd.threshold_examples(cands, df, around=EXAMPLES_AROUND, n=10)
    examples.to_csv(nd.REPORT_DIR / "threshold_examples.csv", index=False)
    print(f"\nPairs just above / below {EXAMPLES_AROUND} (full table: reports/minhash/threshold_examples.csv):")
    with pd.option_context("display.width", 250, "display.max_colwidth", 45):
        print(examples[["side", "jaccard", "doc_a", "doc_b", "is_fake_a", "is_fake_b", "n_words_a", "n_words_b",
                        "only_in_a", "only_in_b"]].to_string(index=False))

    comparison = nd.threshold_table(df["doc_id"], cands, labels)
    comparison.to_csv(nd.REPORT_DIR / "threshold_comparison.csv")
    print("\nGroups for several thresholds:")
    print(comparison.drop(columns=["n_articles"]).to_string())

    with stage("groups", seconds):
        groups = nd.make_groups(df["doc_id"], cands, th)
        summary = nd.group_summary(groups, labels)
        gt = nd.group_table(groups, labels)
        path = nd.save_groups(groups)

    print(f"\nThreshold {th}:")
    print(f"  groups with 2+ articles : {summary['groups_2plus']:,}")
    print(f"  articles in them        : {summary['articles_in_groups_2plus']:,} "
          f"({summary['share_articles_in_groups']:.1%} of all)")
    print(f"  largest group           : {summary['largest_group']}")
    print(f"  mixed fake/real groups  : {summary['mixed_label_groups']} "
          f"({summary['articles_in_mixed_groups']} articles)")
    print(f"  written: {path.relative_to(config.PROJECT_ROOT)}")

    # Largest groups + all mixed ones, with an example title and the start of the text, for inspection.
    by_id = df.set_index("doc_id")
    show = gt.head(25).index.union(gt.index[gt["mixed"]])
    show = gt.loc[show].sort_values("size", ascending=False).copy()
    show["example_title"] = [str(by_id.at[g, "title"])[:100] for g in show.index]
    show["content_start"] = [str(by_id.at[g, "content"])[:200] for g in show.index]
    show.to_csv(nd.REPORT_DIR / "largest_groups.csv")

    report = {
        "settings": {"shingle_words": K, "n_hashes": minhash.N_HASHES, "bands": lsh.BANDS, "rows": lsh.ROWS,
                     "seed": config.SEED, "min_shingles": nd.MIN_SHINGLES, "threshold": th},
        "seconds": seconds,
        "signatures_from_cache": table.from_cache,
        "n_articles": len(df),
        "articles_without_signature": n_short,
        "candidate_pairs": len(cands),
        "buckets_2plus": len(sizes),
        "largest_bucket": int(sizes.max()),
        "minhash_mean_abs_error": round(float(err.mean()), 4),
        "jaccard_histogram": {f"{lo:.2f}": int(c) for lo, c in zip(edges[:-1], counts)},
        "lsh_recall_vs_32x4": recall.to_dict(orient="records"),
        "groups": summary,
    }
    (nd.REPORT_DIR / "near_dup_summary.json").write_text(json.dumps(report, indent=2))
    print(f"  summary: {(nd.REPORT_DIR / 'near_dup_summary.json').relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
