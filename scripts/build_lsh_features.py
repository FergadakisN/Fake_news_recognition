"""Build the LSH neighbour features (input for the final classifier) for one or more split regimes.

    python scripts/build_lsh_features.py --regime random                 # split_random from splits.csv
    python scripts/build_lsh_features.py --regime random grouped         # several regimes, shingles computed once
    python scripts/build_lsh_features.py --regime grouped --seed 1 2     # splits generated in code from seeds
    python scripts/build_lsh_features.py --regime cluster --seed 1 --clusters clusters.csv

Without --seed the split is read from data/processed/splits.csv (split_<regime>); the script stops with a
message if that column does not exist yet. With --seed the split is built in code with make_random_split /
make_group_split (near-duplicate groups; for cluster, the K-means clusters from --clusters, a CSV with
doc_id, cluster) and nothing is written to splits.csv.

For each article in train, val and test: fake_neighbor_share, max_jaccard, neighbor_count from a train-only
LSH index (2-word shingles, 64 x 2 bands, exact Jaccard >= 0.1). Training articles exclude themselves
(random) or their whole near-duplicate group (grouped, cluster). Details: fake_news.similarity.lsh_features.
About 10 minutes per split (plus ~1 minute of shingling once).

Outputs:
    data/processed/features/lsh__<regime>[__seed<N>].csv         doc_id, split, the 3 features
    reports/minhash/lsh_features_sanity_<regime>[__seed<N>].csv  per partition: no-neighbour share, max_jaccard,
                                                                 fake_neighbor_share of fake vs real articles
    reports/minhash/lsh_features_<regime>[__seed<N>].json        settings, fallback, runtime
"""

import argparse
import json
import sys
import time

import pandas as pd

from fake_news import config
from fake_news.data import load_clean, load_splits
from fake_news.data.split import check_split, make_group_split, make_random_split
from fake_news.similarity import lsh_features as lf
from fake_news.similarity.near_duplicates import REPORT_DIR, load_groups


def get_split(regime: str, seed: int | None, clean: pd.DataFrame, groups: pd.Series,
              clusters_csv: str | None) -> pd.Series:
    """doc_id -> train/val/test: read from splits.csv (seed None) or generated in code from ``seed``."""
    if seed is None:
        splits = load_splits()
        col = f"split_{regime}"
        if col not in splits.columns:
            available = [c.removeprefix("split_") for c in splits.columns if c.startswith("split_")]
            raise SystemExit(f"{col} is not in splits.csv yet (available: {available}). Pull the latest "
                             f"splits.csv, or pass --seed to generate a split in code.")
        return splits.set_index("doc_id")[col]

    if regime == "random":
        split = make_random_split(clean, seed=seed)
        check_split(split)
    elif regime == "grouped":
        split = make_group_split(clean, groups, seed=seed)
        check_split(split, groups=groups)
    else:  # cluster: whole K-means clusters held out
        if clusters_csv is None:
            raise SystemExit("--regime cluster with --seed needs the K-means clusters: --clusters <csv with doc_id, cluster>")
        clusters = pd.read_csv(clusters_csv).set_index("doc_id")["cluster"]
        split = make_group_split(clean, clusters, seed=seed, stratify=False)
        check_split(split, groups=clusters)
    return split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", nargs="+", default=["random"], choices=config.REGIMES)
    ap.add_argument("--seed", nargs="+", type=int, default=None,
                    help="generate the split(s) in code from these seeds instead of reading splits.csv")
    ap.add_argument("--clusters", default=None, help="CSV doc_id, cluster (only for --regime cluster --seed)")
    args = ap.parse_args()
    seeds = args.seed or [None]

    clean = load_clean(columns=["content"])
    labels = clean.set_index("doc_id")["is_fake"]
    groups = load_groups()            # near-duplicate groups: for exclude="group" and the sanity report
    # Check every requested split exists before spending time on shingling.
    splits = {(r, s): get_split(r, s, clean, groups, args.clusters) for r in args.regime for s in seeds}

    t0 = time.perf_counter()
    enc = lf.encode(clean["content"].tolist(), shingle_k=lf.SHINGLE_WORDS)
    print(f"{lf.SHINGLE_WORDS}-word shingles + signatures of {len(clean):,} articles: {time.perf_counter() - t0:.0f} s")

    for (regime, seed), split in splits.items():
        tag = regime + (f"__seed{seed}" if seed is not None else "")
        exclude = lf.exclusion_for_regime(regime)
        print(f"\n[{tag}] exclude={exclude!r}", flush=True)
        feats = lf.build_lsh_features(clean, split, exclude=exclude, groups=groups, encoded=enc, verbose=True)
        path = lf.save_lsh_features(feats, regime, seed)

        sanity = lf.sanity_report(feats, labels, groups)
        sanity.round(4).to_csv(REPORT_DIR / f"lsh_features_sanity_{tag}.csv")
        meta = {"regime": regime, "seed": seed, "split_source": "splits.csv" if seed is None else "generated",
                "exclude": exclude, "shingle_words": lf.SHINGLE_WORDS, "bands": lf.BANDS, "rows": lf.ROWS,
                "min_sim": lf.MIN_SIM, "min_agree": lf.MIN_AGREE, "fallback_train_fake_rate": round(feats.attrs["fallback"], 6),
                "seconds": feats.attrs["seconds"], "ms_per_query": round(feats.attrs["ms_per_query"], 2),
                "candidates_per_query": round(feats.attrs["candidates_per_query"], 1)}
        (REPORT_DIR / f"lsh_features_{tag}.json").write_text(json.dumps(meta, indent=2))

        print(f"  saved {path.relative_to(config.PROJECT_ROOT)} ({feats.attrs['seconds']:.0f} s, "
              f"{feats.attrs['ms_per_query']:.1f} ms/article)")
        with pd.option_context("display.width", 250):
            print(sanity.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
