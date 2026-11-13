"""Download WELFake, build the cleaned dataset and (if missing) the random split, then verify it all.

Teammates do not need this script: ``load_dataset()`` downloads, cleans and verifies automatically on
first use. Run it to prepare everything up front and see a full check report, or (Person 1) to
update the reference after changing the cleaning rules.

Usage:
    python3 scripts/prepare_data.py            # download + clean (both variants) + split_random + checks
    python3 scripts/prepare_data.py --rebuild  # rebuild the cleaned CSVs from the raw file

    # Person 1 only, after changing the cleaning rules on purpose (then commit the JSON, the new
    # splits.csv and reports/cleaning/). --new-split recreates splits.csv because the rows changed.
    python3 scripts/prepare_data.py --rebuild --update-fingerprint --new-split

Checks (any failure stops with an explanation):
    1. raw CSV is the pinned Kaggle version (SHA-256)
    2. both cleaned CSVs match data/processed/clean_fingerprint.json
    3. data/processed/splits.csv contains exactly the cleaned doc_ids, with valid partitions

Memory: everything is streamed in chunks of config.CLEAN_CHUNK_ROWS rows, so peak memory stays
around a few hundred MB regardless of the dataset size. The peak is printed at the end.

splits.csv and clean_fingerprint.json are committed to git. An existing `split_random` is never
regenerated, so everyone keeps using the same split.
"""

import argparse
import sys

import pandas as pd

from fake_news import config
from fake_news.data.clean import build_clean_dataset
from fake_news.data.integrity import check_splits_match, verify_clean_file
from fake_news.data.load import download_dataset, load_labels, load_splits
from fake_news.data.split import check_split, make_random_split, save_split_column


def peak_memory_gb() -> float | None:
    """Peak resident memory of this process so far (Linux/macOS; None on Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2  # bytes on macOS, KB on Linux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cleaned CSVs even if they exist")
    parser.add_argument("--update-fingerprint", action="store_true",
                        help="store this build as the new reference (Person 1 only; implies --rebuild)")
    parser.add_argument("--new-split", action="store_true",
                        help="recreate splits.csv from scratch (Person 1 only; when the cleaned rows changed)")
    args = parser.parse_args()

    if args.new_split and config.SPLITS_CSV.exists():
        existing = [c for c in pd.read_csv(config.SPLITS_CSV, nrows=0).columns if c.startswith("split_")]
        if existing != ["split_random"]:
            sys.exit(f"Refusing --new-split: splits.csv already has {existing}; those regimes would be lost.")
        config.SPLITS_CSV.unlink()
        print(f"Removed old {config.SPLITS_CSV.name} (--new-split)")

    print("[1/3] Raw data")
    print(f"  {download_dataset()} (Kaggle version {config.KAGGLE_VERSION}, SHA-256 verified)")

    print("[2/3] Cleaned data")
    missing = not (config.CLEAN_CSV.exists() and config.CLEAN_WITH_MARKERS_CSV.exists())
    if args.rebuild or args.update_fingerprint or missing:
        build_clean_dataset(update_fingerprint=args.update_fingerprint)  # checks fingerprints, writes the log
    else:
        for variant, path in (("clean", config.CLEAN_CSV), ("clean_with_markers", config.CLEAN_WITH_MARKERS_CSV)):
            verify_clean_file(path, variant)
            print(f"  {path.name} matches the reference fingerprint")
    labels = load_labels()
    print(f"  {len(labels):,} articles, {labels.is_fake.mean():.1%} fake")

    print("[3/3] Splits")
    has_random = config.SPLITS_CSV.exists() and "split_random" in pd.read_csv(config.SPLITS_CSV, nrows=0).columns
    if not has_random:
        save_split_column("random", make_random_split(labels))
        print(f"  Created split_random in {config.SPLITS_CSV}")
    splits = load_splits()
    check_splits_match(labels["doc_id"], splits)
    y = labels.set_index("doc_id")["is_fake"]
    for col in [c for c in splits.columns if c.startswith("split_")]:
        print(f"  {col}:")
        print("    " + check_split(splits.set_index("doc_id")[col], labels=y).to_string().replace("\n", "\n    "))

    peak = peak_memory_gb()
    if peak is not None:
        print(f"\nPeak memory: {peak:.2f} GB")
    print("All checks passed: your data is identical to the group's reference.")


if __name__ == "__main__":
    main()
