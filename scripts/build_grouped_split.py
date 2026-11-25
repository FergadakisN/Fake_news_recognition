"""Rebuild split_grouped in data/processed/splits.csv from the final near-duplicate groups.

    python scripts/build_grouped_split.py

Whole groups of data/processed/near_dup_groups.csv (MinHash + embedding tier) go to one partition
(make_group_split, stratified by majority label, config.SEED). check_split must pass: no group may be spread
over two partitions. Only the split_grouped column is replaced; split_random and any other column stay as
they are. Afterwards rebuild the grouped features: python scripts/build_lsh_features.py --regime grouped
"""

import sys

from fake_news.data import load_clean, load_splits
from fake_news.data.split import check_split, make_group_split, save_split_column
from fake_news.similarity.near_duplicates import load_groups


def main() -> int:
    clean = load_clean(columns=[])
    groups = load_groups()
    if set(groups.index) != set(clean["doc_id"]):
        raise SystemExit("near_dup_groups.csv does not cover exactly the cleaned articles; rebuild the groups first.")

    before = load_splits()
    split = make_group_split(clean, groups)
    print(check_split(split, labels=clean.set_index("doc_id")["is_fake"], groups=groups))  # raises on a leak
    save_split_column("grouped", split, overwrite=True)

    after = load_splits()
    for col in before.columns.drop(["doc_id", "split_grouped"], errors="ignore"):   # nothing else changed
        if not before.set_index("doc_id")[col].sort_index().equals(after.set_index("doc_id")[col].sort_index()):
            raise SystemExit(f"{col} changed unexpectedly")
    moved = (before.set_index("doc_id")["split_grouped"] != after.set_index("doc_id")["split_grouped"]).mean() \
        if "split_grouped" in before else 1.0
    print(f"split_grouped rebuilt; {moved:.1%} of articles changed partition. Other columns unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
