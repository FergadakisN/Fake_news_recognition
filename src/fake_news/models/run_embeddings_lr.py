"""Train the main model (frozen sentence embeddings + logistic regression) and save its predictions.

    python3 scripts/run_embeddings_lr.py --regime grouped     # ~1 min on a laptop
    python3 scripts/run_embeddings_lr.py --regime random
    python3 scripts/run_embeddings_lr.py --regime all         # both

Needs the embeddings in data/processed/embeddings/ (committed to git; else run scripts/compute_embeddings.py).
Writes data/processed/predictions/embeddings_lr__<regime>.csv (doc_id, split, prob_fake; val + test, test never
scored) and reports/embeddings_lr/<regime>.json (validation scores for every C, chosen C).
"""

import argparse

from fake_news.features.embeddings import run_classifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["grouped", "random", "all"], default="grouped")
    args = ap.parse_args()
    for regime in (["random", "grouped"] if args.regime == "all" else [args.regime]):
        print(f"\n=== {regime} ===")
        run_classifier(regime)


if __name__ == "__main__":
    main()