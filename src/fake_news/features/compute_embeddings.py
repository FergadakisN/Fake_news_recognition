"""Compute the sentence embeddings of all cleaned articles, then check them.

    python src/fake_news/features/compute_embeddings.py                  # the normal run
    python src/fake_news/features/compute_embeddings.py --sample 1000    # timing test: estimates the full run on this machine
    python src/fake_news/features/compute_embeddings.py --rebuild        # recompute even if the cache is up to date

Interrupted runs resume where they stopped. Output goes to data/processed/embeddings/.
The run must end with "All checks passed".
"""

import argparse
import sys

from fake_news.features import embeddings as emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=0, help="embed only N random articles (timing test)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 128 on cuda, 64 otherwise")
    ap.add_argument("--chunk-size", type=int, default=5000, help="articles per resumable chunk")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    sample = args.sample or None

    meta = emb.build_embeddings(sample, batch_size=args.batch_size, chunk_size=args.chunk_size,
                                device=args.device, rebuild=args.rebuild)
    if sample:
        est = meta["n_clean_total"] / meta["docs_per_second"]
        print(f"Estimate for all {meta['n_clean_total']:,} cleaned articles on this machine "
              f"({meta['device']}): ~{est / 60:.1f} min")

    print("\nChecks:")
    try:
        for line in emb.check_embeddings(sample):
            print(f"  ok  {line}")
    except AssertionError as e:
        print(f"  FAILED  {e}")
        return 1
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())