"""Build ``split_cluster``: whole K-means topic clusters held out in val and test.

    python scripts/build_cluster_split.py              # K = 32, writes split_cluster (refuses to overwrite)
    python scripts/build_cluster_split.py --overwrite  # rebuild an existing split_cluster

Steps (details in ``fake_news.data.cluster_split``):
    1. K-means (K = 32, k-means++, 10 restarts, config.SEED) on the sentence embeddings of all cleaned articles
       (no labels are used, as for the near-duplicate groups)
    2. near-duplicate groups spread over several clusters are moved into their majority cluster
    3. whole clusters are assigned to train / val / test; 500 assignment orders are tried and the one closest to
       70/15/15 with the overall fake share in every partition is kept (at least 3 clusters in val and in test)
    4. checks: no cluster and no near-duplicate group is spread over partitions; split_cluster is added to splits.csv

Outputs:
    data/processed/splits.csv                     new column split_cluster (in git; reviewed by Person 1)
    data/processed/topic_clusters.csv             doc_id, cluster (the clusters behind the split, in git)
    reports/cluster_split/clusters.csv            per cluster: size, fake share, partition, top words
    reports/cluster_split/summary.json            settings, chosen seed, balance, per-partition sizes and fake share
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from fake_news import config
from fake_news.data import load_clean
from fake_news.data.cluster_split import best_cluster_split, merge_near_duplicates
from fake_news.data.split import check_split, save_split_column
from fake_news.features.embeddings import embeddings_for
from fake_news.similarity.near_duplicates import load_groups

K = 32
N_SEEDS = 500
CLUSTERS_CSV = config.PROCESSED_DIR / "topic_clusters.csv"
REPORT_DIR = config.PROJECT_ROOT / "reports" / "cluster_split"


def top_words(texts: pd.Series, clusters: pd.Series, n: int = 10) -> dict:
    """Most characteristic words per cluster: mean TF-IDF in the cluster minus mean TF-IDF overall."""
    tfidf = TfidfVectorizer(max_features=30_000, min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")
    X = tfidf.fit_transform(texts)
    terms = tfidf.get_feature_names_out()
    overall = np.asarray(X.mean(axis=0)).ravel()
    lab = clusters.to_numpy()
    return {k: ", ".join(terms[np.argsort(-(np.asarray(X[lab == k].mean(axis=0)).ravel() - overall))[:n]])
            for k in np.unique(lab)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing split_cluster column")
    args = ap.parse_args()

    t0 = time.time()
    clean = load_clean(columns=["content"])
    labels = clean.set_index("doc_id")["is_fake"]
    E = embeddings_for(clean.doc_id)
    km = KMeans(n_clusters=K, n_init=10, random_state=config.SEED).fit(E)
    raw = pd.Series(km.labels_, index=pd.Index(clean.doc_id.to_numpy(), name="doc_id"), name="cluster")

    dup = load_groups()
    clusters = merge_near_duplicates(raw, dup)
    moved = int((clusters != raw).sum())
    print(f"K-means K = {K} on {E.shape[0]:,} x {E.shape[1]} embeddings; {moved} articles moved so that every "
          f"near-duplicate group sits in one cluster")

    split, seed, score = best_cluster_split(clean, clusters, n_seeds=N_SEEDS)
    summary = check_split(split, labels=labels, groups=clusters)
    check_split(split, groups=dup)              # raises if a near-duplicate group is spread over partitions
    print(f"\nChosen assignment seed {seed} of {N_SEEDS} (balance score {score:.3f}):\n{summary}")

    save_split_column("cluster", split, overwrite=args.overwrite)
    clusters.rename_axis("doc_id").reset_index().to_csv(CLUSTERS_CSV, index=False)

    words = top_words(clean.content, clusters.reindex(clean.doc_id).reset_index(drop=True))
    per = pd.DataFrame({"cluster": clusters.groupby(clusters).size().index,
                        "n": clusters.groupby(clusters).size().to_numpy(),
                        "fake_share": labels.groupby(clusters).mean().round(4).to_numpy(),
                        "partition": split.groupby(clusters).first().to_numpy()})
    per["top_words"] = per.cluster.map(words)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per.sort_values(["partition", "n"], ascending=[True, False]).to_csv(REPORT_DIR / "clusters.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps({
        "k": K, "features": "sentence embeddings (all-MiniLM-L6-v2), all cleaned articles, no labels",
        "kmeans": {"n_init": 10, "random_state": config.SEED},
        "near_duplicate_articles_moved": moved,
        "assignment": {"function": "make_group_split(stratify=False)", "seeds_tried": N_SEEDS, "chosen_seed": seed,
                       "balance_score": round(score, 4), "min_clusters_val_test": 3,
                       "score": "max |fraction - target| + max |fake share - overall fake share|"},
        "partitions": {p: {"n": int(summary.loc[p, "n"]), "fraction": float(summary.loc[p, "fraction"]),
                           "fake_rate": float(summary.loc[p, "fake_rate"]),
                           "clusters": int(per[per.partition == p].shape[0])} for p in config.SPLIT_NAMES},
        "overall_fake_rate": round(float(labels.mean()), 4),
        "seconds": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print("\nHeld-out clusters:")
    print(per[per.partition != "train"].sort_values(["partition", "n"]).to_string(index=False, max_colwidth=80))
    print(f"\nWrote split_cluster to {config.SPLITS_CSV.relative_to(config.PROJECT_ROOT)}, "
          f"{CLUSTERS_CSV.relative_to(config.PROJECT_ROOT)}, reports/cluster_split/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
