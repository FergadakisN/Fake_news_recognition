"""Model C: K-means topic clusters + one logistic regression per cluster.

The idea: fake news may be written differently in different topics, so a classifier that specialises in
one topic could pick up signals a single global classifier misses. Model C:

1. **Topic features (train only).** TF-IDF of ``content`` (sparse, ~50k words), then TruncatedSVD to
   ``n_components`` dense dimensions (LSA "topics"), then L2 normalisation. On unit vectors Euclidean
   K-means groups articles by cosine similarity.
2. **Clusters (train only).** K-means (Lloyd's algorithm, k-means++ initialisation, ``n_init`` restarts)
   on the SVD vectors of the training articles.
3. **One classifier per cluster.** A logistic regression on the full TF-IDF features of that cluster's
   training articles. The SVD vectors only decide the cluster; the words decide fake or real.
   **Fallback:** a cluster with fewer than ``min_cluster_size`` training articles, or with only one class,
   uses a global logistic regression trained on all training articles instead.
4. **Predict.** A new article goes to its nearest centroid and is scored by that cluster's classifier.

Everything is fitted on train only; val and test are only transformed.

The pieces are separate functions so that a tuning script can fit TF-IDF/SVD once, K-means once per K and
the classifiers once per C. ``ModelC`` wraps them for normal use::

    model = ModelC(k=8, c=30).fit(train.content, train.is_fake)
    prob_fake = model.predict_proba(val.content)
    cluster = model.assign(val.content)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from fake_news import config

# Defaults chosen on the grouped split's val set (see scripts/run_model_c.py and reports/model_c/).
DEFAULT_K = 8
DEFAULT_C = 100.0
N_COMPONENTS = 200
MIN_CLUSTER_SIZE = 200
TFIDF_PARAMS = dict(max_features=50_000, min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english",
                    dtype=np.float32)


# ---------------------------------------------------------------- 1. features
@dataclass
class TopicFeatures:
    """TF-IDF (for the classifiers) and normalised SVD vectors (for the clusters), fitted on train."""

    n_components: int = N_COMPONENTS
    seed: int = config.SEED
    tfidf: TfidfVectorizer = field(init=False, repr=False)
    svd: TruncatedSVD = field(init=False, repr=False)

    def fit_transform(self, texts) -> tuple[sp.csr_matrix, np.ndarray]:
        """Fit on the training texts. Returns (TF-IDF matrix, normalised SVD vectors)."""
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        Z = normalize(self.svd.fit_transform(X))
        return X, Z

    def transform(self, texts) -> tuple[sp.csr_matrix, np.ndarray]:
        X = self.tfidf.transform(texts)
        return X, normalize(self.svd.transform(X))

    @property
    def explained_variance(self) -> float:
        return float(self.svd.explained_variance_ratio_.sum())

    def top_terms(self, centers: np.ndarray, n: int = 10) -> list[list[str]]:
        """The ``n`` words with the largest weight in each centroid, mapped back from SVD space to words."""
        terms = self.tfidf.get_feature_names_out()
        weights = self.svd.inverse_transform(centers)
        return [list(terms[np.argsort(-w)[:n]]) for w in weights]


# ---------------------------------------------------------------- 2. clusters
def fit_kmeans(Z: np.ndarray, k: int, seed: int = config.SEED, n_init: int = 10) -> KMeans:
    """K-means (k-means++ initialisation, ``n_init`` restarts, best inertia kept) on training vectors."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)


# ---------------------------------------------------------------- 3. classifiers
def _logreg(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000)


def fit_global_lr(X, y, c: float) -> LogisticRegression:
    return _logreg(c).fit(X, np.asarray(y))


@dataclass
class ClusterClassifiers:
    """One logistic regression per cluster, with the global one as fallback."""

    models: list            # models[k] = classifier used for cluster k
    fallback: np.ndarray    # fallback[k] = True if cluster k uses the global classifier
    n_train: np.ndarray     # training articles per cluster
    fake_share: np.ndarray  # share of fake training articles per cluster

    def predict_proba(self, X, clusters: np.ndarray) -> np.ndarray:
        """P(fake) for each row of ``X``, using the classifier of that row's cluster."""
        p = np.empty(X.shape[0], dtype=float)
        for k, model in enumerate(self.models):
            rows = clusters == k
            if rows.any():
                p[rows] = model.predict_proba(X[rows])[:, 1]
        return p


def fit_cluster_classifiers(X, y, clusters: np.ndarray, c: float, global_lr: LogisticRegression,
                            k: int | None = None, min_cluster_size: int = MIN_CLUSTER_SIZE) -> ClusterClassifiers:
    """Fit one logistic regression per cluster on that cluster's training rows (fallback: ``global_lr``)."""
    y = np.asarray(y)
    k = int(clusters.max()) + 1 if k is None else k
    models, fallback, n_train, share = [], [], [], []
    for j in range(k):
        rows = clusters == j
        yj = y[rows]
        use_global = rows.sum() < min_cluster_size or len(np.unique(yj)) < 2
        models.append(global_lr if use_global else _logreg(c).fit(X[rows], yj))
        fallback.append(use_global)
        n_train.append(int(rows.sum()))
        share.append(float(yj.mean()) if len(yj) else float("nan"))
    return ClusterClassifiers(models, np.array(fallback), np.array(n_train), np.array(share))


def topic_only_proba(fake_share: np.ndarray, clusters: np.ndarray, default: float) -> np.ndarray:
    """Reference predictor that reads nothing but the cluster: P(fake) = fake share of that cluster in train."""
    share = np.where(np.isnan(fake_share), default, fake_share)
    return share[clusters]


# ---------------------------------------------------------------- the model
class ModelC:
    """Topic clusters (K-means) + one logistic regression per cluster. Fit on train only."""

    def __init__(self, k: int = DEFAULT_K, c: float = DEFAULT_C, n_components: int = N_COMPONENTS,
                 min_cluster_size: int = MIN_CLUSTER_SIZE, seed: int = config.SEED):
        self.k, self.c, self.min_cluster_size, self.seed = k, c, min_cluster_size, seed
        self.features = TopicFeatures(n_components=n_components, seed=seed)

    def fit(self, texts, y) -> "ModelC":
        X, Z = self.features.fit_transform(texts)
        self.kmeans = fit_kmeans(Z, self.k, seed=self.seed)
        self.global_lr = fit_global_lr(X, y, self.c)
        self.classifiers = fit_cluster_classifiers(X, y, self.kmeans.labels_, self.c, self.global_lr,
                                                   k=self.k, min_cluster_size=self.min_cluster_size)
        return self

    def assign(self, texts) -> np.ndarray:
        """Cluster id (nearest centroid) of each text."""
        _, Z = self.features.transform(texts)
        return self.kmeans.predict(Z)

    def predict_proba(self, texts) -> np.ndarray:
        """P(fake) of each text."""
        X, Z = self.features.transform(texts)
        return self.classifiers.predict_proba(X, self.kmeans.predict(Z))

    def top_terms(self, n: int = 10) -> list[list[str]]:
        return self.features.top_terms(self.kmeans.cluster_centers_, n)
