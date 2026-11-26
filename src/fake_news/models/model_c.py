"""Model C: K-means topic clusters + one logistic regression per cluster.

The idea: fake news may be written differently in different topics, so a classifier that specialises in
one topic could pick up signals a single global classifier misses. Model C:

1. **Features (train only).** TF-IDF of ``content`` (sparse, ~50k words) for the classifiers.
2. **Clusters (train only).** K-means (Lloyd's algorithm, k-means++ initialisation, ``n_init`` restarts) on one of
   two vector spaces, chosen with ``cluster_on``:

   - ``"embeddings"`` (default): the 384-d sentence embeddings (all-MiniLM-L6-v2, ``embeddings_for``). They capture
     meaning, so articles about the same story cluster together even with different words.
   - ``"svd"``: TruncatedSVD of the TF-IDF matrix to ``n_components`` dense dimensions (LSA "topics").

   Both are L2-normalised; on unit vectors Euclidean K-means groups articles by cosine similarity.
3. **One classifier per cluster.** A logistic regression on the full TF-IDF features of that cluster's
   training articles. The cluster vectors only decide the cluster; the words decide fake or real.
   **Fallback:** a cluster with fewer than ``min_cluster_size`` training articles, or with only one class,
   uses a global logistic regression trained on all training articles instead.
4. **Predict.** A new article goes to its nearest centroid and is scored by that cluster's classifier.

Everything is fitted on train only; val and test are only transformed.

The pieces are separate functions so that a tuning script can fit TF-IDF/SVD once, K-means once per K and
the classifiers once per C. ``ModelC`` wraps them for normal use::

    model = ModelC(k=8, c=100).fit(train.content, train.is_fake, train.doc_id)
    prob_fake = model.predict_proba(val.content, val.doc_id)
    cluster = model.assign(val.content, val.doc_id)

``doc_ids`` are needed to look up the embeddings; with ``cluster_on="svd"`` they can be left out.
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
CLUSTER_SPACES = ("embeddings", "svd")
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


def member_top_terms(X, clusters: np.ndarray, terms, n: int = 10) -> list[list[str]]:
    """Most characteristic words per cluster: mean TF-IDF of its members minus the mean over all rows.

    Works for any cluster space (embedding centroids cannot be mapped back to words).
    """
    overall = np.asarray(X.mean(axis=0)).ravel()
    return [list(np.asarray(terms)[np.argsort(-(np.asarray(X[clusters == k].mean(axis=0)).ravel() - overall))[:n]])
            for k in range(int(clusters.max()) + 1)]


def embedding_vectors(doc_ids) -> np.ndarray:
    """L2-normalised sentence embeddings of ``doc_ids`` (rows in the same order)."""
    from fake_news.features.embeddings import embeddings_for

    return embeddings_for(doc_ids)


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

    def __init__(self, k: int = DEFAULT_K, c: float = DEFAULT_C, cluster_on: str = "embeddings",
                 n_components: int = N_COMPONENTS, min_cluster_size: int = MIN_CLUSTER_SIZE, seed: int = config.SEED):
        if cluster_on not in CLUSTER_SPACES:
            raise ValueError(f"cluster_on must be one of {CLUSTER_SPACES}, got {cluster_on!r}")
        self.k, self.c, self.cluster_on, self.min_cluster_size, self.seed = k, c, cluster_on, min_cluster_size, seed
        self.features = TopicFeatures(n_components=n_components, seed=seed)

    def _vectors(self, Z_svd: np.ndarray, doc_ids) -> np.ndarray:
        if self.cluster_on == "svd":
            return Z_svd
        if doc_ids is None:
            raise ValueError("cluster_on='embeddings' needs the doc_ids of the texts")
        return embedding_vectors(doc_ids)

    def fit(self, texts, y, doc_ids=None) -> "ModelC":
        X, Z_svd = self.features.fit_transform(texts)
        self.kmeans = fit_kmeans(self._vectors(Z_svd, doc_ids), self.k, seed=self.seed)
        self.global_lr = fit_global_lr(X, y, self.c)
        self.classifiers = fit_cluster_classifiers(X, y, self.kmeans.labels_, self.c, self.global_lr,
                                                   k=self.k, min_cluster_size=self.min_cluster_size)
        self._terms = self.features.tfidf.get_feature_names_out()
        self._top = member_top_terms(X, self.kmeans.labels_, self._terms)
        return self

    def assign(self, texts, doc_ids=None) -> np.ndarray:
        """Cluster id (nearest centroid) of each text."""
        _, Z_svd = self.features.transform(texts)
        return self.kmeans.predict(self._vectors(Z_svd, doc_ids))

    def predict_proba(self, texts, doc_ids=None) -> np.ndarray:
        """P(fake) of each text."""
        X, Z_svd = self.features.transform(texts)
        return self.classifiers.predict_proba(X, self.kmeans.predict(self._vectors(Z_svd, doc_ids)))

    def top_terms(self, n: int = 10) -> list[list[str]]:
        """Most characteristic words of each cluster's training articles."""
        return [t[:n] for t in self._top]
