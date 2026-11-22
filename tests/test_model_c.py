"""Unit tests for Model C (K-means topic clusters + one logistic regression per cluster), on synthetic texts."""

import numpy as np
import pytest
import scipy.sparse as sp

from fake_news.models import model_c as mc

TOPICS = {
    "politics": "senate vote election ballot campaign congress senator governor policy bill",
    "health": "vaccine virus doctors hospital patients disease treatment clinic medicine nurses",
    "sport": "football match goal league coach players stadium season trophy striker",
}
FAKE_WORDS = "shocking unbelievable exposed secret truth"
REAL_WORDS = "reported according statement officials announced"


def make_corpus(n_per_topic=60, seed=0):
    """Articles from 3 topics; fake ones contain FAKE_WORDS, real ones REAL_WORDS. Returns texts, labels, topics."""
    rng = np.random.default_rng(seed)
    texts, labels, topics = [], [], []
    for t, (name, words) in enumerate(TOPICS.items()):
        vocab = words.split()
        for i in range(n_per_topic):
            fake = i % 2
            style = (FAKE_WORDS if fake else REAL_WORDS).split()
            texts.append(" ".join(rng.choice(vocab, 25).tolist() + rng.choice(style, 4).tolist()))
            labels.append(fake)
            topics.append(t)
    return texts, np.array(labels), np.array(topics)


@pytest.fixture(scope="module")
def fitted():
    texts, y, topics = make_corpus()
    model = mc.ModelC(k=3, c=1.0, n_components=5, min_cluster_size=10).fit(texts, y)
    return model, texts, y, topics


def test_clusters_recover_the_topics(fitted):
    model, texts, _, topics = fitted
    labels = model.kmeans.labels_
    for k in range(3):                                   # every cluster is (almost) one topic
        assert np.bincount(topics[labels == k]).max() / (labels == k).sum() > 0.95


def test_predictions_are_probabilities_and_separate_the_classes(fitted):
    model, texts, y, _ = fitted
    p = model.predict_proba(texts)
    assert p.shape == (len(texts),) and np.all((p >= 0) & (p <= 1))
    assert p[y == 1].min() > 0.5 > p[y == 0].max()      # every fake ranked above every real article


def test_assign_uses_the_nearest_centroid(fitted):
    model, texts, _, _ = fitted
    assert np.array_equal(model.assign(texts), model.kmeans.labels_)


def test_vocabulary_is_learned_from_training_texts_only(fitted):
    model, _, _, _ = fitted
    model.predict_proba(["zebra " * 30])                 # an unseen word must not enter the vocabulary
    assert "zebra" not in model.features.tfidf.vocabulary_


def test_same_seed_gives_same_model():
    texts, y, _ = make_corpus()
    a = mc.ModelC(k=3, c=1.0, n_components=5, min_cluster_size=10).fit(texts, y).predict_proba(texts)
    b = mc.ModelC(k=3, c=1.0, n_components=5, min_cluster_size=10).fit(texts, y).predict_proba(texts)
    assert np.allclose(a, b)


def test_fallback_for_small_and_one_class_clusters():
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.random((300, 6)))
    y = np.array([0, 1] * 150)
    clusters = np.zeros(300, dtype=int)
    clusters[200:250] = 1                                # small cluster (50 < 100)
    clusters[250:] = 2
    y[250:] = 1                                          # cluster 2 has only one class
    glob = mc.fit_global_lr(X, y, 1.0)
    clf = mc.fit_cluster_classifiers(X, y, clusters, 1.0, glob, k=3, min_cluster_size=100)
    assert clf.fallback.tolist() == [False, True, True]
    assert clf.models[1] is glob and clf.models[2] is glob and clf.models[0] is not glob
    assert clf.n_train.tolist() == [200, 50, 50]
    assert clf.fake_share[2] == 1.0
    p = clf.predict_proba(X, clusters)
    assert np.allclose(p[250:], glob.predict_proba(X[250:])[:, 1])   # fallback rows scored by the global LR


def test_topic_only_uses_the_cluster_fake_share():
    share = np.array([0.1, 0.9, np.nan])
    p = mc.topic_only_proba(share, np.array([0, 1, 2, 1]), default=0.44)
    assert p.tolist() == [0.1, 0.9, 0.44, 0.9]
