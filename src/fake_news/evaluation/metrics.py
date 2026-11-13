"""The one metrics function everyone uses, so results are comparable across models.

Fake is the positive class (``is_fake = 1``). Accuracy, precision, recall and F1 match
the metrics reported in the WELFake paper; ROC-AUC is added because blending uses probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Score predicted probabilities of *fake* against true labels (1 = fake).

    Returns a dict: ``n, accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp``.
    ``roc_auc`` is NaN when ``y_true`` contains only one class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob have different shapes: {y_true.shape} vs {y_prob.shape}")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must contain only 0 (real) and 1 (fake).")
    if np.isnan(y_prob).any() or y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob must be probabilities in [0, 1] with no NaN.")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
