"""Shared prediction files: one CSV per model and split regime, columns ``doc_id, split, prob_fake``.

Files go to ``data/processed/predictions/<model>__<regime>.csv``. Because every model writes
the same format, blending (Model D) only needs to join files on ``doc_id``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fake_news import config
from fake_news.evaluation.metrics import evaluate

PRED_COLUMNS = ["doc_id", "split", "prob_fake"]


def prediction_path(model: str, regime: str) -> Path:
    return config.PREDICTIONS_DIR / f"{model}__{regime}.csv"


def save_predictions(preds: pd.DataFrame, model: str, regime: str) -> Path:
    """Validate and write predictions (``doc_id, split, prob_fake``). Returns the file path."""
    missing = set(PRED_COLUMNS) - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    preds = preds[PRED_COLUMNS]
    if preds["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_ids in predictions.")
    if not preds["split"].isin(config.SPLIT_NAMES).all():
        raise ValueError(f"split must be one of {config.SPLIT_NAMES}")
    if preds["prob_fake"].isna().any() or not preds["prob_fake"].between(0, 1).all():
        raise ValueError("prob_fake must be probabilities in [0, 1] with no NaN.")
    if regime not in config.REGIMES:
        raise ValueError(f"regime must be one of {config.REGIMES}")

    path = prediction_path(model, regime)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.sort_values("doc_id").to_csv(path, index=False)
    return path


def load_predictions(model: str, regime: str) -> pd.DataFrame:
    return pd.read_csv(prediction_path(model, regime))


def evaluate_predictions(preds: pd.DataFrame, split: str = "val", labels: pd.DataFrame | None = None) -> dict:
    """``evaluate()`` on the rows of one partition, joining true labels on ``doc_id``.

    ``labels`` defaults to ``load_labels()``. Evaluate ``split="test"`` only once, at the end.
    """
    if labels is None:
        from fake_news.data.load import load_labels

        labels = load_labels()
    rows = preds[preds["split"] == split].merge(labels, on="doc_id", how="left")
    if rows["is_fake"].isna().any():
        raise ValueError("Some predicted doc_ids are not in the cleaned dataset.")
    return evaluate(rows["is_fake"], rows["prob_fake"])
