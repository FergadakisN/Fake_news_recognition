"""Shared evaluation: one metrics function and one prediction-file format for every model.

    from fake_news.evaluation import evaluate, save_predictions
"""

from fake_news.evaluation.metrics import evaluate
from fake_news.evaluation.predictions import evaluate_predictions, load_predictions, save_predictions

__all__ = ["evaluate", "save_predictions", "load_predictions", "evaluate_predictions"]
