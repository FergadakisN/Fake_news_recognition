import math

import pandas as pd
import pytest

from fake_news import config
from fake_news.evaluation.metrics import evaluate
from fake_news.evaluation.predictions import evaluate_predictions, load_predictions, save_predictions


def test_evaluate_known_values():
    # predictions at 0.5: [1, 1, 0, 0, 1] vs truth [1, 0, 0, 1, 1] -> tp=2 fp=1 tn=1 fn=1
    m = evaluate([1, 0, 0, 1, 1], [0.9, 0.6, 0.2, 0.4, 0.8])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_evaluate_single_class_and_bad_input():
    assert math.isnan(evaluate([1, 1], [0.7, 0.2])["roc_auc"])
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.5, 1.2])
    with pytest.raises(ValueError):
        evaluate([0, 2], [0.5, 0.5])


def test_predictions_roundtrip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path)
    preds = pd.DataFrame({"doc_id": [3, 1, 2], "split": ["val", "val", "test"], "prob_fake": [0.9, 0.1, 0.5]})
    path = save_predictions(preds, "model_a", "random")
    assert path == tmp_path / "model_a__random.csv"
    assert load_predictions("model_a", "random")["doc_id"].tolist() == [1, 2, 3]

    labels = pd.DataFrame({"doc_id": [1, 2, 3], "is_fake": [0, 1, 1]})
    assert evaluate_predictions(preds, "val", labels=labels)["accuracy"] == 1.0

    with pytest.raises(ValueError):
        save_predictions(preds.assign(prob_fake=[0.1, 2.0, 0.3]), "model_a", "random")
    with pytest.raises(ValueError):
        save_predictions(preds, "model_a", "not_a_regime")
