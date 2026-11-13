import numpy as np
import pandas as pd
import pytest

from fake_news import config
from fake_news.data.split import check_split, make_group_split, make_random_split, save_split_column


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame({"doc_id": np.arange(100, 100 + n), "is_fake": (rng.random(n) < 0.45).astype(int)})


def test_random_split_fractions_stratified_and_deterministic(df):
    split = make_random_split(df)
    assert split.index.tolist() == df["doc_id"].tolist()
    summary = check_split(split, labels=df.set_index("doc_id")["is_fake"])
    assert np.allclose(summary["fraction"], config.SPLIT_FRACTIONS, atol=0.01)
    assert summary["fake_rate"].max() - summary["fake_rate"].min() < 0.01
    assert split.equals(make_random_split(df))
    assert not split.equals(make_random_split(df, seed=1))


def test_group_split_keeps_groups_together(df):
    groups = pd.Series(df["doc_id"].to_numpy() // 7, index=df["doc_id"])  # groups of ~7 articles
    split = make_group_split(df, groups)
    summary = check_split(split, labels=df.set_index("doc_id")["is_fake"], groups=groups)
    assert np.allclose(summary["fraction"], config.SPLIT_FRACTIONS, atol=0.02)
    assert split.equals(make_group_split(df, groups))


def test_group_split_treats_missing_as_singletons(df):
    groups = pd.Series({100: "g", 101: "g", 102: "g"})  # everything else ungrouped
    split = make_group_split(df, groups)
    assert split.notna().all()
    assert split[[100, 101, 102]].nunique() == 1


def test_check_split_detects_leaking_group():
    split = pd.Series(["train", "test"], index=pd.Index([1, 2], name="doc_id"))
    with pytest.raises(ValueError, match="spread over"):
        check_split(split, groups=pd.Series({1: "g", 2: "g"}))
    with pytest.raises(ValueError, match="Unknown partition"):
        check_split(pd.Series(["training"], index=[1]))


def test_save_split_column_refuses_overwrite(df, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLITS_CSV", tmp_path / "splits.csv")
    split = make_random_split(df)
    save_split_column("random", split)
    save_split_column("grouped", split)
    assert list(pd.read_csv(config.SPLITS_CSV).columns) == ["doc_id", "split_random", "split_grouped"]
    with pytest.raises(FileExistsError):
        save_split_column("random", split)
    save_split_column("random", make_random_split(df, seed=1), overwrite=True)
    with pytest.raises(ValueError, match="doc_ids differ"):
        save_split_column("cluster", split.iloc[:-1])
