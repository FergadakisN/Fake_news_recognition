"""Unit tests for the integrity checks (synthetic data; always run)."""

import warnings

import pandas as pd
import pytest

from fake_news import config
from fake_news.data.integrity import (
    DataMismatchError,
    check_fingerprint,
    check_splits_match,
    fingerprint,
    save_fingerprint,
    sha256_file,
    verify_raw_file,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "doc_id": [5, 2, 9],
        "title": ["a", "", "c"],
        "text": ["text a", "text b", "text c"],
        "content": ["a. text a", "text b", "c. text c"],
        "n_words": [3, 2, 3],
        "is_fake": [0, 1, 1],
    })


@pytest.fixture(autouse=True)
def tmp_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FINGERPRINT_JSON", tmp_path / "fp.json")


def test_fingerprint_ignores_row_order(df):
    assert fingerprint(df) == fingerprint(df.sample(frac=1, random_state=1))


def test_fingerprint_detects_changes(df):
    base = fingerprint(df)
    assert fingerprint(df.assign(is_fake=[1, 1, 1]))["doc_ids_labels_sha256"] != base["doc_ids_labels_sha256"]
    assert fingerprint(df.assign(title=["a", " ", "c"]))["title_text_sha256"] != base["title_text_sha256"]
    assert fingerprint(df.iloc[:2])["n_rows"] == 2


def test_streamed_file_fingerprint_equals_in_memory(df, tmp_path):
    from fake_news.data.integrity import fingerprint_file

    path = tmp_path / "clean.csv"
    df.sort_values("doc_id").to_csv(path, index=False)
    assert fingerprint_file(path) == fingerprint(df)


def test_partial_columns_are_checked_on_what_is_loaded(df):
    save_fingerprint(df, "clean")
    assert check_fingerprint(df[["doc_id", "is_fake", "content"]], "clean")
    assert check_fingerprint(df[["doc_id", "is_fake"]], "clean")
    with pytest.raises(DataMismatchError, match="content_sha256"):
        check_fingerprint(df[["doc_id", "is_fake"]].assign(content="changed"), "clean")


def test_save_then_check(df):
    save_fingerprint(df, "clean")
    assert check_fingerprint(df, "clean")
    with pytest.raises(DataMismatchError, match="content_sha256"):
        check_fingerprint(df.assign(content=["x", "y", "z"]), "clean")


def test_missing_reference_warns(df):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert check_fingerprint(df, "clean_with_markers") is False
    assert "No reference fingerprint" in str(w[0].message)


def test_check_splits_match(df):
    check_splits_match(df["doc_id"], pd.DataFrame({"doc_id": [2, 5, 9]}))
    with pytest.raises(DataMismatchError, match="1 split rows are not in the cleaned data"):
        check_splits_match(df["doc_id"], pd.DataFrame({"doc_id": [2, 5, 9, 11]}))
    with pytest.raises(DataMismatchError, match="1 cleaned articles have no split"):
        check_splits_match(df["doc_id"], pd.DataFrame({"doc_id": [2, 5]}))


def test_verify_raw_file(tmp_path, monkeypatch):
    f = tmp_path / "raw.csv"
    f.write_text("a,b\n1,2\n")
    monkeypatch.setattr(config, "RAW_SHA256", sha256_file(f))
    verify_raw_file(f)
    f.write_text("a,b\n1,3\n")
    with pytest.raises(DataMismatchError, match="SHA-256"):
        verify_raw_file(f)
