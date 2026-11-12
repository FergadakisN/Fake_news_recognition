"""Checks on the REAL data. Skipped automatically when the files have not been generated yet.

Run `python3 scripts/prepare_data.py` first; then `python3 -m pytest` verifies that your local
data is identical to the group's reference (pinned raw file, committed fingerprint, splits.csv).
Files are streamed in chunks, so these tests need little memory.
"""

import pandas as pd
import pytest

from fake_news import config
from fake_news.data.integrity import (
    DataMismatchError,
    check_fingerprint,
    check_splits_match,
    load_expected,
    verify_clean_file,
    verify_raw_file,
)
from fake_news.data.load import read_clean_csv, read_clean_csv_chunks
from fake_news.data.split import check_split


def _require(*paths):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"not generated yet: {missing} (run scripts/prepare_data.py)")


def test_raw_file_is_pinned_version():
    _require(config.RAW_CSV)
    verify_raw_file()


@pytest.mark.parametrize("variant, attr", [("clean", "CLEAN_CSV"), ("clean_with_markers", "CLEAN_WITH_MARKERS_CSV")])
def test_clean_files_match_reference_fingerprint(variant, attr):
    path = getattr(config, attr)
    _require(path, config.FINGERPRINT_JSON)
    assert verify_clean_file(path, variant)


def test_both_variants_have_identical_rows():
    _require(config.FINGERPRINT_JSON)
    expected = load_expected()
    assert expected["clean"]["doc_ids_labels_sha256"] == expected["clean_with_markers"]["doc_ids_labels_sha256"]


def test_clean_data_basic_properties():
    _require(config.CLEAN_CSV)
    seen, n_rows = set(), 0
    for chunk in read_clean_csv_chunks(config.CLEAN_CSV, columns=["text", "content"]):
        assert chunk["doc_id"].is_monotonic_increasing
        assert set(chunk["is_fake"].unique()) <= {0, 1}
        assert (chunk["text"] != "").all()
        seen.update(chunk["content"].map(hash))
        n_rows += len(chunk)
    assert len(seen) == n_rows, "duplicate contents in the cleaned data"


def test_splits_cover_exactly_the_clean_data():
    _require(config.CLEAN_CSV, config.SPLITS_CSV)
    labels = read_clean_csv(config.CLEAN_CSV, columns=[])
    splits = pd.read_csv(config.SPLITS_CSV)
    check_splits_match(labels["doc_id"], splits)
    y = labels.set_index("doc_id")["is_fake"]
    for col in [c for c in splits.columns if c.startswith("split_")]:
        summary = check_split(splits.set_index("doc_id")[col], labels=y)
        assert (summary["n"] > 0).all(), f"{col} has an empty partition"


def test_a_single_changed_character_is_detected():
    _require(config.CLEAN_CSV, config.FINGERPRINT_JSON)
    df = read_clean_csv(config.CLEAN_CSV, columns=["content"])  # ~0.5 GB: only the content column
    assert check_fingerprint(df, "clean")
    df.loc[0, "content"] = df.loc[0, "content"] + "x"
    with pytest.raises(DataMismatchError, match="content_sha256"):
        check_fingerprint(df, "clean")
