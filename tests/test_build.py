"""The chunked (streaming) build must give exactly the same result as in-memory cleaning."""

import pandas as pd
import pytest

from fake_news import config
from fake_news.data.clean import build_clean_dataset, clean_dataset
from fake_news.data.integrity import DataMismatchError, load_expected, sha256_file
from fake_news.data.load import read_clean_csv

RAW = pd.DataFrame(
    [
        (0, "A", "Same story here.", 0),
        (1, "B", None, 1),                                # empty -> dropped
        (2, "A", "Same story here.", 0),                  # duplicate of 0, in a later chunk
        (3, "Senate vote (VIDEO)", "WASHINGTON (Reuters) - The Senate voted.", 0),
        (4, "D", "Disputed text.", 0),
        (5, "Senate vote", "The Senate voted.", 0),       # duplicate of 3 once markers are removed
        (6, "D", "Disputed text.", 1),                    # conflicts with 4 (other label) -> both dropped
        (7, "Other", "Different text. Featured image via Getty Images", 1),
        (8, None, "https://only-a-link.com/x", 1),        # empty after marker removal -> dropped
    ],
    columns=["Unnamed: 0", "title", "text", "label"],
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw_csv = tmp_path / "raw.csv"
    RAW.to_csv(raw_csv, index=False)
    monkeypatch.setattr(config, "RAW_CSV", raw_csv)
    monkeypatch.setattr(config, "RAW_SHA256", sha256_file(raw_csv))
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path)
    monkeypatch.setattr(config, "CLEAN_CSV", tmp_path / "clean.csv")
    monkeypatch.setattr(config, "CLEAN_WITH_MARKERS_CSV", tmp_path / "clean_with_markers.csv")
    monkeypatch.setattr(config, "FINGERPRINT_JSON", tmp_path / "fp.json")
    monkeypatch.setattr(config, "CLEAN_CHUNK_ROWS", 2)  # duplicates/conflicts span chunk boundaries
    return tmp_path


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 100])
def test_streaming_build_equals_in_memory(setup, monkeypatch, chunk_rows):
    monkeypatch.setattr(config, "CLEAN_CHUNK_ROWS", chunk_rows)
    report = build_clean_dataset()
    raw = RAW.rename(columns={"Unnamed: 0": "doc_id"})
    for remove_markers, path in ((True, config.CLEAN_CSV), (False, config.CLEAN_WITH_MARKERS_CSV)):
        expected = clean_dataset(raw, remove_markers=remove_markers)
        pd.testing.assert_frame_equal(read_clean_csv(path), expected, check_dtype=False)
    assert report == expected.attrs["report"] == {
        "raw_rows": 9, "dropped_empty_text": 2, "dropped_conflicting_labels": 2,
        "dropped_duplicates": 2, "clean_rows": 3,
    }
    assert read_clean_csv(config.CLEAN_CSV)["doc_id"].tolist() == [0, 3, 7]


def test_first_build_stores_reference_and_rebuild_matches(setup, capsys):
    build_clean_dataset()
    assert set(load_expected()) == {"clean", "clean_with_markers"}
    assert "stored as reference" in capsys.readouterr().out
    build_clean_dataset()
    assert capsys.readouterr().out.count("matches the reference fingerprint") == 2
    assert not list(setup.glob("*.tmp"))


def test_mismatch_keeps_old_file_and_removes_tmp(setup):
    build_clean_dataset()
    before = config.CLEAN_CSV.read_bytes()
    fp = load_expected()
    fp["clean"]["content_sha256"] = "0" * 64  # pretend the reference differs
    config.FINGERPRINT_JSON.write_text(pd.Series(fp).to_json())
    with pytest.raises(DataMismatchError, match="content_sha256"):
        build_clean_dataset()
    assert config.CLEAN_CSV.read_bytes() == before  # never replaced by mismatching data
    assert not list(setup.glob("*.tmp"))


def test_wrong_raw_file_is_refused(setup, monkeypatch):
    monkeypatch.setattr(config, "RAW_SHA256", "0" * 64)
    with pytest.raises(DataMismatchError, match="SHA-256"):
        build_clean_dataset()
    assert not config.CLEAN_CSV.exists()
