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
    monkeypatch.setattr(config, "CLEANING_REPORT_DIR", tmp_path / "cleaning_log")  # never touch reports/
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
        "raw_rows": 9, "dropped_empty_text": 2, "dropped_non_english": 0, "dropped_conflicting_labels": 2,
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


# ---- what a teammate experiences: everything happens on the first load
def test_first_load_downloads_and_builds(setup, monkeypatch, tmp_path):
    import kagglehub

    from fake_news.data import load_clean

    cache = tmp_path / "kaggle_cache"
    cache.mkdir()
    config.RAW_CSV.rename(cache / "WELFake_Dataset.csv")  # no raw file locally yet
    handles = []

    def fake_download(handle):
        handles.append(handle)
        return str(cache)

    monkeypatch.setattr(config, "RAW_CSV", tmp_path / "raw" / "WELFake_Dataset.csv")
    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)
    df = load_clean()
    assert handles == [f"{config.KAGGLE_DATASET}/versions/{config.KAGGLE_VERSION}"]
    assert config.RAW_CSV.exists() and df["doc_id"].tolist() == [0, 3, 7]


def test_outdated_cache_is_rebuilt_automatically(setup, capsys):
    from fake_news.data import load_clean

    build_clean_dataset()
    good = config.CLEAN_CSV.read_text()
    config.CLEAN_CSV.write_text(good.replace("Same story here", "Same story HERE"))  # stale/modified cache
    df = load_clean()
    assert "rebuilding" in capsys.readouterr().out
    assert config.CLEAN_CSV.read_text() == good and len(df) == 3


def test_fresh_build_that_differs_still_raises(setup):
    from fake_news.data import load_clean

    build_clean_dataset()
    fp = load_expected()
    fp["clean"]["content_sha256"] = "0" * 64  # e.g. an incompatible library version
    config.FINGERPRINT_JSON.write_text(pd.Series(fp).to_json())
    with pytest.raises(DataMismatchError):
        load_clean()


def test_rule_subset_reproduces_an_earlier_stage(setup, monkeypatch, tmp_path):
    from fake_news.data.clean import build_rule_subset

    spanish = ("Las imágenes libres de derechos más destacadas de la semana, el black friday y la lotería de navidad "
               "que todos los años se celebra en el país con una gran fiesta para las familias")
    raw = pd.DataFrame([(0, "A", "Ben Sasse (@BenSasse) September 28, 2017Spencer responded.", 1),
                        (1, "B", spanish, 1)], columns=["Unnamed: 0", "title", "text", "label"])
    raw_csv = tmp_path / "raw2.csv"
    raw.to_csv(raw_csv, index=False)
    monkeypatch.setattr(config, "RAW_CSV", raw_csv)
    monkeypatch.setattr(config, "RAW_SHA256", sha256_file(raw_csv))

    out = tmp_path / "subset" / "round1.csv"
    report = build_rule_subset(out, groups=("R1",), language_filter=False)
    subset = read_clean_csv(out)
    assert report["clean_rows"] == 2 and report["dropped_non_english"] == 0  # no language filter in round 1
    assert "september 28, 2017spencer" in subset.loc[0, "content"]            # the round-2 tweet rule did not run

    full = clean_dataset(raw.rename(columns={"Unnamed: 0": "doc_id"}))
    assert full["doc_id"].tolist() == [0] and "september" not in full.loc[0, "content"]
    assert not list(config.CLEANING_REPORT_DIR.glob("*")) if config.CLEANING_REPORT_DIR.exists() else True
