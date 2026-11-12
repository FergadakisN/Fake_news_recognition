"""Load the WELFake dataset: raw file, cleaned articles, shared splits.

Typical use::

    from fake_news.data.load import load_dataset
    df = load_dataset("random")          # cleaned articles + a `split` column
    train = df[df.split == "train"]
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fake_news import config
from fake_news.data.integrity import check_fingerprint, check_splits_match, verify_raw_file

_TEXT_COLS = {"title": str, "text": str, "content": str}


def download_dataset(force: bool = False) -> Path:
    """Download the pinned Kaggle version of ``WELFake_Dataset.csv`` into ``data/raw/``.

    Skips the download when the file already exists (unless ``force=True``). Either way the
    file's SHA-256 is checked against ``config.RAW_SHA256``. Public dataset: usually no Kaggle
    login is needed; if Kaggle asks for one, run ``kagglehub.login()`` once.
    Returns the path to the CSV.
    """
    dest = config.RAW_CSV
    if force or not dest.exists():
        import kagglehub  # lazy import: only needed for downloading

        handle = f"{config.KAGGLE_DATASET}/versions/{config.KAGGLE_VERSION}"
        cache_dir = Path(kagglehub.dataset_download(handle))
        matches = list(cache_dir.rglob(dest.name))
        if not matches:
            raise FileNotFoundError(f"{dest.name} not found in downloaded files at {cache_dir}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], dest)

    verify_raw_file(dest)
    return dest


def load_raw(download: bool = True) -> pd.DataFrame:
    """Raw WELFake rows with columns ``doc_id, title, text, label`` (72,134 rows).

    ``doc_id`` is the original row number of the CSV and is the key used in every
    shared file. Missing titles/texts are left as NaN; use :func:`load_clean` for modelling.
    """
    if not config.RAW_CSV.exists():
        if not download:
            raise FileNotFoundError(f"{config.RAW_CSV} not found. Run `python3 scripts/prepare_data.py`.")
        download_dataset()
    df = pd.read_csv(config.RAW_CSV)
    return df.rename(columns={"Unnamed: 0": "doc_id"})[["doc_id", "title", "text", "label"]]


_ALWAYS = ["doc_id", "is_fake"]


def _usecols(columns):
    if columns is None:
        return None
    return list(dict.fromkeys(_ALWAYS + list(columns)))  # keep order, no duplicates


def read_clean_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a cleaned CSV exactly as written (empty titles stay "", not NaN).

    ``columns`` limits what is loaded (``doc_id`` and ``is_fake`` are always included).
    """
    return pd.read_csv(path, keep_default_na=False, dtype=_TEXT_COLS, usecols=_usecols(columns))


def read_clean_csv_chunks(path: Path, columns: list[str] | None = None, chunk_rows: int | None = None):
    """Iterate over a cleaned CSV in chunks of ``config.CLEAN_CHUNK_ROWS`` rows (bounded memory)."""
    return pd.read_csv(path, keep_default_na=False, dtype=_TEXT_COLS, usecols=_usecols(columns),
                       chunksize=chunk_rows or config.CLEAN_CHUNK_ROWS)


def load_clean(remove_markers: bool = True, rebuild: bool = False, verify: bool = True,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Cleaned, deduplicated articles: ``doc_id, title, text, content, n_words, is_fake``.

    - ``content`` (lower-cased title + text) is the column to model on.
    - ``remove_markers=False`` gives the same rows but with source markers such as
      "(Reuters)" left in; only for measuring how much those markers inflate results.
    - ``columns=["content"]`` loads only those columns (plus ``doc_id``, ``is_fake``). The full
      table takes roughly 1-1.5 GB of RAM; ``doc_id, is_fake, content`` about half of that.

    The result is cached in ``data/interim/``; it is built on first use (a few minutes).
    With ``verify=True`` the loaded columns are compared with the committed fingerprint and a
    ``DataMismatchError`` is raised if they differ from everyone else's.
    """
    path = config.CLEAN_CSV if remove_markers else config.CLEAN_WITH_MARKERS_CSV
    if rebuild or not path.exists():
        from fake_news.data.clean import build_clean_dataset

        build_clean_dataset()
    df = read_clean_csv(path, columns)
    if verify:
        check_fingerprint(df, "clean" if remove_markers else "clean_with_markers")
    return df


def load_labels(verify: bool = True) -> pd.DataFrame:
    """Just ``doc_id, is_fake`` of the cleaned dataset (fast, a few MB of memory)."""
    return load_clean(columns=[], verify=verify)


def load_splits() -> pd.DataFrame:
    """The shared split file: ``doc_id`` plus one ``split_<regime>`` column per regime."""
    if not config.SPLITS_CSV.exists():
        raise FileNotFoundError(f"{config.SPLITS_CSV} not found. Run `python3 scripts/prepare_data.py`.")
    return pd.read_csv(config.SPLITS_CSV)


def load_dataset(regime: str = "random", remove_markers: bool = True, verify: bool = True,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Cleaned articles joined with one split regime, as a ``split`` column (train/val/test).

    ``regime`` is one of ``"random"``, ``"grouped"``, ``"cluster"`` (only those already
    present in ``data/processed/splits.csv``). ``columns`` works as in :func:`load_clean`
    (e.g. ``columns=["content"]`` to save memory). Raises ``DataMismatchError`` if the cleaned
    data differs from the reference or does not match ``splits.csv`` exactly.
    """
    splits = load_splits()
    col = f"split_{regime}"
    if col not in splits.columns:
        available = [c.removeprefix("split_") for c in splits.columns if c.startswith("split_")]
        raise KeyError(f"Split regime {regime!r} not available yet. Available: {available}")

    df = load_clean(remove_markers=remove_markers, verify=verify, columns=columns)
    check_splits_match(df["doc_id"], splits)
    return df.merge(splits[["doc_id", col]].rename(columns={col: "split"}), on="doc_id", how="left")
