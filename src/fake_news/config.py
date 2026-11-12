"""Shared paths and conventions. Import these instead of hard-coding paths or seeds.

Label convention: ``is_fake = 1`` means fake, ``0`` means real. The raw WELFake
``label`` column already follows this (the Kaggle page says the opposite, but
"(Reuters)" datelines appear in 21,266 label-0 articles and only 16 label-1 ones).
"""

from pathlib import Path

# src/fake_news/config.py -> parents[2] is the repository root (requires `pip install -e .`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

KAGGLE_DATASET = "saurabhshahane/fake-news-classification"
KAGGLE_VERSION = 77  # pinned: a newer upload on Kaggle must not silently change our data
RAW_CSV = RAW_DIR / "WELFake_Dataset.csv"
RAW_SHA256 = "665331424230fc452e9482c3547a6a199a2c29745ade8d236950d1d105223773"
CLEAN_CSV = INTERIM_DIR / "welfake_clean.csv"
CLEAN_WITH_MARKERS_CSV = INTERIM_DIR / "welfake_clean_with_markers.csv"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
# Expected row count + hashes of the cleaned data (committed to git, checked on every load)
FINGERPRINT_JSON = PROCESSED_DIR / "clean_fingerprint.json"
# Rows per chunk when building/verifying the cleaned data. Peak memory of the build scales with
# this, not with the dataset size (5,000 rows ~ a few hundred MB).
CLEAN_CHUNK_ROWS = 5_000

SEED = 42
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
REGIMES = ("random", "grouped", "cluster")
