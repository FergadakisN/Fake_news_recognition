"""Shared data access. Everything most people need:

    from fake_news.data import load_dataset
    df = load_dataset("random")      # cleaned articles + a `split` column (train/val/test)

The first call downloads and cleans the data automatically (a few minutes); later calls read the cache.
"""

from fake_news.data.load import download_dataset, load_clean, load_dataset, load_labels, load_raw, load_splits

__all__ = ["load_dataset", "load_clean", "load_labels", "load_splits", "load_raw", "download_dataset"]
