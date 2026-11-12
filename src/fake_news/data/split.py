"""Train/val/test splits shared by the whole group.

All splits live in one file, ``data/processed/splits.csv``, with columns
``doc_id, split_random, split_grouped, split_cluster`` (columns are added as they become
available). The file is committed to git so everyone uses identical splits.

- ``split_random``: stratified random 70/15/15 (the provisional split until ~18 Nov,
  and the "naive" regime of the experiment).
- ``split_grouped``: whole near-duplicate groups (from MinHash/LSH) kept in one partition.
- ``split_cluster``: whole K-means clusters held out.

The last two are built with :func:`make_group_split` and added with :func:`save_split_column`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fake_news import config


def make_random_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = config.SPLIT_FRACTIONS,
    seed: int = config.SEED,
) -> pd.Series:
    """Stratified (by ``is_fake``) random split. Returns a Series of train/val/test indexed by ``doc_id``."""
    _check_fractions(fractions)
    ids, y = df["doc_id"].to_numpy(), df["is_fake"].to_numpy()
    _, val_frac, test_frac = fractions
    train_ids, rest_ids, _, y_rest = train_test_split(
        ids, y, test_size=val_frac + test_frac, stratify=y, random_state=seed)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=test_frac / (val_frac + test_frac), stratify=y_rest, random_state=seed)

    split = pd.Series("train", index=pd.Index(ids, name="doc_id"), name="split")
    split[val_ids] = "val"
    split[test_ids] = "test"
    return split


def make_group_split(
    df: pd.DataFrame,
    groups: pd.Series,
    fractions: tuple[float, float, float] = config.SPLIT_FRACTIONS,
    seed: int = config.SEED,
    stratify: bool = True,
) -> pd.Series:
    """Split so that every group lands entirely in one partition.

    ``groups`` maps ``doc_id`` -> group id (near-duplicate group or cluster id). Articles
    missing from ``groups`` are treated as singleton groups. Groups are visited in random
    order and each goes to the partition that is furthest below its target size. With
    ``stratify=True`` this is done separately for mostly-fake and mostly-real groups, which
    keeps the class balance close to the full data (use ``stratify=False`` for a few large
    clusters). Returns a Series of train/val/test indexed by ``doc_id``.
    """
    _check_fractions(fractions)
    g = df[["doc_id", "is_fake"]].copy()
    g["group"] = g["doc_id"].map(groups)
    singleton = g["group"].isna()
    g["group"] = g["group"].astype(object)
    g.loc[singleton, "group"] = "__singleton_" + g.loc[singleton, "doc_id"].astype(str)

    stats = g.groupby("group", sort=False)["is_fake"].agg(size="size", fake_share="mean")
    stats["stratum"] = (stats["fake_share"] >= 0.5).astype(int) if stratify else 0

    rng = np.random.default_rng(seed)
    assignment: dict = {}
    for _, part in stats.groupby("stratum"):
        order = rng.permutation(len(part))
        targets = np.array(fractions) * part["size"].sum()
        filled = np.zeros(3)
        for i in order:
            k = int(np.argmin(filled / targets))  # partition furthest below its target
            assignment[part.index[i]] = config.SPLIT_NAMES[k]
            filled[k] += part["size"].iloc[i]

    split = g["group"].map(assignment)
    split.index = pd.Index(g["doc_id"].to_numpy(), name="doc_id")
    split.name = "split"
    return split


def check_split(split: pd.Series, labels: pd.Series | None = None, groups: pd.Series | None = None) -> pd.DataFrame:
    """Validate a split and summarize it.

    Raises ``ValueError`` on unknown partition names, duplicate doc_ids, or (if ``groups``
    is given) a group spread over several partitions. Returns size, fraction and (if
    ``labels`` indexed by doc_id is given) fake rate per partition.
    """
    if split.index.has_duplicates:
        raise ValueError("Split has duplicate doc_ids.")
    bad = set(split.unique()) - set(config.SPLIT_NAMES)
    if bad:
        raise ValueError(f"Unknown partition names: {bad}")
    if groups is not None:
        per_group = split.groupby(split.index.map(groups)).nunique()
        leaking = per_group[per_group > 1]
        if len(leaking):
            raise ValueError(f"{len(leaking)} groups are spread over several partitions, e.g. {list(leaking.index[:5])}")

    summary = split.value_counts().reindex(config.SPLIT_NAMES).fillna(0).astype(int).to_frame("n")
    summary["fraction"] = (summary["n"] / summary["n"].sum()).round(3)
    if labels is not None:
        summary["fake_rate"] = labels.reindex(split.index).groupby(split).mean().round(3)
    return summary


def save_split_column(regime: str, split: pd.Series, overwrite: bool = False) -> None:
    """Add ``split_<regime>`` to ``data/processed/splits.csv`` (created if missing).

    Refuses to replace an existing column unless ``overwrite=True``, and requires the same
    doc_ids as the columns already in the file.
    """
    col = f"split_{regime}"
    path = config.SPLITS_CSV
    new = split.rename(col).rename_axis("doc_id").reset_index()

    if path.exists():
        current = pd.read_csv(path)
        if col in current.columns and not overwrite:
            raise FileExistsError(f"{col} already exists in {path}. Pass overwrite=True to replace it.")
        if set(current["doc_id"]) != set(new["doc_id"]):
            raise ValueError("doc_ids differ from those already in splits.csv; was the clean data rebuilt?")
        out = current.drop(columns=[col], errors="ignore").merge(new, on="doc_id")
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("doc_id").to_csv(path, index=False)


def _check_fractions(fractions: tuple[float, float, float]) -> None:
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9 or min(fractions) <= 0:
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions}")
