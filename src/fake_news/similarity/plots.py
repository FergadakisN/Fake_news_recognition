"""Figures for the MinHash/LSH section (shared by scripts/find_near_duplicates.py and notebook 04).

Same look as the figures of notebooks 01-03: real = blue, fake = orange, light surface,
recessive grid. matplotlib is imported lazily so the rest of the package does not need it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.similarity import lsh

REAL, FAKE = "#2a78d6", "#eb6834"  # real = blue, fake = orange in every figure
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#2b2a27", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BARS = "#7d7b74"  # neutral bars: a plain count, not a class


def use_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110,
    })


def save(fig, name: str) -> str:
    """Save to reports/figures/<name>.png and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return str(path)


def _threshold_lines(ax, threshold: float | None, reference: float | None) -> None:
    """Solid line at the threshold (label on its left), dotted line at the reference value (label on its right)."""
    lines = [(threshold, f"threshold {threshold}", "-", -4, "right"),
             (reference, f"{reference} (examples)", ":", 4, "left")]
    for x, label, style, dx, ha in lines:
        if x is not None and not (style == ":" and reference == threshold):
            ax.axvline(x, color=INK, linestyle=style, linewidth=1.1)
            ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(dx, -2),
                        textcoords="offset points", ha=ha, va="top", fontsize=9, color=INK2)


def jaccard_histogram(cands: pd.DataFrame, threshold: float | None = None, reference: float | None = 0.8,
                      recall: pd.DataFrame | None = None, bands: int = lsh.BANDS, rows: int = lsh.ROWS):
    """Two panels sharing the x axis (exact Jaccard similarity of a pair):

    top: histogram of the exact Jaccard similarity of all LSH candidate pairs;
    bottom: the LSH S-curve P(candidate | s) for ``bands x rows``, with the measured recall per bin
    from :func:`near_duplicates.binned_recall` if given (share of the pairs found by a looser
    32 x 4 banding that our banding also finds).
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    bins = np.arange(0, 1.0001, 0.025)
    ax.hist(cands["jaccard"], bins=bins, color=BARS, edgecolor=SURFACE, linewidth=1)
    _threshold_lines(ax, threshold, reference)
    ax.set(ylabel="candidate pairs",
           title=f"Exact Jaccard similarity of the {len(cands):,} LSH candidate pairs (3-word shingles)")

    s = np.linspace(0, 1, 201)
    ax2.plot(s, lsh.candidate_probability(s, bands, rows), color=INK2, linewidth=2,
             label=f"P(candidate), {bands} bands x {rows} rows")
    if recall is not None:
        ax2.plot(recall["center"], recall["recall"], "o", color=INK, markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label="measured (vs 32 x 4 banding), per 0.05 bin")
    if threshold is not None:
        ax2.axvline(threshold, color=INK, linewidth=1.1)
    ax2.set(xlabel="Jaccard similarity", ylabel="probability", ylim=(-0.03, 1.05), xlim=(0, 1))
    ax2.legend(loc="upper left", fontsize=9)
    return fig


def estimate_vs_exact(cands: pd.DataFrame, n_hashes: int = 128):
    """MinHash estimate against exact Jaccard for the candidate pairs (the diagonal = perfect estimate)."""
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.scatter(cands["jaccard"], cands["minhash"], s=8, color=BARS, alpha=0.5, linewidths=0)
    err = (cands["minhash"] - cands["jaccard"]).abs()
    ax.set(xlabel="exact Jaccard", ylabel=f"MinHash estimate ({n_hashes} hashes)", xlim=(0.2, 1.01),
           ylim=(0.2, 1.01), title=f"Estimate vs exact (mean |error| {err.mean():.3f})")
    ax.set_aspect("equal")
    return fig


def group_sizes(table: pd.DataFrame):
    """Number of near-duplicate groups per group size (groups with 2+ articles), split by majority label."""
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    majority_fake = table["n_fake"] * 2 >= table["size"]
    sizes = np.arange(2, table["size"].max() + 1)
    real = table.loc[~majority_fake, "size"].value_counts().reindex(sizes, fill_value=0)
    fake = table.loc[majority_fake, "size"].value_counts().reindex(sizes, fill_value=0)
    # Side by side, not stacked: on a log scale stacked segments would misrepresent the shares.
    ax.bar(sizes - 0.2, real, color=REAL, width=0.4, label="mostly real", edgecolor=SURFACE, linewidth=1)
    ax.bar(sizes + 0.2, fake, color=FAKE, width=0.4, label="mostly fake", edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)
    ax.set(xlabel="group size (articles)", ylabel="groups (log scale)",
           title=f"{len(table):,} near-duplicate groups with 2+ articles")
    ax.legend(loc="upper right")
    return fig


def s_curves(curves: pd.DataFrame):
    """Measured vs theoretical LSH S-curve, one panel per banding (scripts/evaluate_lsh.py).

    ``curves`` has ``data`` (sample / full), ``banding`` ("16x8"), ``shingle_words``, ``mean_j``, ``measured``,
    ``ci_low``, ``ci_high``. Line: theoretical P(candidate | s). Hollow dots: the 3,000-article sample (all
    pairs). Filled dots: the full data (pairs with J >= 0.2). Whiskers: 95% Wilson interval.
    """
    import matplotlib.pyplot as plt

    use_style()
    panels = curves[["shingle_words", "banding"]].drop_duplicates().sort_values(["shingle_words", "banding"],
                                                                                ascending=[False, True])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4), sharey=True)
    s = np.linspace(0, 1, 201)
    for ax, (_, p) in zip(np.atleast_1d(axes), panels.iterrows()):
        bands, rows = map(int, p["banding"].split("x"))
        ax.plot(s, lsh.candidate_probability(s, bands, rows), color=INK2, linewidth=2, label="theoretical")
        d = curves[(curves["banding"] == p["banding"]) & (curves["shingle_words"] == p["shingle_words"])]
        for data, face, label in (("sample", SURFACE, "measured, 3,000 sample"), ("full", INK, "measured, full data")):
            m = d[d["data"] == data]
            if len(m):
                ax.errorbar(m["mean_j"], m["measured"], yerr=[m["measured"] - m["ci_low"], m["ci_high"] - m["measured"]],
                            fmt="o", color=INK, markerfacecolor=face, markersize=5, elinewidth=1, capsize=0,
                            label=label)
        ax.set(title=f"{p['banding'].replace('x', ' x ')}, {p['shingle_words']}-word shingles", xlabel="exact Jaccard",
               xlim=(0, 1), ylim=(-0.03, 1.05))
    np.atleast_1d(axes)[0].set_ylabel("P(candidate pair)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
    return fig


def leak_share_curves(curves: pd.DataFrame):
    """Share of val articles whose closest training article has 2-word Jaccard >= t, random vs grouped split."""
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for col, style, color in (("random", "--", MUTED), ("grouped", "-", INK)):
        ax.plot(curves["threshold"], 100 * curves[col], style, color=color, linewidth=2)
        last = curves.iloc[np.searchsorted(curves["threshold"], 0.6)]
        ax.annotate(f"{col} split", (last["threshold"], 100 * last[col]), xytext=(4, 4 if col == "random" else -12),
                    textcoords="offset points", fontsize=9, color=INK2)
    for t in (0.3, 0.5):
        ax.axvline(t, color=AXIS, linewidth=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))   # 0.1, 1, 10 instead of 10^x
    ax.set(xlabel="threshold t (2-word Jaccard of the closest training article)",
           ylabel="val articles with a train neighbour >= t (%)", xlim=(0.1, 1),
           title="How close is the nearest training article? (val, train-only LSH index)")
    return fig


def feature_distributions(feats: pd.DataFrame, labels: pd.Series, part: str = "val"):
    """fake_neighbor_share and max_jaccard of the articles of one partition that have a neighbour, real vs fake.

    ``feats`` is a table from ``load_lsh_features``; articles without neighbours (the fallback value) are left
    out, and their number is in the title.
    """
    import matplotlib.pyplot as plt

    use_style()
    p = feats[feats["split"] == part]
    y = labels.reindex(p["doc_id"]).to_numpy()
    has = p["neighbor_count"].to_numpy() > 0
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, col, bins in ((axes[0], "fake_neighbor_share", np.linspace(0, 1, 11)),
                          (axes[1], "max_jaccard", np.linspace(0.1, 1, 19))):
        values = p[col].to_numpy()
        counts = [np.histogram(values[has & (y == c)], bins=bins)[0] for c in (0, 1)]
        w = (bins[1] - bins[0]) * 0.42
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers - w / 2, counts[0], width=w, color=REAL, label="real", edgecolor=SURFACE, linewidth=1)
        ax.bar(centers + w / 2, counts[1], width=w, color=FAKE, label="fake", edgecolor=SURFACE, linewidth=1)
        ax.set(xlabel=col, ylabel=f"{part} articles")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].set_ylabel(f"{part} articles (log scale)")
    axes[0].legend(loc="upper center")
    fig.suptitle(f"LSH neighbour features of the {has.sum():,} {part} articles with a neighbour "
                 f"({(~has).sum():,} without one are not shown)", x=0.01, ha="left", fontsize=11, color=INK)
    return fig
