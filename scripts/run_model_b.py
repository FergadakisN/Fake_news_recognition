"""ANALYSIS TOOL, not a pipeline model: how informative are the LSH neighbours on their own?

The neighbour vote is used as a stand-alone classifier on val: prob_fake = similarity-weighted share of fake
labels among the top-k training neighbours with Jaccard >= min_sim (train fake rate if there are none). The
pipeline itself does not use this vote; it uses the neighbour features from scripts/build_lsh_features.py.
This script is how the retrieval design of those features (2-word shingles, 64 x 2 bands) was chosen.

    python scripts/run_model_b.py                     # regime "random": compare retrieval designs, pick one
    python scripts/run_model_b.py --regime grouped    # same design as random; only k and min_sim retuned

--design compare (default for "random"): every retrieval method below is compared on val and the best is
chosen (highest val ROC-AUC). --design fixed (default for every other regime): the design chosen on random
(read from reports/lsh_features/voting_random_summary.json) is kept, so regimes differ only in the split;
k and min_sim are retuned on that regime's val, and the random-chosen k / min_sim is also scored.

Only val is used: no test article is queried or scored, and no prediction file is written.
Main metrics: coverage, F1 on covered articles, ROC-AUC; overall F1 is secondary.

Outputs:
    reports/lsh_features/voting_<regime>_comparison.csv   best setting per retrieval method (val)
    reports/lsh_features/voting_<regime>_tuning.csv       every (method, k, min_sim) on val
    reports/lsh_features/voting_<regime>_summary.json     chosen setting, val metrics
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.data import load_dataset
from fake_news.similarity import lsh_features as lf

METHODS = {  # name -> (bands, rows, exact search?, words per shingle)
    "lsh_16x8": (16, 8, False, 3),
    "lsh_32x4": (32, 4, False, 3),
    "lsh_64x2": (64, 2, False, 3),
    "exact": (64, 2, True, 3),        # every training article compared; bands/rows unused
    "lsh_64x2_k2": (64, 2, False, 2),  # ablation: 2-word shingles
}
REPORT_DIR = config.PROJECT_ROOT / "reports" / "lsh_features"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="random", choices=config.REGIMES)
    ap.add_argument("--skip-exact", action="store_true", help="leave out the exact-search reference (~1-2 min)")
    ap.add_argument("--design", choices=("compare", "fixed"), default=None,
                    help="compare retrieval designs, or keep the one chosen on random (default: compare for "
                         "random, fixed otherwise)")
    args = ap.parse_args()
    regime = args.regime
    design = args.design or ("compare" if regime == "random" else "fixed")

    methods, reference = METHODS, None
    if design == "fixed":
        ref_path = REPORT_DIR / "voting_random_summary.json"
        if not ref_path.exists():
            print(f"{ref_path} not found: run the random regime first (python scripts/run_model_b.py).")
            return 1
        reference = json.loads(ref_path.read_text())   # design + k/min_sim chosen on random val
        methods = {reference["chosen_method"]: METHODS[reference["chosen_method"]]}
        print(f"fixed design from random: {reference['chosen_method']} "
              f"(random setting k={reference['k']}, min_sim={reference['min_sim']})")

    try:
        df = load_dataset(regime, columns=["content"])
    except KeyError as e:
        print(f"{e}\nsplit_{regime} is not in splits.csv yet. Pull the newest splits.csv, then re-run.")
        return 1
    train, val = (df[df.split == s].reset_index(drop=True) for s in ("train", "val"))   # no test articles
    print(f"regime {regime!r}: train {len(train):,}, val {len(val):,}")

    encodings: dict[int, dict] = {}  # words per shingle -> {"train"|"val": Encoded}

    def encoded(shingle_k: int) -> dict:
        """Shingles + signatures of train and val, computed once per shingle size."""
        if shingle_k not in encodings:
            t0 = time.perf_counter()
            encodings[shingle_k] = {name: lf.encode(part["content"].tolist(), shingle_k=shingle_k)
                                    for name, part in (("train", train), ("val", val))}
            print(f"  ({shingle_k}-word shingles + signatures: {time.perf_counter() - t0:.0f} s)")
        return encodings[shingle_k]

    # ---- compare retrieval methods on val -----------------------------------------------------------------
    grids, best_rows = [], []
    for name, (bands, rows, exact, shingle_k) in methods.items():
        if exact and args.skip_exact:
            continue
        enc = encoded(shingle_k)
        t0 = time.perf_counter()
        knn = lf.LSHNeighborIndex(bands, rows, shingle_k=shingle_k).fit(None, train["is_fake"], encoded=enc["train"])
        fit_s = time.perf_counter() - t0
        # same prefilter as the features (identical neighbours, faster); not applicable to exact search
        nb = knn.neighbors(encoded=enc["val"], exact=exact, min_agree=0 if exact else lf.MIN_AGREE)
        grid = lf.tune(knn, nb, val["is_fake"]).assign(method=name)
        best = lf.best_setting(grid)
        best_rows.append({
            "method": name, "shingle_words": shingle_k, "k": int(best["k"]), "min_sim": best["min_sim"],
            **{c: best[c] for c in ("coverage", "coverage_fake", "coverage_real", "f1_covered", "accuracy_covered",
                                    "f1_overall", "accuracy_overall", "roc_auc_overall")},
            "coverage_at_0.1": grid.loc[grid.min_sim == lf.MIN_SIM_GRID[0], "coverage"].iloc[0],
            "candidates_per_query": nb.n_candidates.mean(),
            "candidates_p90": np.quantile(nb.n_candidates, 0.9),
            "ms_per_query": 1000 * (nb.lookup_seconds + nb.scoring_seconds) / len(nb),
            "index_seconds": fit_s,
        })
        grids.append(grid)
        if reference is not None:
            # the k / min_sim chosen on random val, scored unchanged on this regime's val: effect of the split alone
            prob, covered = knn.predict_proba(nb, int(reference["k"]), float(reference["min_sim"]))
            best_rows.append({**best_rows[-1], "method": "random_setting", "k": int(reference["k"]),
                              "min_sim": float(reference["min_sim"]), **lf.score(prob, covered, val["is_fake"])})
            print(f"  random setting (k={reference['k']}, min_sim={reference['min_sim']}) on {regime} val: "
                  f"coverage {best_rows[-1]['coverage']:.3f}  F1 covered {best_rows[-1]['f1_covered']:.3f}  "
                  f"AUC {best_rows[-1]['roc_auc_overall']:.3f}")
        del knn, nb  # an index holds a copy of the train shingles
        print(f"  {name:12s} best k={int(best['k']):2d} min_sim={best['min_sim']:.2f}  coverage {best['coverage']:.3f}  "
              f"F1 covered {best['f1_covered']:.3f}  AUC {best['roc_auc_overall']:.3f}  "
              f"F1 overall {best['f1_overall']:.3f}  "
              f"{best_rows[-1]['candidates_per_query']:.0f} cand/query  {best_rows[-1]['ms_per_query']:.2f} ms/query")

    comparison = pd.DataFrame(best_rows).set_index("method")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(REPORT_DIR / f"voting_{regime}_comparison.csv")
    pd.concat(grids).round(4).to_csv(REPORT_DIR / f"voting_{regime}_tuning.csv", index=False)

    # ---- chosen design: best LSH method on val (AUC; ties: F1 on covered, then speed) ---------------------
    lsh_rows = comparison.loc[[m for m in comparison.index if m.startswith("lsh_")]]
    chosen = lsh_rows.sort_values(["roc_auc_overall", "f1_covered", "ms_per_query"],
                                  ascending=[False, False, True]).index[0]
    k, min_sim = int(comparison.at[chosen, "k"]), float(comparison.at[chosen, "min_sim"])
    metric_names = ("coverage", "coverage_fake", "coverage_real", "f1_covered", "accuracy_covered",
                    "f1_overall", "accuracy_overall", "roc_auc_overall")
    val_metrics = {m: comparison.at[chosen, m] for m in metric_names}
    print(f"\nchosen on val: {chosen}, k={k}, min_sim={min_sim}")

    summary = {
        "regime": regime, "design": design, "chosen_method": chosen, "shingle_words": METHODS[chosen][3],
        "bands": METHODS[chosen][0], "rows": METHODS[chosen][1], "k": k, "min_sim": min_sim,
        "fallback": "train fake rate", "selection": "max val ROC-AUC over LSH methods",
        "metric_note": "main: coverage, f1_covered, roc_auc_overall; secondary: f1_overall, accuracy_overall",
        "val": {m: round(float(v), 4) for m, v in val_metrics.items()},
        "n": {"train": len(train), "val": len(val)},
    }
    if reference is not None:
        rs = comparison.loc["random_setting"]
        summary["random_setting_on_val"] = {"k": int(rs["k"]), "min_sim": float(rs["min_sim"]),
                                            **{m: round(float(rs[m]), 4) for m in val_metrics}}
    (REPORT_DIR / f"voting_{regime}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nval metrics of the chosen setting:")
    print(pd.Series(val_metrics).round(3).to_string())
    print("\nComparison on val (best k / min_sim per method):")
    with pd.option_context("display.width", 250):
        print(comparison.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
