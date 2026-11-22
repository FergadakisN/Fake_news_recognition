"""Model C: K-means topic clusters + one logistic regression per cluster. Tune on val, save predictions.

    python scripts/run_model_c.py --regime grouped    # final split: compare K and C on val, pick the best
    python scripts/run_model_c.py --regime random     # keeps K and C chosen on grouped; only the split differs

Design: with --design compare (default for "grouped", the final split) every K in {8, 16, 32} and C in
{10, 30, 100} is scored on val and the Model C setting with the best val ROC-AUC is chosen. With
--design fixed (default for every other regime) K and C are read from reports/model_c/grouped_summary.json,
so regimes differ only in the split.

Steps (everything fitted on TRAIN only):
    1. TF-IDF of content, TruncatedSVD to 200 dimensions, L2 normalisation
    2. K-means on the train SVD vectors, for each K
    3. per C: a global logistic regression (the comparison) and one logistic regression per cluster
       (fallback to the global one for clusters with < 200 train articles or one class)
    4. scored on val next to two references: "topic_only" (P(fake) = fake share of the article's cluster in
       train) and "hybrid_avg" (average of Model C and the global LR at its best C; analysis only, not a
       submitted model)
    5. the chosen setting predicts val + test; predictions and cluster assignments are saved

Test articles are predicted but never scored here (README rule 3: test is used once, at the end, for all models).

Outputs:
    reports/model_c/<regime>_tuning.csv        every (model, K, C) on val
    reports/model_c/<regime>_clusters.csv      per cluster of the chosen K: size, fake share, fallback, top terms, val scores
    reports/model_c/<regime>_summary.json      chosen setting, val metrics of Model C and the references, timings
    data/processed/predictions/model_c__<regime>.csv               doc_id, split, prob_fake (val + test)
    data/processed/predictions/model_c__<regime>__clusters.csv     doc_id, split, cluster (all articles)
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fake_news import config
from fake_news.data import load_dataset
from fake_news.evaluation import evaluate, save_predictions
from fake_news.evaluation.predictions import prediction_path
from fake_news.models import model_c as mc

K_GRID = (8, 16, 32)
C_GRID = (10.0, 30.0, 100.0)
DESIGN_REGIME = "grouped"
REPORT_DIR = config.PROJECT_ROOT / "reports" / "model_c"
METRICS = ("roc_auc", "accuracy", "f1", "precision", "recall")


def scores(y, p) -> dict:
    s = evaluate(y, p)
    return {m: round(s[m], 4) for m in METRICS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default=DESIGN_REGIME, choices=config.REGIMES)
    ap.add_argument("--design", choices=("compare", "fixed"), default=None,
                    help=f"compare K and C on val, or keep the ones chosen on {DESIGN_REGIME} "
                         f"(default: compare for {DESIGN_REGIME}, fixed otherwise)")
    args = ap.parse_args()
    design = args.design or ("compare" if args.regime == DESIGN_REGIME else "fixed")
    if design == "compare":
        k_grid, c_grid = K_GRID, C_GRID
    else:
        path = REPORT_DIR / f"{DESIGN_REGIME}_summary.json"
        if not path.exists():
            sys.exit(f"{path} not found: run `python scripts/run_model_c.py --regime {DESIGN_REGIME}` first.")
        chosen = json.loads(path.read_text())
        k_grid, c_grid = (chosen["k"],), (chosen["c"],)
    timings = {}

    t = time.time()
    df = load_dataset(args.regime, columns=["content"])
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in config.SPLIT_NAMES)
    ytr, yva = train.is_fake.to_numpy(), val.is_fake.to_numpy()
    print(f"[{args.regime}, design {design}] train {len(train):,}  val {len(val):,}  test {len(test):,}")
    timings["load"] = time.time() - t

    t = time.time()
    feats = mc.TopicFeatures()
    Xtr, Ztr = feats.fit_transform(train.content)
    Xva, Zva = feats.transform(val.content)
    timings["features"] = time.time() - t
    print(f"TF-IDF {Xtr.shape[1]:,} words, SVD {feats.n_components} dims "
          f"(explained variance {feats.explained_variance:.1%}), {timings['features']:.0f}s")

    t = time.time()
    globals_, rows = {}, []
    # The global LR (the comparison) is always tuned over the full C grid, also when Model C's K and C are fixed,
    # so that Model C is never compared with an untuned opponent.
    for c in sorted(set(C_GRID) | set(c_grid)):
        globals_[c] = mc.fit_global_lr(Xtr, ytr, c)
        p_glob = globals_[c].predict_proba(Xva)[:, 1]
        rows.append(dict(model="global_lr", k=1, c=c, fallback_clusters=0, **scores(yva, p_glob)))
    # hybrid_avg always averages Model C with the *tuned* global LR (its own best C on val)
    c_glob = max(globals_, key=lambda c: next(r["roc_auc"] for r in rows if r["model"] == "global_lr" and r["c"] == c))
    p_glob_best = globals_[c_glob].predict_proba(Xva)[:, 1]
    kms, fitted = {}, {}
    for k in k_grid:
        kms[k] = km = mc.fit_kmeans(Ztr, k)
        cva = km.predict(Zva)
        for c in c_grid:
            clf = mc.fit_cluster_classifiers(Xtr, ytr, km.labels_, c, globals_[c], k=k)
            p_c = clf.predict_proba(Xva, cva)
            fitted[(k, c)] = clf
            rows.append(dict(model="model_c", k=k, c=c, fallback_clusters=int(clf.fallback.sum()), **scores(yva, p_c)))
            rows.append(dict(model="hybrid_avg", k=k, c=c, fallback_clusters=int(clf.fallback.sum()),
                             **scores(yva, (p_c + p_glob_best) / 2)))
        p_topic = mc.topic_only_proba(fitted[(k, c_grid[0])].fake_share, cva, default=ytr.mean())
        rows.append(dict(model="topic_only", k=k, c=np.nan, fallback_clusters=0, **scores(yva, p_topic)))
        print(f"  K = {k}: done")
    timings["tuning"] = time.time() - t
    tuning = pd.DataFrame(rows)

    best = tuning[tuning.model == "model_c"].sort_values("roc_auc", ascending=False).iloc[0]
    k, c = int(best.k), float(best.c)
    km, clf = kms[k], fitted[(k, c)]
    best_of = lambda m: tuning[tuning.model == m].sort_values("roc_auc", ascending=False).iloc[0]
    print(f"\nChosen: K = {k}, C = {c:g} (max val ROC-AUC of Model C)\n")
    print(tuning.sort_values(["model", "roc_auc"], ascending=[True, False]).groupby("model").head(1)
          .to_string(index=False))

    # predictions of the chosen setting: val + test (test never scored)
    Xte, Zte = feats.transform(test.content)
    c_va, c_te = km.predict(Zva), km.predict(Zte)
    p_va, p_te = clf.predict_proba(Xva, c_va), clf.predict_proba(Xte, c_te)
    preds = pd.concat([pd.DataFrame({"doc_id": val.doc_id, "split": "val", "prob_fake": p_va}),
                       pd.DataFrame({"doc_id": test.doc_id, "split": "test", "prob_fake": p_te})])
    pred_file = save_predictions(preds, model="model_c", regime=args.regime)
    clusters = pd.concat([pd.DataFrame({"doc_id": d.doc_id, "split": s, "cluster": lab})
                          for d, s, lab in ((train, "train", km.labels_), (val, "val", c_va), (test, "test", c_te))])
    cluster_file = prediction_path("model_c", args.regime).with_name(f"model_c__{args.regime}__clusters.csv")
    clusters.sort_values("doc_id").to_csv(cluster_file, index=False)

    # per-cluster table
    p_glob_va = globals_[c].predict_proba(Xva)[:, 1]
    auc = lambda y, p: round(roc_auc_score(y, p), 4) if len(np.unique(y)) == 2 else np.nan
    table = pd.DataFrame([dict(cluster=j, n_train=int(clf.n_train[j]), fake_share_train=round(clf.fake_share[j], 4),
                               fallback=bool(clf.fallback[j]), n_val=int((c_va == j).sum()),
                               val_auc_model_c=auc(yva[c_va == j], p_va[c_va == j]),
                               val_auc_global_lr=auc(yva[c_va == j], p_glob_va[c_va == j]),
                               top_terms=", ".join(terms))
                          for j, terms in enumerate(feats.top_terms(km.cluster_centers_, 12))])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(REPORT_DIR / f"{args.regime}_tuning.csv", index=False)
    table.sort_values("n_train", ascending=False).to_csv(REPORT_DIR / f"{args.regime}_clusters.csv", index=False)
    summary = {
        "regime": args.regime, "design": design, "k": k, "c": c,
        "n_components": feats.n_components, "explained_variance": round(feats.explained_variance, 4),
        "tfidf_words": int(Xtr.shape[1]), "min_cluster_size": mc.MIN_CLUSTER_SIZE,
        "fallback_clusters": int(clf.fallback.sum()),
        "selection": "max val ROC-AUC of model_c over the K x C grid" if design == "compare"
                     else f"K and C fixed from {DESIGN_REGIME}_summary.json",
        "grid": {"k": list(k_grid), "c": list(c_grid)},
        "val": {
            "model_c": scores(yva, p_va),
            "global_lr": {"c": float(best_of("global_lr").c), **{m: best_of("global_lr")[m] for m in METRICS}},
            "topic_only": {"k": k, **{m: tuning[(tuning.model == "topic_only") & (tuning.k == k)].iloc[0][m] for m in METRICS}},
            "hybrid_avg": {"k": k, "c": c, "global_c": float(c_glob), **{m: tuning[(tuning.model == "hybrid_avg") & (tuning.k == k) & (tuning.c == c)].iloc[0][m] for m in METRICS}},
        },
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "seconds": {key: round(v, 1) for key, v in timings.items()},
    }
    (REPORT_DIR / f"{args.regime}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"\nWrote {pred_file.relative_to(config.PROJECT_ROOT)}, {cluster_file.relative_to(config.PROJECT_ROOT)}, "
          f"reports/model_c/{args.regime}_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
