# Fake News Recognition

Fake news detection on the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
(DTU 02807). The research question is how much near-duplicate and topic overlap inflate
detection accuracy, and whether a similarity- and cluster-aware pipeline stays accurate on unseen stories.

This README explains how to set up the project and how to use the **shared code**: loading
the data, the splits, `evaluate()` and the prediction file format. Everyone should use these,
so that all models are trained and scored on exactly the same data.

> **Start here**
>
> ```bash
> pip install -r requirements.txt
> pip install -e .                  # makes `import fake_news` work everywhere
> ```
>
> ```python
> from fake_news.data import load_dataset
> df = load_dataset("random")       # cleaned articles + a `split` column (train / val / test)
> ```
>
> Nothing else to prepare. The **first** call downloads the raw data, cleans it and checks it against
> the group's reference (about 3 minutes); every later call takes about 2 seconds.
> Walkthrough: [notebooks/00-how-to-use.ipynb](notebooks/00-how-to-use.ipynb).

---

## 1. Setup (once)

```bash
git clone <repo-url> && cd Fake_news_recognition
python3 -m venv .venv            # optional but recommended (Python 3.10 and 3.12 are tested)
source .venv/bin/activate
pip install --upgrade pip        # editable installs need pip >= 21.3
pip install -r requirements.txt
pip install -e .                 # installs the `fake_news` package from src/
```

Optional: `python3 scripts/prepare_data.py` prepares the data up front and prints a full check report
(it must end with **`All checks passed`**), and `python3 -m pytest` runs the tests. To regenerate every
generated file, see [section 5](#5-reports).

### What does `pip install -e .` do?

The code lives in `src/fake_news/`. Python only finds packages that are installed or on its
search path, so without this step `import fake_news` fails in notebooks (which run from
`notebooks/`) and in scripts (which run from `scripts/`).

`pip install -e .` reads [pyproject.toml](pyproject.toml) and installs the package in
**editable** mode: Python is pointed at `src/`, nothing is copied. So:

- `from fake_news... import ...` works from any notebook, script or test, with no `sys.path` hacks.
- Edits to files in `src/` take effect immediately; you never need to reinstall.
- Run it once per Python environment (again if you create a new venv or switch the notebook kernel).

**Notebooks cache imported modules.** After pulling or editing code in `src/`, restart the kernel,
or put this at the top of your notebook:

```python
%load_ext autoreload
%autoreload 2
```

---

## 2. Shared conventions

| Convention | Value |
|---|---|
| Article key | `doc_id` = row number in the original `WELFake_Dataset.csv`. Every shared file is joined on it. |
| Label | `is_fake`: **1 = fake, 0 = real**. (The Kaggle page says the opposite, but the data shows otherwise: "(Reuters)" datelines appear in 21,266 label-0 articles and only 16 label-1 ones.) |
| Text to model on | `content`: cleaned title + text, lower-cased. |
| Splits | `data/processed/splits.csv`: `doc_id, split_random[, split_grouped, split_cluster]`, values `train` / `val` / `test`, 70/15/15. **Committed to git.** |
| Seed | `fake_news.config.SEED` (42). |
| Metrics | `evaluate(y_true, y_prob)` with fake as the positive class. |
| Predictions | One CSV per model and regime: `doc_id, split, prob_fake`, written with `save_predictions()`. |
| Embeddings | `embeddings_for(doc_ids)`: 384-d Sentence-BERT vectors (`all-MiniLM-L6-v2`) of `content`, L2-normalized, in `data/processed/embeddings/`. **Committed to git.** |

### Rules that keep the results honest

1. **Fit everything on `train` only**: TF-IDF vocabularies, scalers, K-means, the LSH index, classifiers.
2. **Tune on `val`.** Hyperparameters, thresholds and the choice of K are picked on validation.
3. **Touch `test` once**, in the final run, for all models together.
4. **Never re-generate `splits.csv` on your own machine.** Pull it from git. Changes to it go through Person 1.

The WELFake paper breaks the first three rules (vectorizer fit on all data, tuned on the test set,
split ratio chosen on test accuracy). Avoiding that is part of our contribution.

---

## 3. Using the shared code

### Load data

```python
from fake_news.data import load_dataset

df = load_dataset("random")          # doc_id, title, text, content, n_words, is_fake, split
train = df[df.split == "train"]
val   = df[df.split == "val"]

df = load_dataset("random", columns=["content"])   # less memory: doc_id, content, is_fake, split
```

`load_dataset(regime)` takes `"random"`, `"grouped"` (near-duplicate groups kept together) and `"cluster"`
(whole topic clusters held out, see below).

| Function (`from fake_news.data import ...`) | Returns |
|---|---|
| `load_dataset(regime, columns=None)` | Cleaned articles + `split` column for that regime |
| `load_clean(remove_markers=True, columns=None)` | Cleaned articles without split. `remove_markers=False` gives the same rows with source markers kept, only for measuring their effect |
| `load_labels()` | Just `doc_id, is_fake` (a few MB) |
| `load_splits()` | The `splits.csv` table |
| `load_raw()` | Original rows as `doc_id, title, text, label` |

The full table takes about 1–1.5 GB of memory; `columns=["content"]` about half.

### Evaluate and save predictions

```python
import pandas as pd
from fake_news.evaluation import evaluate, save_predictions, load_predictions, evaluate_predictions

scores = evaluate(val.is_fake, prob_fake)   # prob_fake = your model's P(fake) for each val row
# {'n', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'tn', 'fp', 'fn', 'tp'}

preds = pd.DataFrame({"doc_id": val.doc_id, "split": "val", "prob_fake": prob_fake})
save_predictions(preds, model="my_model", regime="random")   # -> data/processed/predictions/my_model__random.csv
evaluate_predictions(load_predictions("my_model", "random"), split="val")
```

`save_predictions` checks the format (columns, no duplicate `doc_id`, probabilities in [0, 1]),
so Model D can join any two files on `doc_id` without surprises.

### Add a split regime (MinHash pair / Person 2 → Person 1)

```python
from fake_news.data import load_clean
from fake_news.data.split import make_group_split, check_split, save_split_column

clean = load_clean(columns=[])
groups = ...                      # pd.Series: doc_id -> near-duplicate group id (or cluster id)
split = make_group_split(clean, groups)                     # whole groups stay in one partition
print(check_split(split, labels=clean.set_index("doc_id").is_fake, groups=groups))  # raises if a group leaks
save_split_column("grouped", split)                         # adds split_grouped to splits.csv
```

For a few large K-means clusters use `make_group_split(clean, clusters, stratify=False)`.
`save_split_column` refuses to overwrite an existing column unless `overwrite=True`.

### The cluster split (Person 2)

`split_cluster` holds out whole **topic clusters**: val and test contain topics that no training article belongs to.
It is the hardest of the three regimes and tests whether a model generalises to unseen stories.

```bash
python scripts/build_cluster_split.py      # ~25 s; refuses to overwrite split_cluster (pass --overwrite)
```

1. K-means (K = 32) on the sentence embeddings of **all** cleaned articles. No labels are used, so, like the
   near-duplicate groups, the clusters can exist before the split. (K-means *inside* a model is still fitted on train.)
2. A near-duplicate group whose copies fall into different clusters is moved into its majority cluster (258 articles),
   so no copy of a story crosses partitions.
3. Whole clusters are assigned with `make_group_split(stratify=False)`. 500 assignment orders are tried and the one
   closest to 70/15/15 with the overall fake share in every partition is kept (at least 3 clusters in val and in test).
   Only sizes and the overall label balance are used, never a model result.

| Partition | Share | Fake share | Held-out topics |
|---|---|---|---|
| train | 69.3% | 44.3% | 22 clusters |
| val | 15.0% | 44.9% | Brexit, Clinton emails/FBI, North Korea, Russia investigation, police/shootings |
| test | 15.7% | 44.4% | health/medicine, Russia/Putin, White House staff, Democratic primaries, Republican primaries |

Neighbouring topics can stay in train (e.g. other Russia or election clusters), so "unseen" means an unseen
*cluster*, not a subject that never occurs. Files: `data/processed/topic_clusters.csv` (`doc_id, cluster`, in git),
`reports/cluster_split/clusters.csv` (size, fake share, partition, top words per cluster) and `summary.json`.

### Sentence embeddings (Person 3)

The embeddings are committed to git: after pulling, nothing needs to be computed.

```python
from fake_news.features.embeddings import embeddings_for

X = embeddings_for(train.doc_id)     # (n, 384) float32, row i belongs to train.doc_id.iloc[i]
```

- The encoder is pretrained and never fitted on our data, so embeddings cover all cleaned articles and do not depend
  on the split. Anything fitted *on* them (PCA, K-means, classifiers) is fitted on `train` only.
- Vectors are normalized: cosine similarity is `X[i] @ X[j]`, and Euclidean K-means behaves like cosine K-means.
- Always load them with `embeddings_for`, never with `np.load` directly: it matches rows to `doc_id`s and checks
  that the embeddings belong to the current cleaned data (`StaleEmbeddingsError` otherwise).

Details and figures: [notebooks/03-embeddings.ipynb](notebooks/03-embeddings.ipynb).

### Near-duplicates and LSH neighbour features (MinHash/LSH)

MinHash and LSH are implemented from scratch in `src/fake_news/similarity/` (NumPy and the standard library only).
Walkthrough and figures: [notebooks/04-minhash-lsh.ipynb](notebooks/04-minhash-lsh.ipynb).

**Near-duplicate groups** (input for `split_grouped`) have two tiers, merged with union-find:

| Tier | Finds | How | Script |
|---|---|---|---|
| MinHash | near-copies (same wording) | 3-word shingles, 128 MinHash values, LSH 16 × 8, exact Jaccard ≥ 0.7 | `find_near_duplicates.py` |
| Embeddings | paraphrases and copies MinHash misses (e.g. truncated copies) | MiniLM vectors, exact top-10 cosine neighbours: cosine ≥ 0.92, or 0.90–0.92 with 3-word containment ≥ 0.7 | `find_paraphrases.py` |

```bash
python scripts/find_near_duplicates.py                          # MinHash tier, ~1 min
python scripts/find_paraphrases.py --write                      # embedding tier + merge -> final groups, ~1 min
python scripts/build_grouped_split.py                           # rebuild split_grouped from the final groups
python scripts/build_lsh_features.py --regime grouped           # then rebuild the grouped features
```

- `data/processed/near_dup_groups.csv` (**in git**): the final groups, `doc_id, group_id, tier` for every cleaned
  article. `group_id` is the smallest `doc_id` in the group (`group_id == doc_id` for an article without
  near-duplicates). `tier` says what linked the group: `minhash`, `embedding`, `both` (a MinHash group enlarged or
  joined by embedding links) or `none` (singleton); `+containment` is appended (`embedding+containment`,
  `both+containment`) when a link from the containment rule changed the group. Why these rules:
  [LSH_FEATURES_DESIGN.md](reports/minhash/LSH_FEATURES_DESIGN.md).
- `data/processed/near_dup_groups_minhash.csv` (**in git**): the MinHash tier alone (`doc_id, group_id`), the input
  of the merge.
- `reports/minhash/`: statistics, threshold examples and comparisons for both tiers, the largest and mixed-label
  groups, and `NOTE_cleaning_residue.md` (cleaning residue for Person 1).

```python
from fake_news.similarity.near_duplicates import load_groups
groups = load_groups()               # pd.Series doc_id -> group_id (final groups), for make_group_split(clean, groups)
```

**LSH neighbour features** (input for the final classifier):

```bash
python scripts/build_lsh_features.py --regime random grouped     # ~10 min per regime
python scripts/build_lsh_features.py --regime grouped --seed 1 2  # splits generated in code (nothing written to splits.csv)
```

For every article in train, val and test, a train-only LSH index (2-word shingles, 64 × 2 bands, exact Jaccard
≥ 0.1) finds similar training articles and gives three features:

| Feature | Meaning | Without neighbours |
|---|---|---|
| `fake_neighbor_share` | similarity-weighted share of fake neighbours | training fake rate (≈ 0.444) |
| `max_jaccard` | similarity of the closest neighbour | 0 |
| `neighbor_count` | number of neighbours with J ≥ 0.1 | 0 |

**Exclusion rule.** Training articles are in the index themselves, so a training article never counts itself as a
neighbour (random split). In the grouped and cluster splits it also ignores its whole near-duplicate group, because a
val/test article never has a group member in train there. This keeps train features comparable to val/test
features. Val/test features use training labels only.

- `data/processed/features/lsh__<regime>.csv` (**in git**, ~2 MB each): `doc_id, split` and the three features,
  for `random`, `grouped` and `cluster` (`lsh__cluster.csv`, built from `split_cluster` with
  `python scripts/build_lsh_features.py --regime cluster`). Under the cluster split the features rarely fire on the
  held-out topics: 95% of val articles have no neighbour, against 82% of train articles
  ([REPORT_NOTES.md](reports/minhash/REPORT_NOTES.md)). Files for generated splits (`lsh__<regime>__seed<N>.csv`) are not in git; rebuild them with the command above.
- `reports/minhash/lsh_features_sanity_<regime>.csv`: per partition, the share without neighbours, `max_jaccard`
  and `fake_neighbor_share` for fake vs real articles (train and val should look alike).

```python
from fake_news.similarity.lsh_features import load_lsh_features, FEATURE_COLUMNS
feats = load_lsh_features("grouped")      # doc_id, split, fake_neighbor_share, max_jaccard, neighbor_count
df = load_dataset("grouped").merge(feats.drop(columns="split"), on="doc_id")
```

`load_lsh_features` checks that the file was built for the current `split_<regime>` in `splits.csv` and tells you
to rebuild it otherwise.

**How to use the features in a model:**

- **Never use `fake_neighbor_share` alone.** About 80% of articles have no neighbour and get the constant fallback
  (≈ 0.444). Only together with `neighbor_count` or `max_jaccard` can the model tell "no evidence" apart from
  "neighbours split evenly".
- **Use `log1p(neighbor_count)`** instead of the raw count, which is heavily skewed (mostly 0, a few articles have
  dozens of neighbours).
- **Standardize all three features**, with the mean and standard deviation of the training rows only.

**Neighbour voting analysis** (`scripts/run_model_b.py`): **not one of the pipeline's models.** It uses the
neighbours on their own as a classifier on val, to show how informative they are, and it is how the retrieval
design above was chosen ([reasons](reports/minhash/LSH_FEATURES_DESIGN.md)). It only uses val, writes no
predictions, and saves its tables to `reports/lsh_features/voting_<regime>_*`.

**Evaluation** (`scripts/evaluate_lsh.py`, ~35 min the first time, then cached): LSH against exact Jaccard (recall,
false positives, S-curves, runtime) on a 3,000-article sample and on the full data, the feature index against exact
search, and a leakage audit of `split_grouped` on val. Tables in `reports/minhash/lsh_eval_*` and `leakage_audit_*`;
all parameters, key numbers and limitations in [REPORT_NOTES.md](reports/minhash/REPORT_NOTES.md).

### Topic clusters and Model C (K-means, Person 2)

Model C groups the articles into topics with K-means and trains one logistic regression per topic (cluster routing).
Walkthrough, figures and the analysis of the topic shortcut: [notebooks/05-kmeans.ipynb](notebooks/05-kmeans.ipynb).

1. Classifier features: TF-IDF of `content` (fitted on train).
2. Cluster vectors (`--cluster-on`): the sentence **embeddings** (default) or TF-IDF reduced to 200 dimensions with
   TruncatedSVD (`svd`, a comparison). K-means on the train vectors (k-means++, 10 restarts).
3. One logistic regression per cluster on the TF-IDF features. A cluster with fewer than 200 train articles or only one
   class falls back to a global logistic regression.
4. A new article goes to its nearest centroid and is scored by that cluster's classifier.

```bash
python scripts/run_model_c.py --regime grouped                    # ~40 s: compares K in {8, 16, 32} x C in {10, 30, 100} on val
python scripts/run_model_c.py --regime random                     # keeps the K and C chosen on grouped; also --regime cluster
python scripts/run_model_c.py --regime grouped --cluster-on svd   # the TF-IDF + SVD comparison (files tagged __svd / _svd)
```

```python
from fake_news.models.model_c import ModelC
model = ModelC(k=8, c=100).fit(train.content, train.is_fake, train.doc_id)     # doc_ids look up the embeddings
prob_fake, cluster = model.predict_proba(val.content, val.doc_id), model.assign(val.content, val.doc_id)
```

The setting is chosen on grouped val by Model C's ROC-AUC (**K = 8, C = 100** in both spaces) and kept on the other
splits. Every run also scores a global logistic regression on the same features (the "no routing" configuration, tuned
over the same C values) and a `topic_only` reference that predicts the cluster's train fake share.

| Val ROC-AUC (K = 8) | random | grouped | cluster |
|---|---|---|---|
| global LR (no routing) | 0.978 | 0.979 | 0.975 |
| Model C, embedding clusters | 0.975 | 0.975 | 0.962 |
| Model C, SVD clusters | 0.976 | 0.977 | 0.969 |
| cluster alone (embeddings / SVD) | 0.71 / 0.75 | 0.71 / 0.75 | 0.72 / 0.77 |

Cluster routing does not beat one global classifier, and it loses clearly more on unseen topics (`cluster`: −2.7
accuracy points with embedding clusters vs −0.75 for the global LR). Averaged with the global LR it gives a small, real
gain on seen topics (+0.0015; controls in notebook 05), which disappears on unseen topics.

- `data/processed/predictions/model_c[_svd]__<regime>.csv`: `doc_id, split, prob_fake` for val and test.
- `data/processed/predictions/model_c[_svd]__<regime>__clusters.csv`: `doc_id, split, cluster` for every article.
- `reports/model_c/<regime>[__svd]_tuning.csv`, `_clusters.csv`, `_summary.json`: every setting on val, a per-cluster
  table (size, fake share, characteristic words, val ROC-AUC of Model C and the global model), and the chosen setting.

### Paths and constants

```python
from fake_news import config
config.SEED, config.SPLIT_FRACTIONS, config.SPLITS_CSV, config.PREDICTIONS_DIR, config.FIGURES_DIR, config.EMBEDDINGS_DIR
```

---

## 4. The data

| File | In git? | Contents |
|---|---|---|
| `data/raw/WELFake_Dataset.csv` | no | Original 72,134 rows (downloaded automatically) |
| `data/interim/welfake_clean.csv` | no | 62,525 cleaned articles (built automatically) |
| `data/interim/welfake_clean_with_markers.csv` | no | Same rows, source markers kept |
| `data/processed/splits.csv` | **yes** | Shared splits |
| `data/processed/clean_fingerprint.json` | **yes** | Row counts + hashes of the cleaned data |
| `data/processed/embeddings/minilm__clean{.npy,__doc_ids.npy,.json}` | **yes** | Sentence embeddings (float16, ~48 MB), their `doc_id`s and metadata |
| `data/processed/near_dup_groups.csv` | **yes** | Near-duplicate groups: `doc_id, group_id, tier` (MinHash J ≥ 0.7 + embeddings cosine ≥ 0.92 / containment rule) |
| `data/processed/near_dup_groups_minhash.csv` | **yes** | MinHash tier of the groups alone: `doc_id, group_id` |
| `data/processed/features/lsh__<regime>.csv` | **yes** | LSH neighbour features per article (`build_lsh_features.py`) |
| `data/processed/predictions/<model>__<regime>.csv` | no | `doc_id, split, prob_fake` |

### What cleaning does

The principle: **remove what identifies the outlet or the scraping process, keep what the author wrote.**
WELFake merges four source datasets; text such as Reuters datelines, image credits or embedded-tweet
lines reveals the source, and with it the label. Left in, a classifier learns *who published the
article*, not what it says.

1. **Normalize** HTML entities, unicode, quotes, glued sentences, whitespace.
2. **Remove markers**: 44 rules (datelines, outlet suffixes, credits, bylines, boilerplate, embeds, links).
3. **Build `content`**: title + text, lower-cased. No stemming or stop-word removal (embeddings and
   MinHash need natural text); do model-specific preprocessing inside your own model.
4. **Drop rows**: empty texts (883), non-English texts (462), exact duplicates (8,264; lowest `doc_id` kept).

**Result: 62,525 articles, 44.4% fake** (raw: 72,134, 51.4% fake). Kept on purpose: house style,
topic and time, outlet names inside sentences, very short texts and title labels (they are what the
authors wrote; their effect is part of the results).

| Read more | Where |
|---|---|
| Description of the raw dataset | [notebooks/01-data-analysis.ipynb](notebooks/01-data-analysis.ipynb) |
| How the rules were found, every decision, before/after, effect | [notebooks/02-data-cleaning.ipynb](notebooks/02-data-cleaning.ipynb) |
| The decision record | [reports/cleaning/DECISIONS.md](reports/cleaning/DECISIONS.md) |
| Every rule with group, source and a real example | [src/fake_news/data/markers.py](src/fake_news/data/markers.py) |

### Is my data identical to everyone else's?

Yes, or you get an error. The raw file is the pinned Kaggle version 77 (SHA-256 checked), and every
load compares the cleaned data with `clean_fingerprint.json` (row counts and hashes). If your cached
copy is outdated, e.g. after pulling new cleaning rules, it is rebuilt automatically. A
`DataMismatchError` appears only if a fresh build still differs (e.g. an incompatible library
version); the message says what to do. Building streams the data in chunks (about 0.7 GB of memory)
and never leaves a half-written file.

**Person 1, after changing the cleaning on purpose:**
`python3 scripts/prepare_data.py --rebuild --update-fingerprint --new-split`, then commit
`data/processed/` and `reports/cleaning/`.

---

## 5. Reports

Everything needed to describe the data and the cleaning in the report:

| Path | What it is |
|---|---|
| `reports/figures/` | Figures saved by notebooks 01, 02 and 03 |
| `reports/cleaning/DECISIONS.md` | The cleaning decision record (written by hand): principle, process, every decision with its reason |
| `reports/cleaning/cleaning_summary.json` | Rows per cleaning step, dropped rows per reason and class, rules version |
| `reports/cleaning/marker_hits.csv` | Per cleaning rule: articles it changed, per class (raw and cleaned) |
| `reports/cleaning/dropped_rows.csv` | Every dropped `doc_id` with its reason (and, for duplicates, the copy that was kept) |

Everything except `DECISIONS.md` is generated by code. To regenerate all of it, in this order:

```bash
python scripts/prepare_data.py                     # data, splits, fingerprint, cleaning log (~3 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # notebooks, figures, leak scans
```

The leak scans run inside `notebooks/02-data-cleaning.ipynb` (counting only, training split). They list
the phrases that are ≥ 98% one class: first on the data cleaned with only the 15 round-1 rules (rebuilt
in a temporary folder), which revealed the problems behind the round-2 rules, then on the final data.
Teammates don't need to run them.

---

## 6. Tests

```bash
python3 -m pytest
```

Unit tests on small examples, plus checks of your real data against the reference
(`tests/test_data_integrity.py`, skipped if the data has not been built yet).

---

## Project structure

```
├── configs/            Experiment / model configuration files
├── data/
│   ├── raw/            Original WELFake CSV (not in git)
│   ├── external/       Third-party data
│   ├── interim/        Cleaned data (not in git)
│   └── processed/      splits.csv, clean_fingerprint.json, near_dup_groups[_minhash].csv, features/, embeddings/ (in git), predictions/ (not in git)
├── models/             Trained and serialized models
├── notebooks/          00-how-to-use, 01-data-analysis, 02-data-cleaning, 03-embeddings, 04-minhash-lsh, 05-kmeans
├── reports/
│   ├── cleaning/       Cleaning log, DECISIONS.md (in git)
│   ├── minhash/        Near-duplicate statistics, LSH feature design and sanity checks, LSH evaluation,
│   │                   REPORT_NOTES.md, note on cleaning residue
│   ├── lsh_features/   Neighbour voting analysis on val (how informative the neighbours are)
│   ├── model_c/        Model C tuning, per-cluster tables and chosen setting
│   └── figures/        Figures for the report
├── scripts/            prepare_data.py (data), compute_embeddings.py (embeddings),
│                       find_near_duplicates.py + find_paraphrases.py (groups), build_grouped_split.py,
│                       build_lsh_features.py (LSH features),
│                       run_model_b.py (neighbour voting analysis, not a pipeline model),
│                       evaluate_lsh.py (LSH vs exact Jaccard, leakage audit),
│                       run_model_c.py (Model C: K-means + per-cluster LR)
├── src/fake_news/
│   ├── config.py       Paths, seed, split fractions
│   ├── data/           load.py, clean.py, markers.py (rules + decisions), split.py, integrity.py
│   ├── features/       Text preprocessing and feature extraction; embeddings.py (sentence embeddings)
│   ├── similarity/     MinHash, LSH, near-duplicate groups, LSH neighbour features (from scratch)
│   ├── models/         Training and prediction; model_c.py (K-means + per-cluster LR)
│   └── evaluation/     metrics.py (evaluate), predictions.py (shared prediction files)
├── tests/              Unit tests
├── pyproject.toml      Package definition for `pip install -e .`
└── requirements.txt
```