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

`load_dataset(regime)` takes `"random"` now; `"grouped"` (near-duplicate groups kept together) and
`"cluster"` (whole K-means clusters held out) appear once those columns are added to `splits.csv`.

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
save_predictions(preds, model="model_b", regime="random")   # -> data/processed/predictions/model_b__random.csv
evaluate_predictions(load_predictions("model_b", "random"), split="val")
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

### Paths and constants

```python
from fake_news import config
config.SEED, config.SPLIT_FRACTIONS, config.SPLITS_CSV, config.PREDICTIONS_DIR, config.FIGURES_DIR
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
| `reports/figures/` | Figures saved by notebooks 01 and 02 |
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
│   └── processed/      splits.csv, clean_fingerprint.json (in git), predictions/ (not in git)
├── models/             Trained and serialized models
├── notebooks/          00-how-to-use, 01-data-analysis, 02-data-cleaning
├── reports/
│   ├── cleaning/       Cleaning log, DECISIONS.md (in git)
│   └── figures/        Figures for the report
├── scripts/            prepare_data.py (data)
├── src/fake_news/
│   ├── config.py       Paths, seed, split fractions
│   ├── data/           load.py, clean.py, markers.py (rules + decisions), split.py, integrity.py
│   ├── features/       Text preprocessing and feature extraction
│   ├── models/         Training and prediction
│   └── evaluation/     metrics.py (evaluate), predictions.py (shared prediction files)
├── tests/              Unit tests
├── pyproject.toml      Package definition for `pip install -e .`
└── requirements.txt
```
