# Fake News Recognition

Fake news detection on the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
(DTU 02807). The research question is how much near-duplicate and topic overlap inflate
detection accuracy, and whether a similarity- and cluster-aware pipeline stays accurate on unseen stories.

This README explains how to set up the project and how to use the **shared code**: loading
the data, the splits, `evaluate()` and the prediction file format. Everyone should use these,
so that all models are trained and scored on exactly the same data.

For a runnable walkthrough, open [notebooks/00-how-to-use.ipynb](notebooks/00-how-to-use.ipynb).

> **Start here: run these once, before opening any notebook or running any model**
>
> ```bash
> pip install -r requirements.txt
> pip install -e .                  # makes `import fake_news` work everywhere
> python3 scripts/prepare_data.py   # downloads, cleans and verifies the data (~3 min)
> ```
>
> `prepare_data.py` must finish with **`All checks passed`**. Only then is your data identical to
> everyone else's. Run it again (`--rebuild`) whenever you pull changes to `src/fake_news/data/`
> or to `data/processed/`. Details in [section 1](#1-setup-once).

---

## 1. Setup (once)

```bash
git clone <repo-url> && cd Fake_news_recognition
python3 -m venv .venv            # optional but recommended
source .venv/bin/activate
pip install --upgrade pip        # editable installs need pip >= 21.3
pip install -r requirements.txt
pip install -e .                 # installs the `fake_news` package from src/
python3 scripts/prepare_data.py  # downloads + cleans + verifies the data (~3 minutes the first time)
python3 -m pytest                # optional: unit tests + checks that your data matches the reference
```

`prepare_data.py` must end with **`All checks passed`**. If it stops with `DataMismatchError`,
your data differs from the group's; the message says what to do (see [section 3](#3-data)).

### What does `pip install -e .` do?

The code lives in `src/fake_news/`. Python only finds packages that are installed or on its
search path, so without this step `import fake_news` fails in notebooks (which run from
`notebooks/`) and in scripts (which run from `scripts/`).

`pip install -e .` reads [pyproject.toml](pyproject.toml) and installs the package in
**editable** mode: Python is pointed at `src/`, nothing is copied. So:

- `from fake_news... import ...` works from any notebook, script or test, with no `sys.path` hacks.
- Edits to files in `src/` take effect immediately; you never need to reinstall.
- Run it once per Python environment. If you create a new venv or switch the notebook kernel to
  another Python, run it again there.

**Notebooks cache imported modules.** After pulling or editing code in `src/`, restart the kernel,
or put this at the top of your notebook so changes are picked up automatically:

```python
%load_ext autoreload
%autoreload 2
```

Check the install: `python3 -c "import fake_news; print(fake_news.__file__)"` should print a path inside `src/`.

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

The WELFake paper breaks all three of the first rules (vectorizer fit on all data, tuned on the
test set, split ratio chosen on test accuracy). Avoiding that is part of our contribution.

---

## 3. Data

| File | In git? | Made by | Contents |
|---|---|---|---|
| `data/raw/WELFake_Dataset.csv` | no | `download_dataset()` | Original 72,134 rows |
| `data/interim/welfake_clean.csv` | no | `build_clean_dataset()` | 63,018 cleaned articles (markers removed) |
| `data/interim/welfake_clean_with_markers.csv` | no | `build_clean_dataset()` | Same rows, source markers kept |
| `data/processed/splits.csv` | **yes** | Person 1 | Shared splits |
| `data/processed/clean_fingerprint.json` | **yes** | Person 1 | Row counts + hashes of the cleaned data (see below) |
| `data/processed/predictions/<model>__<regime>.csv` | no | each model | `doc_id, split, prob_fake` |

### Is my data identical to everyone else's?

The CSVs are generated locally, so the project checks that everyone ends up with byte-identical data:

1. **Raw file**: the Kaggle version is pinned (`config.KAGGLE_VERSION = 77`) and the download's
   SHA-256 must equal `config.RAW_SHA256`. A newer upload on Kaggle can't silently change our data.
2. **Cleaned data**: `clean_fingerprint.json` (committed) stores the row count, the number of fakes and
   hashes of `doc_id`+`is_fake`, `title`+`text` and `content` for both cleaned variants. Building
   and every `load_clean()` / `load_dataset()` compare against it. A different library version that changes
   even one character raises `DataMismatchError`, which says what differs and how to fix it.
3. **Splits**: `splits.csv` must contain exactly the cleaned `doc_id`s.

`python3 scripts/prepare_data.py` ends with `All checks passed` when all three hold, and
`python3 -m pytest` re-runs them in `tests/test_data_integrity.py` (skipped if the data isn't generated yet).

### Memory

- **Building** the cleaned data streams the raw file in chunks of `config.CLEAN_CHUNK_ROWS`
  (5,000) rows and finds duplicates through 20-byte content hashes. Peak memory is about
  **0.3 GB** whatever the dataset size (the first in-memory version needed 4.4 GB and was
  killed on an 8 GB machine). `prepare_data.py` prints its peak memory at the end.
- Each cleaned CSV is first written to a `.tmp` file and only moved into place once it matches
  the fingerprint, so an interrupted build never leaves a half-written CSV behind.
- **Loading** the full cleaned table in a notebook takes roughly 1–1.5 GB. If you only need
  some columns, load just those: `load_dataset("random", columns=["content"])` (about half).
  `load_labels()` (just `doc_id, is_fake`) needs a few MB.

If the cleaning rules are changed **on purpose**, Person 1 runs
`python3 scripts/prepare_data.py --rebuild --update-fingerprint`, regenerates the splits if the rows
changed, and commits the new JSON. Everyone else then pulls and runs `prepare_data.py --rebuild`.

### What cleaning does

Implemented in [src/fake_news/data/clean.py](src/fake_news/data/clean.py). The principle: **remove
what identifies the outlet or the scraping process, keep what the author wrote.**

WELFake merges four source datasets, and some text reveals the source, and with it the label,
almost perfectly. Left in, a classifier learns *who published the article*, not what it says:

| Marker (examples) | Real | Fake |
|---|---|---|
| `WASHINGTON (Reuters) -` dateline | 21,174 | 0 |
| Title suffix `- The New York Times`, `- Breitbart`, ... | 8,561 | 606 |
| Title tags `(VIDEO)`, `[TWEETS]` | 4 | 6,778 |
| `Featured image via ...`, `Getty Images` credits | 0 | ~8,000 |
| URLs, `pic.twitter.com`, `bit.ly` | 262 | 7,454 |
| 21st Century Wire boilerplate | 0 | 1,556 |
| Agency credits `(Reporting by ...; Editing by ...)`, social bylines, emails, @handles | mostly real | mixed |

Steps:
1. **Normalize**: HTML entities, unicode, curly quotes, glued sentences (`Twitter.Featured` -> `Twitter. Featured`), whitespace.
2. **Remove markers**: rules in `TEXT_MARKERS` / `TITLE_MARKERS` (counts before vs after: `count_markers()`).
3. **Build `content`**: title + text, lower-cased, apostrophes -> spaces (one source wrote `It s` instead of `It's`; this makes them identical). **No stemming and no stop-word removal**, because sentence embeddings need natural text and stop words matter for MinHash shingles. Do model-specific preprocessing inside your own model (e.g. `TfidfVectorizer(stop_words="english")`).
4. **Drop empty texts** (867; all 782 originally empty ones are fake, so keeping them would leak the label).
5. **Drop exact duplicates** of `content` (8,249; keeps the lowest `doc_id`) and any content that appears with both labels (none).

The class balance after cleaning is 44.8% fake (raw: 51.4%); most dropped rows were fake.

---

## 4. Using the shared code

### Load data with a split

```python
from fake_news.data.load import load_dataset

df = load_dataset("random")          # doc_id, title, text, content, n_words, is_fake, split
train = df[df.split == "train"]
val   = df[df.split == "val"]

df = load_dataset("random", columns=["content"])   # less memory: doc_id, content, is_fake, split
```

`load_dataset(regime)` takes `"random"` now; `"grouped"` (near-duplicate groups kept together)
and `"cluster"` (whole K-means clusters held out) will appear once those columns are added to `splits.csv`.

### Other loaders (`fake_news.data.load`)

| Function | Returns |
|---|---|
| `load_clean(remove_markers=True, columns=None)` | Cleaned articles without split. `remove_markers=False` gives the same rows with markers kept, only for measuring the marker effect. `columns=[...]` loads only those columns (plus `doc_id`, `is_fake`). |
| `load_labels()` | Just `doc_id, is_fake` (fast, a few MB). |
| `load_splits()` | The raw `splits.csv` table. |
| `load_raw()` | Original rows as `doc_id, title, text, label`. |
| `download_dataset()` | Downloads the CSV into `data/raw/` if missing. |

### Evaluate

```python
from fake_news.evaluation.metrics import evaluate

scores = evaluate(val.is_fake, prob_fake)   # prob_fake = model's P(fake) for each val row
# {'n', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'tn', 'fp', 'fn', 'tp'}
```

### Save predictions for blending

```python
import pandas as pd
from fake_news.evaluation.predictions import save_predictions, load_predictions, evaluate_predictions

preds = pd.DataFrame({"doc_id": val.doc_id, "split": "val", "prob_fake": prob_fake})
save_predictions(preds, model="model_b", regime="random")   # -> data/processed/predictions/model_b__random.csv

evaluate_predictions(load_predictions("model_b", "random"), split="val")
```

`save_predictions` checks the format (columns, no duplicate `doc_id`, probabilities in [0, 1]),
so Model D can join any two files on `doc_id` without surprises.

### Add a split regime (MinHash pair / Person 2 -> Person 1)

```python
from fake_news.data.load import load_clean
from fake_news.data.split import make_group_split, check_split, save_split_column

df = load_clean()
groups = ...                      # pd.Series: doc_id -> near-duplicate group id (or cluster id)
split = make_group_split(df, groups)                       # whole groups stay in one partition
print(check_split(split, labels=df.set_index("doc_id").is_fake, groups=groups))  # raises if a group leaks
save_split_column("grouped", split)                         # adds split_grouped to splits.csv
```

For a few large K-means clusters use `make_group_split(df, clusters, stratify=False)`.
`save_split_column` refuses to overwrite an existing column unless `overwrite=True`.

### Paths and constants

```python
from fake_news import config
config.SEED, config.SPLIT_FRACTIONS, config.CLEAN_CSV, config.SPLITS_CSV, config.PREDICTIONS_DIR
```

---

## 5. Tests

```bash
python3 -m pytest
```

---

## Project structure

```
├── configs/            Experiment / model configuration files
├── data/
│   ├── raw/            Original WELFake CSV (not in git)
│   ├── external/       Third-party data
│   ├── interim/        Cleaned data (not in git)
│   └── processed/      splits.csv (in git), predictions/ (not in git)
├── models/             Trained and serialized models
├── notebooks/          00-how-to-use.ipynb and exploration notebooks
├── reports/
│   └── figures/        Generated plots for the report
├── scripts/            prepare_data.py and other command-line entry points
├── src/fake_news/
│   ├── config.py       Paths, seed, split fractions
│   ├── data/           load.py (loading), clean.py (cleaning), split.py (splits)
│   ├── features/       Text preprocessing and feature extraction
│   ├── models/         Training and prediction
│   └── evaluation/     metrics.py (evaluate), predictions.py (shared prediction files)
├── tests/              Unit tests
├── pyproject.toml      Package definition for `pip install -e .`
└── requirements.txt
```
