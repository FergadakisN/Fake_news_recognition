# Fake News Recognition

Classifying news articles as reliable or fake.

## Project structure

```
├── configs/            Experiment / model configuration files
├── data/
│   ├── raw/            Original, immutable datasets
│   ├── external/       Third-party data
│   ├── interim/        Intermediate, partially processed data
│   └── processed/      Final datasets used for modelling
├── models/             Trained and serialized models
├── notebooks/          Jupyter notebooks for exploration (e.g. 01-eda.ipynb)
├── reports/
│   └── figures/        Generated plots for the report
├── scripts/            Command-line entry points (download, train, evaluate)
├── src/fake_news/
│   ├── data/           Data loading and cleaning
│   ├── features/       Text preprocessing and feature extraction
│   ├── models/         Training and prediction
│   └── evaluation/     Metrics and evaluation
├── tests/              Unit tests
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
