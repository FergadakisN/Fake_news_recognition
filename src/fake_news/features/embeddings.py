"""Sentence embeddings (Sentence-BERT) for the cleaned WELFake articles.

The encoder is pretrained and never fitted on our data, so embeddings are computed once for *all* cleaned
articles, before and independently of any split, and cached by ``doc_id``. Everything fitted *on* the
embeddings (PCA, K-means, classifiers) must still be fitted on ``train`` only.

Typical use::

    from fake_news.features.embeddings import embeddings_for

    df = load_dataset("random", columns=["content"])
    X = embeddings_for(df.doc_id)            # (n, 384) float32, row i belongs to df.doc_id.iloc[i]

The main model (frozen embeddings + logistic regression) is at the end of this module: ``run_classifier``,
run by ``python3 scripts/run_embeddings_lr.py --regime grouped|random``.

Build the cache with ``python scripts/compute_embeddings.py`` (GPU recommended). Vectors are L2-normalized,
so cosine similarity is a dot product: ``X[i] @ X[j]``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fake_news import config

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KEY = "minilm"
MAX_SEQ_LENGTH = 256  # word pieces; the model truncates anything longer
DIM = 384
# Vectors are stored as float16 on disk (~48 MB instead of ~96 MB, so they can be committed to git)
# and returned as float32. The rounding changes cosine similarities by about 0.001.
STORE_DTYPE = np.float16


class StaleEmbeddingsError(RuntimeError):
    """The cached embeddings were built from a different version of the cleaned data."""


# ---------------------------------------------------------------------------------------------------------
# Paths and fingerprints
# ---------------------------------------------------------------------------------------------------------

def embeddings_dir() -> Path:
    return Path(getattr(config, "EMBEDDINGS_DIR", Path(config.SPLITS_CSV).parent / "embeddings"))


def _fingerprint_file() -> Path:
    return Path(getattr(config, "CLEAN_FINGERPRINT", Path(config.SPLITS_CSV).parent / "clean_fingerprint.json"))


def data_fingerprint() -> str:
    "Short hash of clean_fingerprint.json: changes whenever Person 1 changes the cleaned data."
    f = _fingerprint_file()
    if not f.exists():
        return "missing"
    content = f.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()[:16]


def _matches_data_fingerprint(fingerprint: str) -> bool:
    "Accept current fingerprints and legacy hashes made before normalizing line endings."
    f = _fingerprint_file()
    if not f.exists():
        return fingerprint == "missing"
    content = f.read_bytes()
    normalized = content.replace(b"\r\n", b"\n")
    candidates = {
        hashlib.sha256(normalized).hexdigest()[:16],
        hashlib.sha256(content).hexdigest()[:16],
        hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()[:16],
    }
    return fingerprint in candidates


def _ids_hash(doc_ids) -> str:
    return hashlib.sha256(np.asarray(doc_ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def artifact_paths(sample: int | None = None) -> dict[str, Path]:
    "The files of the cache (a sample run is stored separately)."
    name = f"{MODEL_KEY}__clean" + (f"__sample{sample}" if sample else "")
    d = embeddings_dir()
    return {
        "vectors": d / f"{name}.npy",
        "doc_ids": d / f"{name}__doc_ids.npy",
        "meta": d / f"{name}.json",
        "parts": d / "parts" / name,
    }


def _save_npy(path: Path, arr: np.ndarray) -> None:
    "Write atomically, so an interrupted run never leaves a half-written file."
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _save_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------
# Encoder and input
# ---------------------------------------------------------------------------------------------------------

def get_device(preferred: str | None = None) -> str:
    "cuda > mps (Apple Silicon) > cpu, unless a device is given."
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(device: str | None = None):
    "The pretrained Sentence-BERT model (downloads ~90 MB the first time)."
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_NAME, device=get_device(device))
    encoder.max_seq_length = MAX_SEQ_LENGTH
    return encoder


def embedding_texts(df: pd.DataFrame) -> list[str]:
    "The text that is embedded: the shared ``content`` column (cleaned title + text, lower-cased)."
    return df["content"].fillna("").astype(str).tolist()


def token_lengths(tokenizer, texts: list[str], batch: int = 512) -> np.ndarray:
    "Number of word pieces per text (with [CLS]/[SEP]), *before* truncation."
    out: list[int] = []
    for i in range(0, len(texts), batch):
        ids = tokenizer(texts[i:i + batch], add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        out.extend(len(x) for x in ids)
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------
# Building the cache
# ---------------------------------------------------------------------------------------------------------

def encode_chunked(encoder, texts: list[str], parts_dir: Path, key: str, batch_size: int = 64,
                   chunk_size: int = 5000, log: Callable[[str], None] = print) -> np.ndarray:
    """Encode in chunks saved to ``parts_dir``; a rerun skips finished chunks.

    ``key`` identifies the inputs: if it changed (new cleaning, other sample), old chunks are discarded.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest = parts_dir / "manifest.json"
    want = {"key": key, "n": len(texts), "chunk_size": chunk_size}
    if manifest.exists() and json.loads(manifest.read_text()) != want:
        log("  inputs changed since the interrupted run: discarding old chunks")
        shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True)
    manifest.write_text(json.dumps(want))

    n_chunks = max(1, (len(texts) + chunk_size - 1) // chunk_size)
    for c in range(n_chunks):
        f = parts_dir / f"part_{c:04d}.npy"
        if f.exists():
            log(f"  chunk {c + 1}/{n_chunks} already done, skipping")
            continue
        s, e = c * chunk_size, min((c + 1) * chunk_size, len(texts))
        t0 = time.time()
        emb = encoder.encode(texts[s:e], batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        _save_npy(f, np.asarray(emb, dtype=np.float32))
        log(f"  chunk {c + 1}/{n_chunks}: {e - s:,} articles in {time.time() - t0:.1f}s")
    return np.concatenate([np.load(parts_dir / f"part_{c:04d}.npy") for c in range(n_chunks)])


def _versions() -> dict[str, str]:
    out = {}
    for mod in ("sentence_transformers", "transformers", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # not installed
            pass
    return out


def build_embeddings(sample: int | None = None, *, batch_size: int | None = None, chunk_size: int = 5000,
                     device: str | None = None, rebuild: bool = False, encoder=None,
                     log: Callable[[str], None] = print) -> dict:
    """Compute and cache the embeddings of all cleaned articles; returns the metadata.

    Does nothing if an up-to-date cache exists (unless ``rebuild``). ``sample`` embeds a random subset
    (seed ``config.SEED``), stored separately: for timing runs only.
    """
    from fake_news.data import load_clean

    paths = artifact_paths(sample)
    if not rebuild and paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text())
        if _matches_data_fingerprint(meta.get("data_fingerprint", "")) and paths["vectors"].exists():
            log(f"{paths['vectors'].name} is up to date ({meta['n_docs']:,} articles); use rebuild=True to redo")
            return meta
        log("cleaned data changed since the cache was built: rebuilding")

    df = load_clean(columns=["content"]).sort_values("doc_id")
    n_total = len(df)
    if sample:
        df = df.sample(n=min(sample, n_total), random_state=config.SEED).sort_values("doc_id")
    texts, doc_ids = embedding_texts(df), df["doc_id"].to_numpy(dtype=np.int64)

    encoder = encoder if encoder is not None else load_encoder(device)
    dev = str(getattr(encoder, "device", device or "cpu"))
    batch_size = batch_size or (128 if "cuda" in dev else 64)
    log(f"Embedding {len(texts):,} articles with {MODEL_NAME} on {dev}, batch size {batch_size}")

    key = "|".join([MODEL_NAME, str(MAX_SEQ_LENGTH), data_fingerprint(), _ids_hash(doc_ids)])
    t0 = time.time()
    E = encode_chunked(encoder, texts, paths["parts"], key, batch_size, chunk_size, log)
    seconds = time.time() - t0

    if E.shape != (len(texts), DIM):
        raise RuntimeError(f"expected shape {(len(texts), DIM)}, got {E.shape}")
    paths["vectors"].parent.mkdir(parents=True, exist_ok=True)
    _save_npy(paths["vectors"], E.astype(STORE_DTYPE))
    _save_npy(paths["doc_ids"], doc_ids)
    meta = {
        "model": MODEL_NAME, "max_seq_length": MAX_SEQ_LENGTH, "dim": DIM, "normalized": True,
        "stored_as": np.dtype(STORE_DTYPE).name, "input": "content (load_clean)",
        "sample": sample, "n_docs": int(len(texts)), "n_clean_total": int(n_total),
        "n_empty_inputs": int(sum(1 for t in texts if not t)),
        "data_fingerprint": data_fingerprint(), "doc_ids_hash": _ids_hash(doc_ids),
        "device": dev, "batch_size": batch_size, "seconds": round(seconds, 1),
        "docs_per_second": round(len(texts) / max(seconds, 1e-9), 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": _versions(),
    }
    _save_text(paths["meta"], json.dumps(meta, indent=2))
    shutil.rmtree(paths["parts"], ignore_errors=True)
    log(f"Saved {paths['vectors'].name}: {E.shape} in {seconds:.0f}s ({meta['docs_per_second']:.0f} articles/s)")
    return meta


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------

def load_meta(sample: int | None = None) -> dict:
    p = artifact_paths(sample)["meta"]
    if not p.exists():
        raise FileNotFoundError(f"No embeddings at {p}. Build them with: python scripts/compute_embeddings.py"
                                + (f" --sample {sample}" if sample else ""))
    return json.loads(p.read_text())


def load_embeddings(sample: int | None = None, check: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """All cached vectors (float32) and their doc_ids (sorted). Raises StaleEmbeddingsError if the cleaned
    data changed since they were built (``check=False`` skips this)."""
    meta, paths = load_meta(sample), artifact_paths(sample)
    if check and not _matches_data_fingerprint(meta["data_fingerprint"]):
        raise StaleEmbeddingsError(
            f"{paths['vectors'].name} was built from another version of the cleaned data "
            f"({meta['data_fingerprint']} vs {data_fingerprint()}). Rebuild: python scripts/compute_embeddings.py")
    return np.load(paths["vectors"]).astype(np.float32), np.load(paths["doc_ids"])


def embeddings_for(doc_ids, sample: int | None = None) -> np.ndarray:
    "Vectors aligned to ``doc_ids`` (row i belongs to doc_ids[i]). KeyError if any doc_id has none."
    E, ids = load_embeddings(sample)
    want = np.asarray(doc_ids, dtype=np.int64)
    pos = pd.Index(ids).get_indexer(want)
    if (pos < 0).any():
        missing = want[pos < 0]
        raise KeyError(f"{len(missing):,} doc_ids have no embedding (e.g. {missing[:5].tolist()}); "
                       "are they cleaned articles? Is the cache built from a sample?")
    return E[pos]


# ---------------------------------------------------------------------------------------------------------
# Checks (used by scripts/compute_embeddings.py and the tests)
# ---------------------------------------------------------------------------------------------------------

def check_embeddings(sample: int | None = None) -> list[str]:
    "Raises AssertionError on the first failed check; returns the list of passed checks."
    from fake_news.data import load_clean

    meta = load_meta(sample)
    E, ids = load_embeddings(sample)  # raises if stale
    passed = ["cache matches the current cleaned data"]
    assert E.shape == (meta["n_docs"], DIM) and len(ids) == len(E), f"shape {E.shape}, {len(ids)} doc_ids"
    passed.append(f"shape {E.shape}, one doc_id per row")
    assert np.isfinite(E).all(), "non-finite values"
    passed.append("all values finite")
    norms = np.linalg.norm(E, axis=1)
    assert np.allclose(norms, 1, atol=5e-3), f"norms in [{norms.min():.4f}, {norms.max():.4f}]"
    passed.append("every vector has unit length (up to float16 rounding)")
    assert (np.diff(ids) > 0).all(), "doc_ids not unique and sorted"
    passed.append("doc_ids unique and sorted")
    if not sample:
        clean_ids = np.sort(load_clean(columns=[])["doc_id"].to_numpy())
        assert np.array_equal(ids, clean_ids), "doc_ids differ from load_clean()"
        passed.append(f"exactly the {len(clean_ids):,} cleaned articles")
    return passed

# ---------------------------------------------------------------------------------------------------------
# Main model: frozen embeddings + logistic regression
# ---------------------------------------------------------------------------------------------------------
# Protocol, as for every model in the project: fit on ``train`` only, choose C on ``val`` (ROC-AUC, the
# same criterion as Model C), predict ``val`` and ``test``. ``test`` is predicted but never scored here.
# Run it with ``python3 scripts/run_embeddings_lr.py --regime grouped`` (and ``--regime random``).

LR_MODEL = "embeddings_lr"
LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 1000.0)
LR_SELECT_BY = "roc_auc"
LR_REPORT_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def lr_report_dir() -> Path:
    "reports/embeddings_lr/: validation scores and chosen C per regime."
    return Path(getattr(config, "REPORTS_DIR", Path(config.FIGURES_DIR).parent)) / LR_MODEL


def make_classifier(C: float):
    """Standardize the 384 dimensions, then L2-regularized logistic regression.

    The vectors are L2-normalized, so each dimension is tiny (~0.05); standardizing puts them on a common
    scale, which makes the C grid meaningful. The scaler is part of the model: fitted on train only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=config.SEED))


LR_TOL = 0.001  # val ROC-AUC differences smaller than this are treated as noise


def tune_classifier(X_tr, y_tr, X_va, y_va, Cs=LR_C_GRID, log: Callable[[str], None] = print):
    """Fit one model per C on train and score each on val. Returns (model, C, table of val scores).

    Picks the smallest C (strongest regularization) whose val ROC-AUC is within LR_TOL of the best:
    the simplest model that performs as well as the best one.
    """
    from fake_news.evaluation import evaluate

    rows, models = [], {}
    for C in Cs:
        t0 = time.time()
        models[C] = make_classifier(C).fit(X_tr, y_tr)
        s = evaluate(y_va, models[C].predict_proba(X_va)[:, 1])
        rows.append({"C": C, **{k: float(s[k]) for k in LR_REPORT_KEYS}, "seconds": round(time.time() - t0, 1)})
        log(f"  C={C:<7g} val ROC-AUC {s['roc_auc']:.4f}  F1 {s['f1']:.4f}  ({time.time() - t0:.1f}s)")
    table = pd.DataFrame(rows).set_index("C")
    best = table[LR_SELECT_BY].max()
    best_C = min(c for c in Cs if table.loc[c, LR_SELECT_BY] >= best - LR_TOL)
    return models[best_C], best_C, table


def _load_regime(regime: str) -> pd.DataFrame:
    from fake_news.data import load_dataset

    try:
        df = load_dataset(regime, columns=[])          # doc_id, is_fake, split
    except Exception:
        df = load_dataset(regime, columns=["content"])
    return df[["doc_id", "is_fake", "split"]].reset_index(drop=True)


def run_classifier(regime: str = "grouped", Cs=LR_C_GRID, log: Callable[[str], None] = print) -> dict:
    """Train on train, tune C on val, save val + test predictions and the tuning report. Returns the report.

    Writes ``data/processed/predictions/embeddings_lr__<regime>.csv`` (``doc_id, split, prob_fake``) and
    ``reports/embeddings_lr/<regime>.json``.
    """
    from fake_news.evaluation import save_predictions

    df = _load_regime(regime)
    X = embeddings_for(df.doc_id)
    y = df.is_fake.to_numpy()
    m = {s: (df.split == s).to_numpy() for s in ("train", "val", "test")}
    log(f"{LR_MODEL} on the {regime} split: train {m['train'].sum():,}, val {m['val'].sum():,}, "
        f"test {m['test'].sum():,} (test predicted, not scored)")

    model, best_C, table = tune_classifier(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]], Cs, log)

    keep = m["val"] | m["test"]
    preds = pd.DataFrame({"doc_id": df.doc_id[keep].to_numpy(), "split": df.split[keep].to_numpy(),
                          "prob_fake": model.predict_proba(X[keep])[:, 1]})
    path = save_predictions(preds, model=LR_MODEL, regime=regime)

    meta = load_meta()
    report = {
        "model": LR_MODEL, "regime": regime,
        "encoder": meta["model"], "encoder_frozen": True, "embedding_dim": meta["dim"],
        "classifier": "StandardScaler + LogisticRegression (L2, lbfgs)",
        "selected_by": f"val {LR_SELECT_BY}", "C": best_C,
        "val": table.loc[best_C, list(LR_REPORT_KEYS)].to_dict(),
        "val_by_C": table.reset_index().to_dict(orient="records"),
        "n": {s: int(v.sum()) for s, v in m.items()},
        "data_fingerprint": meta["data_fingerprint"], "seed": config.SEED,
    }
    out = lr_report_dir() / f"{regime}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_text(out, json.dumps(report, indent=2))
    log(f"Best C = {best_C}: val ROC-AUC {report['val']['roc_auc']:.4f}, F1 {report['val']['f1']:.4f}")
    if best_C in (min(Cs), max(Cs)):
        log(f"  WARNING: best C is at the edge of the grid {tuple(Cs)}; consider extending LR_C_GRID")
    log(f"Saved {path} and {out}")
    return report