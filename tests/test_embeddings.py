"""Unit tests for fake_news.features.embeddings: the embedding cache (with a fake encoder: no model download,
no GPU) and the logistic-regression classifier (on synthetic data)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import fake_news.data
from fake_news import config
from fake_news.features import embeddings as emb


class FakeEncoder:
    "Bag-of-words hashed into 384 dims: same words -> same vector. Counts encode calls."
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        out = np.zeros((len(texts), emb.DIM), np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                out[i, hash(w) % emb.DIM] += 1
            out[i, 0] += 1e-3  # never all-zero
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    fp = tmp_path / "clean_fingerprint.json"
    fp.write_text('{"rows": 6}')
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path / "embeddings", raising=False)
    monkeypatch.setattr(config, "CLEAN_FINGERPRINT", fp, raising=False)
    docs = pd.DataFrame({
        "doc_id": [9, 2, 5, 7, 11, 3],
        "content": ["senate passes tax bill", "senate passes the tax bill", "storm hits florida coast",
                    "celebrity wedding photos", "storm hits the florida coast", "election results tonight"],
        "is_fake": [0, 0, 1, 1, 1, 0],
    })

    def load_clean(remove_markers=True, columns=None):
        return docs[["doc_id", *(columns or []), "is_fake"]].copy() if columns is not None else docs.copy()

    monkeypatch.setattr(fake_news.data, "load_clean", load_clean)
    return {"fp": fp}


def test_build_load_and_align(env):
    meta = emb.build_embeddings(encoder=FakeEncoder(), chunk_size=4, log=lambda *_: None)
    assert meta["n_docs"] == 6
    E, ids = emb.load_embeddings()
    assert ids.tolist() == [2, 3, 5, 7, 9, 11] and E.shape == (6, emb.DIM)
    X = emb.embeddings_for([11, 2])
    assert X.dtype == np.float32 and np.allclose(X[0], E[-1]) and np.allclose(X[1], E[0])
    with pytest.raises(KeyError):
        emb.embeddings_for([2, 999])
    assert any("unit length" in c for c in emb.check_embeddings())


def test_cache_is_reused_and_detects_stale_data(env):
    enc = FakeEncoder()
    emb.build_embeddings(encoder=enc, log=lambda *_: None)
    emb.build_embeddings(encoder=enc, log=lambda *_: None)
    assert enc.calls == 1  # second call used the cache
    env["fp"].write_text('{"rows": 7}')  # Person 1 changed the cleaning
    with pytest.raises(emb.StaleEmbeddingsError):
        emb.load_embeddings()
    emb.build_embeddings(encoder=enc, log=lambda *_: None)
    assert enc.calls == 2


def test_fingerprint_ignores_line_endings_and_accepts_legacy_hash(env):
    fp = env["fp"]
    lf_content = fp.read_bytes()
    lf_fingerprint = emb.data_fingerprint()
    crlf_content = lf_content.replace(b"\n", b"\r\n")
    fp.write_bytes(crlf_content)
    assert emb.data_fingerprint() == lf_fingerprint

    enc = FakeEncoder()
    meta = emb.build_embeddings(encoder=enc, log=lambda *_: None)
    meta["data_fingerprint"] = hashlib.sha256(crlf_content).hexdigest()[:16]
    emb.artifact_paths()["meta"].write_text(json.dumps(meta))
    fp.write_bytes(lf_content)

    emb.build_embeddings(encoder=enc, log=lambda *_: None)
    emb.load_embeddings()
    assert enc.calls == 1


def test_sample_is_stored_separately(env):
    emb.build_embeddings(sample=3, encoder=FakeEncoder(), log=lambda *_: None)
    E, ids = emb.load_embeddings(sample=3)
    assert len(ids) == 3 and E.shape == (3, emb.DIM)
    with pytest.raises(FileNotFoundError):
        emb.load_embeddings()  # the full cache does not exist yet


def test_resume_skips_finished_chunks(env, tmp_path):
    enc = FakeEncoder()
    texts = [f"text {i}" for i in range(10)]
    parts = tmp_path / "parts"
    full = emb.encode_chunked(enc, texts, parts, "k", chunk_size=3, log=lambda *_: None)
    (parts / "part_0002.npy").unlink()  # as if interrupted during chunk 3
    enc.calls = 0
    again = emb.encode_chunked(enc, texts, parts, "k", chunk_size=3, log=lambda *_: None)
    assert enc.calls == 1 and np.allclose(full, again)
    enc.calls = 0
    emb.encode_chunked(enc, texts, parts, "other inputs", chunk_size=3, log=lambda *_: None)
    assert enc.calls == 4  # changed key: everything redone