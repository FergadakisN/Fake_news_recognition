"""Clean WELFake articles: normalize, remove source markers, drop empty and duplicate rows.

Principle: remove what identifies the *outlet* or the *scraping process*; keep what the
author wrote. WELFake merges several source datasets, and some markers reveal the label
almost perfectly (e.g. "(Reuters)" datelines appear only in real news, "Featured image
via ..." only in fake news). Left in, a classifier learns the source, not the content.

Steps (see :func:`clean_dataset`):

1. Normalize title and text: HTML entities, unicode, curly quotes, glued sentences
   ("Twitter.Featured" -> "Twitter. Featured"), whitespace.
2. Remove source/scraping markers (:data:`TEXT_MARKERS`, :data:`TITLE_MARKERS`).
3. Build ``content`` = title + text, lower-cased, apostrophes -> spaces (so the scraped
   "It s" and the normal "It's" look the same). No stemming or stop-word removal:
   sentence embeddings need natural text and stop words matter for MinHash shingles.
4. Drop rows whose text is empty after cleaning (all 782 empty texts in WELFake are fake).
5. Drop exact duplicates of ``content`` (keep the lowest ``doc_id``) and drop all copies
   of any content that appears with both labels.

Row selection always uses the marker-free content, so ``remove_markers=True`` and
``False`` return exactly the same ``doc_id`` set.

Memory: :func:`build_clean_dataset` streams the raw CSV in chunks of
``config.CLEAN_CHUNK_ROWS`` rows. Duplicates are found with 20-byte content hashes, so only
one chunk of text is in memory at any time and peak memory does not grow with the dataset.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from fake_news import config

# Outlet names that appear as title suffixes ("... - The New York Times", "... | RT").
TITLE_SOURCES = [
    "The New York Times", "Breitbart", "America's Finest News Source", "TruthFeed",
    "Russia News Now", "New Eastern Outlook", "Новое восточное обозрение",
    "EndingFed News Network", "Daily Wire", "RedFlag News", "Guardian Liberty Voice",
    "The Vineyard of the Saker", "RT Arabic", "Collective Evolution", "Russia & India Report",
    "Katehon think tank. Geopolitics & Tradition", "Conservative Daily Post", "RT",
    "GomerBlog", "The Federalist Papers", "New Century Times", "USAPoliticsNow",
    "Morning News USA", "twitchy.com",
]

# (name, regex). Applied in this order to the normalized text.
TEXT_MARKERS: list[tuple[str, re.Pattern]] = [
    ("reuters_dateline", re.compile(r"^\s*.{0,120}?\(Reuters\)\s*-+\s*")),
    ("reuters_tag", re.compile(r"\(Reuters\)")),
    ("agency_credits", re.compile(
        r"\((?:additional )?(?:reporting|writing|editing|compiled)\s+by\s[^)]{0,250}\)"
        r"|this article was funded in part by.{0,250}?creation or production\."
        r"|(?:^|(?<=[.!?] ))[^.]{0,80}?\bcontributed to this report\.", re.I)),  # anchored: fast
    ("21wire_boilerplate", re.compile(
        r"(?:support our work by )?subscrib(?:e|ing)\s*&\s*becom(?:e|ing) a member"
        r"|tune in to the alternate current radio network[^.]{0,200}\.?"
        r"|\b21st century wire\b|@?\s?\b21wire(?:\.tv)?\b", re.I)),
    ("scraper_boilerplate", re.compile(
        r"get short url|share on facebook(?: share on twitter)?|next prev swipe left/right"
        r"|you are here: home|\(?want to get this briefing by email\?[^)]{0,60}\)?"
        r"|for entire article click link|\bwe are change\b", re.I)),
    ("social_byline", re.compile(r"\b(?:follow|like) (?:\w+ ){0,3}on (?:twitter|facebook)\b(?: at)?[!.]?", re.I)),
    ("url", re.compile(r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+(?:com|org|net|ly|co|tv|me)/\S*", re.I)),
    # also catches the scraped "name@nytimes. com" form
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.\s?(?:com|org|net|edu|gov|uk|tv|us|io|info)\b", re.I)),
    ("twitter_handle", re.compile(r"(?<!\w)@\w+")),
    ("image_credit", re.compile(
        r"(?:featured\s+)?(?:image|photo|picture)s?\s+(?:via|credit|courtesy)\b.{0,120}$", re.I)),
    ("featured_image", re.compile(r"\bfeatured image\b[^.]{0,120}\.?", re.I)),
    ("photo_by_getty", re.compile(r"photo(?:graph)?\s+by\s+[^.]{0,60}?/\s*getty images\.?", re.I)),
    ("getty_images", re.compile(r"\bgetty images\b", re.I)),
]

TITLE_MARKERS: list[tuple[str, re.Pattern]] = [
    # all-caps tags in brackets: "(VIDEO)", "[TWEETS]", "(VIDEO/TWEETS)"
    ("title_tag", re.compile(r"\s*[\[(][A-Z0-9][A-Z0-9 /&,!'+-]*[\])]")),
    ("title_source_suffix", re.compile(
        r"\s+[-|·—–]\s+(?:" + "|".join(map(re.escape, TITLE_SOURCES)) + r")\s*$")),
]

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
                         "–": "-", "—": "-", "‐": "-", "\xa0": " "})
_GLUED_SENTENCE = re.compile(r"(?<=[a-z])([.!?])(?=[A-Z][a-z])")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Fix encoding and formatting noise; keeps case and wording."""
    s = unicodedata.normalize("NFKC", html.unescape(s)).translate(_QUOTES)
    s = _GLUED_SENTENCE.sub(r"\1 ", s)
    return _WS.sub(" ", s).strip()


def remove_text_markers(s: str) -> str:
    """Remove source/scraping markers from a normalized article text."""
    for _, pattern in TEXT_MARKERS:
        s = pattern.sub(" ", s)
    return _WS.sub(" ", s).strip()


def remove_title_markers(s: str) -> str:
    """Remove bracket tags and outlet-name suffixes from a normalized title."""
    for _, pattern in TITLE_MARKERS:
        s = pattern.sub("", s)
    return _WS.sub(" ", s).strip()


def make_content(title: str, text: str) -> str:
    """Model input: title + text, lower-cased, apostrophes replaced by spaces."""
    if title and title[-1] not in ".!?":
        title += "."
    s = f"{title} {text}".lower().replace("'", " ")
    return _WS.sub(" ", s).strip()


def _clean_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize + remove markers for a chunk of raw rows (``doc_id, title, text, label``).

    Returns ``doc_id, is_fake`` plus normalized (``title_n``, ``text_n``) and marker-free
    (``title_c``, ``text_c``, ``content_c``) text.
    """
    out = pd.DataFrame({
        "doc_id": raw["doc_id"].to_numpy(),
        "is_fake": raw["label"].astype(int).to_numpy(),
        "title_n": raw["title"].fillna("").astype(str).map(normalize).to_numpy(),
        "text_n": raw["text"].fillna("").astype(str).map(normalize).to_numpy(),
    })
    out["title_c"] = out["title_n"].map(remove_title_markers)
    out["text_c"] = out["text_n"].map(remove_text_markers)
    out["content_c"] = [make_content(t, x) for t, x in zip(out["title_c"], out["text_c"])]
    return out


class _RowSelector:
    """Chooses the rows to keep, chunk by chunk, remembering only content hashes.

    Rows must arrive in increasing ``doc_id`` order, so "first occurrence" = lowest doc_id.
    :meth:`keep_mask` drops empty texts and repeated contents as they arrive; contents that
    turn out to carry both labels are only known at the end (:meth:`conflicting_doc_ids`).
    """

    def __init__(self) -> None:
        self.first_doc: dict[bytes, int] = {}  # content hash -> doc_id of the kept (first) copy
        self.labels: dict[bytes, int] = {}     # content hash -> bitmask of labels seen (1=real, 2=fake)
        self.count: dict[bytes, int] = {}      # content hash -> number of non-empty rows
        self.raw_rows = 0
        self.empty = 0
        self.last_doc_id: int | None = None

    def keep_mask(self, chunk: pd.DataFrame) -> np.ndarray:
        ids = chunk["doc_id"].to_numpy()
        if len(ids) and ((self.last_doc_id is not None and ids[0] <= self.last_doc_id)
                         or not (np.diff(ids) > 0).all()):
            raise ValueError("Rows must be sorted by increasing doc_id.")
        if len(ids):
            self.last_doc_id = int(ids[-1])

        keep = np.zeros(len(chunk), dtype=bool)
        for i, (doc_id, y, text, content) in enumerate(
                zip(ids, chunk["is_fake"], chunk["text_c"], chunk["content_c"])):
            self.raw_rows += 1
            if text == "":
                self.empty += 1
                continue
            h = hashlib.sha1(content.encode("utf-8")).digest()
            if h in self.first_doc:
                self.count[h] += 1
                self.labels[h] |= 1 << int(y)
            else:
                self.first_doc[h] = int(doc_id)
                self.count[h] = 1
                self.labels[h] = 1 << int(y)
                keep[i] = True
        return keep

    def _conflicts(self) -> list[bytes]:
        return [h for h, mask in self.labels.items() if mask == 0b11]

    def conflicting_doc_ids(self) -> set[int]:
        return {self.first_doc[h] for h in self._conflicts()}

    def report(self) -> dict:
        conflicts = self._conflicts()
        conflict_rows = sum(self.count[h] for h in conflicts)
        duplicates = sum(c - 1 for c in self.count.values()) - sum(self.count[h] - 1 for h in conflicts)
        return {
            "raw_rows": self.raw_rows,
            "dropped_empty_text": self.empty,
            "dropped_conflicting_labels": conflict_rows,
            "dropped_duplicates": duplicates,
            "clean_rows": len(self.first_doc) - len(conflicts),
        }


def _output(chunk: pd.DataFrame, remove_markers: bool) -> pd.DataFrame:
    """Final columns ``doc_id, title, text, content, n_words, is_fake`` for one variant."""
    if remove_markers:
        title, text, content = chunk["title_c"].to_numpy(), chunk["text_c"].to_numpy(), chunk["content_c"].to_numpy()
    else:
        title, text = chunk["title_n"].to_numpy(), chunk["text_n"].to_numpy()
        content = [make_content(t, x) for t, x in zip(title, text)]
    out = pd.DataFrame({"doc_id": chunk["doc_id"].to_numpy(), "title": title, "text": text, "content": content})
    out["n_words"] = [len(c.split()) for c in out["content"]]  # one article at a time: no huge word lists
    out["is_fake"] = chunk["is_fake"].to_numpy()
    return out


def clean_dataset(raw: pd.DataFrame, remove_markers: bool = True) -> pd.DataFrame:
    """Clean a raw frame in memory (``doc_id, title, text, label``); for small frames and tests.

    Same rules and result as :func:`build_clean_dataset`. Returns
    ``doc_id, title, text, content, n_words, is_fake`` sorted by ``doc_id``; a summary of
    removed rows is stored in ``result.attrs["report"]``.
    """
    chunk = _clean_chunk(raw.sort_values("doc_id"))
    selector = _RowSelector()
    chunk = chunk[selector.keep_mask(chunk)]
    chunk = chunk[~chunk["doc_id"].isin(selector.conflicting_doc_ids())]
    out = _output(chunk, remove_markers).reset_index(drop=True)
    out.attrs["report"] = selector.report()
    return out


def count_markers(df: pd.DataFrame, label_col: str = "is_fake") -> pd.DataFrame:
    """How many articles per class match each marker rule (for audits / before-after checks).

    ``df`` needs ``title``, ``text`` and a label column. Works on raw or cleaned frames.
    """
    titles = df["title"].fillna("").astype(str).map(normalize)
    texts = df["text"].fillna("").astype(str).map(normalize)
    rows = {}
    for series, markers in ((titles, TITLE_MARKERS), (texts, TEXT_MARKERS)):
        for name, pattern in markers:
            hit = series.map(lambda s, p=pattern: p.search(s) is not None)
            rows[name] = df.loc[hit, label_col].value_counts()
    out = pd.DataFrame(rows).T.reindex(columns=[0, 1]).fillna(0).astype(int)
    out.columns = ["real", "fake"]
    return out


_OUTPUTS = (("clean", True, "CLEAN_CSV"), ("clean_with_markers", False, "CLEAN_WITH_MARKERS_CSV"))


def build_clean_dataset(update_fingerprint: bool = False) -> dict:
    """Clean the raw CSV in chunks and write both variants to ``data/interim/`` (bounded memory).

    Writes ``welfake_clean.csv`` (markers removed) and ``welfake_clean_with_markers.csv``
    (same rows, markers kept). Each is first written to a ``.tmp`` file and checked against
    ``data/processed/clean_fingerprint.json``:

    - fingerprint matches -> the ``.tmp`` file replaces the old CSV;
    - no fingerprint stored yet, or ``update_fingerprint=True`` -> this build becomes the
      reference (Person 1 commits the JSON);
    - otherwise -> the ``.tmp`` file is deleted and ``DataMismatchError`` is raised.

    So a CSV at the final path is never half-written and never differs from the reference.
    Returns the cleaning report (row counts).
    """
    from fake_news.data.integrity import (
        check_fingerprint_values, fingerprint_file, load_expected, save_fingerprint_values, verify_raw_file,
    )

    verify_raw_file()
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    targets = {variant: getattr(config, attr) for variant, _, attr in _OUTPUTS}
    tmp = {variant: path.with_name(path.name + ".tmp") for variant, path in targets.items()}

    try:
        # Pass 1: stream raw -> clean -> keep first copies -> append to the .tmp files
        selector = _RowSelector()
        first = True
        for raw in pd.read_csv(config.RAW_CSV, chunksize=config.CLEAN_CHUNK_ROWS):
            chunk = _clean_chunk(raw.rename(columns={"Unnamed: 0": "doc_id"}))
            chunk = chunk[selector.keep_mask(chunk)]
            for variant, remove_markers, _ in _OUTPUTS:
                _output(chunk, remove_markers).to_csv(tmp[variant], index=False, header=first,
                                                      mode="w" if first else "a")
            first = False

        report = selector.report()
        for k, v in report.items():
            print(f"  {k:28s} {v:>7,}")

        # Pass 2 (only if needed): remove contents that appeared with both labels
        conflicts = selector.conflicting_doc_ids()
        if conflicts:
            for path in tmp.values():
                _drop_rows_from_csv(path, conflicts)

        # Check each file against the reference, then move it into place
        expected = load_expected()
        for variant, _, _ in _OUTPUTS:
            actual = fingerprint_file(tmp[variant])
            if update_fingerprint or variant not in expected:
                save_fingerprint_values(actual, variant)
                status = f"stored as reference fingerprint in {config.FINGERPRINT_JSON.name}"
            else:
                check_fingerprint_values(actual, variant)  # raises DataMismatchError
                status = "matches the reference fingerprint"
            os.replace(tmp[variant], targets[variant])
            print(f"Wrote {targets[variant].name}; {status}")
    finally:
        # On any error, never leave temporary files behind (the final CSVs are untouched)
        for path in tmp.values():
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".filtered").unlink(missing_ok=True)
    return report


def _drop_rows_from_csv(path, doc_ids: set[int]) -> None:
    """Rewrite a cleaned CSV without the given doc_ids, chunk by chunk."""
    from fake_news.data.load import read_clean_csv_chunks

    out = path.with_name(path.name + ".filtered")
    first = True
    for chunk in read_clean_csv_chunks(path):
        chunk[~chunk["doc_id"].isin(doc_ids)].to_csv(out, index=False, header=first, mode="w" if first else "a")
        first = False
    os.replace(out, path)
