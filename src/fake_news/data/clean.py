"""Clean WELFake articles: normalize, remove source markers, drop unusable and duplicate rows.

Principle: remove what identifies the *outlet* or the *scraping process*; keep what the
author wrote. Every rule and decision is documented in :mod:`fake_news.data.markers` and
``reports/cleaning/DECISIONS.md``; every build logs what each rule did (see below).

Steps (see :func:`clean_dataset`):

1. Normalize title and text: HTML entities, unicode, curly quotes, glued sentences
   ("Twitter.Featured" -> "Twitter. Featured"), whitespace.
2. Remove source/scraping markers (``markers.TITLE_MARKERS``, ``markers.TEXT_MARKERS``).
3. Build ``content`` = title + text, lower-cased, apostrophes -> spaces (so the scraped
   "It s" and the normal "It's" look the same). No stemming or stop-word removal:
   sentence embeddings need natural text and stop words matter for MinHash shingles.
4. Drop rows (``markers.ROW_FILTERS``): empty text, non-English text, exact duplicates of
   ``content`` (keep the lowest ``doc_id``), contents that appear with both labels.

Row selection always uses the marker-free text, so ``remove_markers=True`` and ``False``
return exactly the same ``doc_id`` set.

Cleaning log (written by :func:`build_clean_dataset` to ``reports/cleaning/``):

- ``cleaning_summary.json``: row counts per step and the rules version.
- ``marker_hits.csv``: for every rule, how many articles it changed, per class, in the raw
  data and in the final cleaned data.
- ``dropped_rows.csv``: every dropped ``doc_id`` with its reason (and the kept duplicate).

Memory: :func:`build_clean_dataset` streams the raw CSV in chunks of
``config.CLEAN_CHUNK_ROWS`` rows. Duplicates are found with 20-byte content hashes, so only
one chunk of text is in memory at any time and peak memory does not grow with the dataset.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from fake_news import config
from fake_news.data.markers import MARKERS, ROW_FILTERS, TEXT_MARKERS, TITLE_MARKERS, rules_version

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


def _apply(markers, s: str, hits: list | None, collapse: bool, use_triggers: bool = True) -> str:
    low = s.lower()
    for m in markers:
        if use_triggers and m.triggers and not any(t in low for t in m.triggers):
            continue  # cheap pre-check: this rule cannot match
        new = m.pattern.sub(m.replacement, s)
        if new != s:
            if hits is not None:
                hits.append(m.name)
            s, low = new, new.lower()
    return _WS.sub(" ", s).strip() if collapse else s


def _select(markers, groups):
    return markers if groups is None else [m for m in markers if m.group in groups]


def remove_text_markers(s: str, hits: list | None = None, use_triggers: bool = True, groups=None) -> str:
    """Remove source/scraping markers from a normalized text; names of rules that fired go to ``hits``.

    ``groups`` limits the rules to some groups (e.g. ``("R1",)``); ``use_triggers=False`` skips the
    speed-up pre-checks (same output; used by the tests).
    """
    return _apply(_select(TEXT_MARKERS, groups), s, hits, collapse=True, use_triggers=use_triggers)


def remove_title_markers(s: str, hits: list | None = None, groups=None) -> str:
    """Remove bracket tags and outlet-name suffixes from a normalized title."""
    return _apply(_select(TITLE_MARKERS, groups), s, hits, collapse=True)


def make_content(title: str, text: str) -> str:
    """Model input: title + text, lower-cased, apostrophes replaced by spaces."""
    if title and title[-1] not in ".!?":
        title += "."
    s = f"{title} {text}".lower().replace("'", " ")
    return _WS.sub(" ", s).strip()


# ---- language filter (G3) ------------------------------------------------------------------

_EN_WORDS = frozenset(
    "the of and to in is that for it on was with as he be by at this have from are his not but they an has "
    "we you had or were their been which will who would she her its there what about all said more one can "
    "when if so no".split())
_FOREIGN_WORDS = frozenset("""
    de la que el en los las del se por un una con para es al lo como más pero sus le ya o fue este ha sí porque esta son entre
    le les des et est une du dans pour qui pas sur au avec ce il sont ne se plus par je mais ou nous vous ont été cette leur
    der die und den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach
    wird bei il di che è per una sono della con non si gli le da nel alla anche come più ma questo
    não uma com os para em por mais pelo pela foi são""".split()) - _EN_WORDS
_WORD = re.compile(r"[^\W\d_]+")
_NON_LATIN_LETTER = re.compile("[^\\W\\d_\\u0000-\\u024f]")  # letters outside the Latin script (> U+024F)
_LANG_WINDOW = 3000  # characters used for the function-word comparison


def is_non_english(text: str) -> bool:
    """True for texts that are not English (see ``markers.ROW_FILTERS["non_english"]``)."""
    if not text.isascii():
        letters = sum(map(str.isalpha, text))
        if letters and len(_NON_LATIN_LETTER.findall(text)) / letters > 0.3:  # non-Latin script
            return True
    words = _WORD.findall(text[:_LANG_WINDOW].lower())
    if len(words) < 20:
        return False
    return sum(w in _FOREIGN_WORDS for w in words) > sum(w in _EN_WORDS for w in words)


# ---- chunk processing ------------------------------------------------------------------------

def _clean_chunk(raw: pd.DataFrame, groups=None) -> pd.DataFrame:
    """Normalize + remove markers for a chunk of raw rows (``doc_id, title, text, label``).

    Returns ``doc_id, is_fake``, normalized (``title_n``, ``text_n``) and marker-free
    (``title_c``, ``text_c``, ``content_c``) text, and ``hits`` (names of rules that fired).
    """
    out = pd.DataFrame({
        "doc_id": raw["doc_id"].to_numpy(),
        "is_fake": raw["label"].astype(int).to_numpy(),
        "title_n": raw["title"].fillna("").astype(str).map(normalize).to_numpy(),
        "text_n": raw["text"].fillna("").astype(str).map(normalize).to_numpy(),
    })
    title_c, text_c, hits = [], [], []
    for title, text in zip(out["title_n"], out["text_n"]):
        h: list[str] = []
        title_c.append(remove_title_markers(title, h, groups=groups))
        text_c.append(remove_text_markers(text, h, groups=groups))
        hits.append(h)
    out["title_c"], out["text_c"], out["hits"] = title_c, text_c, hits
    out["content_c"] = [make_content(t, x) for t, x in zip(out["title_c"], out["text_c"])]
    return out


class _RowSelector:
    """Chooses the rows to keep, chunk by chunk, remembering only content hashes.

    Rows must arrive in increasing ``doc_id`` order, so "first occurrence" = lowest doc_id.
    :meth:`keep_mask` drops empty, non-English and repeated contents as they arrive; contents
    that turn out to carry both labels are only known at the end (:meth:`conflicting_doc_ids`).
    Every dropped row is recorded with its reason (:meth:`dropped_rows`).
    """

    def __init__(self, language_filter: bool = True) -> None:
        self.language_filter = language_filter
        self.first_doc: dict[bytes, int] = {}  # content hash -> doc_id of the kept (first) copy
        self.first_y: dict[bytes, int] = {}    # content hash -> label of the kept (first) copy
        self.labels: dict[bytes, int] = {}     # content hash -> bitmask of labels seen (1=real, 2=fake)
        self.count: dict[bytes, int] = {}      # content hash -> number of non-empty English rows
        self.raw_rows = 0
        self.last_doc_id: int | None = None
        self._dropped: list[tuple] = []        # (doc_id, is_fake, reason, duplicate_of, content hash)

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
            doc_id, y = int(doc_id), int(y)
            self.raw_rows += 1
            if text == "":
                self._dropped.append((doc_id, y, "empty_text", None, None))
                continue
            if self.language_filter and is_non_english(text):
                self._dropped.append((doc_id, y, "non_english", None, None))
                continue
            h = hashlib.sha1(content.encode("utf-8")).digest()
            if h in self.first_doc:
                self.count[h] += 1
                self.labels[h] |= 1 << y
                self._dropped.append((doc_id, y, "duplicate", self.first_doc[h], h))
            else:
                self.first_doc[h] = doc_id
                self.first_y[h] = y
                self.count[h] = 1
                self.labels[h] = 1 << y
                keep[i] = True
        return keep

    def _conflicts(self) -> set[bytes]:
        return {h for h, mask in self.labels.items() if mask == 0b11}

    def conflicting_doc_ids(self) -> set[int]:
        return {self.first_doc[h] for h in self._conflicts()}

    def dropped_rows(self) -> pd.DataFrame:
        """``doc_id, is_fake, reason, duplicate_of`` for every dropped row, sorted by doc_id."""
        conflicts = self._conflicts()
        rows = [(d, y, "conflicting_labels" if h in conflicts else reason, None if h in conflicts else dup)
                for d, y, reason, dup, h in self._dropped]
        rows += [(self.first_doc[h], self.first_y[h], "conflicting_labels", None) for h in conflicts]
        out = pd.DataFrame(rows, columns=["doc_id", "is_fake", "reason", "duplicate_of"])
        out["duplicate_of"] = out["duplicate_of"].astype("Int64")
        out["is_fake"] = out["is_fake"].astype("Int64")
        return out.sort_values("doc_id").reset_index(drop=True)

    def report(self) -> dict:
        conflicts = self._conflicts()
        reasons = Counter(r[2] for r in self._dropped)
        conflict_rows = sum(self.count[h] for h in conflicts)
        duplicates = reasons["duplicate"] - sum(self.count[h] - 1 for h in conflicts)
        return {
            "raw_rows": self.raw_rows,
            "dropped_empty_text": reasons["empty_text"],
            "dropped_non_english": reasons["non_english"],
            "dropped_conflicting_labels": conflict_rows,
            "dropped_duplicates": duplicates,
            "clean_rows": len(self.first_doc) - len(conflicts),
        }


class _HitCounter:
    """Counts, per rule and class, how many articles each rule changed (raw rows and kept rows)."""

    def __init__(self) -> None:
        self.raw: Counter = Counter()          # (rule, is_fake) -> articles
        self.kept: dict[int, tuple] = {}       # doc_id -> (is_fake, hits) for kept rows

    def add(self, chunk: pd.DataFrame, keep: np.ndarray) -> None:
        for doc_id, y, hits, k in zip(chunk["doc_id"], chunk["is_fake"], chunk["hits"], keep):
            for name in set(hits):
                self.raw[(name, int(y))] += 1
            if k and hits:
                self.kept[int(doc_id)] = (int(y), set(hits))

    def table(self, excluded_doc_ids: set[int]) -> pd.DataFrame:
        kept: Counter = Counter()
        for doc_id, (y, hits) in self.kept.items():
            if doc_id not in excluded_doc_ids:
                for name in hits:
                    kept[(name, y)] += 1
        rows = [{"order": i, "rule": m.name, "field": m.field, "group": m.group, "source": m.source,
                 "raw_real": self.raw[(m.name, 0)], "raw_fake": self.raw[(m.name, 1)],
                 "kept_real": kept[(m.name, 0)], "kept_fake": kept[(m.name, 1)],
                 "description": m.description, "example": m.example}
                for i, m in enumerate(MARKERS, start=1)]
        return pd.DataFrame(rows)


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
    ``doc_id, title, text, content, n_words, is_fake`` sorted by ``doc_id``. The cleaning log
    is attached as ``attrs["report"]`` (row counts), ``attrs["dropped"]`` and ``attrs["marker_hits"]``.
    """
    chunk = _clean_chunk(raw.sort_values("doc_id"))
    selector, hit_counter = _RowSelector(), _HitCounter()
    keep = selector.keep_mask(chunk)
    hit_counter.add(chunk, keep)
    conflicts = selector.conflicting_doc_ids()
    chunk = chunk[keep]
    chunk = chunk[~chunk["doc_id"].isin(conflicts)]
    out = _output(chunk, remove_markers).reset_index(drop=True)
    out.attrs["report"] = selector.report()
    out.attrs["dropped"] = selector.dropped_rows()
    out.attrs["marker_hits"] = hit_counter.table(conflicts)
    return out


# ---- full build ------------------------------------------------------------------------------

_OUTPUTS = (("clean", True, "CLEAN_CSV"), ("clean_with_markers", False, "CLEAN_WITH_MARKERS_CSV"))


def _stream_clean(outputs, groups=None, language_filter: bool = True):
    """Stream the raw CSV in chunks, clean it and append each chunk to every output.

    ``outputs`` is a list of ``(remove_markers, path)``. Returns the row selector (row counts,
    dropped rows), the rule-hit counter and the doc_ids dropped for conflicting labels.
    """
    from fake_news.data.load import RAW_DTYPES, download_dataset

    download_dataset()  # downloads the pinned raw file if missing; always checks its SHA-256
    selector, hit_counter = _RowSelector(language_filter), _HitCounter()
    first = True
    for raw in pd.read_csv(config.RAW_CSV, dtype=RAW_DTYPES, chunksize=config.CLEAN_CHUNK_ROWS):
        chunk = _clean_chunk(raw.rename(columns={"Unnamed: 0": "doc_id"}), groups)
        keep = selector.keep_mask(chunk)
        hit_counter.add(chunk, keep)
        for remove_markers, path in outputs:
            _output(chunk[keep], remove_markers).to_csv(path, index=False, header=first, mode="w" if first else "a")
        first = False

    conflicts = selector.conflicting_doc_ids()  # (pass 2, only if needed) contents seen with both labels
    if conflicts:
        for _, path in outputs:
            _drop_rows_from_csv(path, conflicts)
    return selector, hit_counter, conflicts


def build_clean_dataset(update_fingerprint: bool = False) -> dict:
    """Clean the raw CSV in chunks and write both variants to ``data/interim/`` (bounded memory).

    Writes ``welfake_clean.csv`` (markers removed) and ``welfake_clean_with_markers.csv``
    (same rows, markers kept). Each is first written to a ``.tmp`` file and checked against
    ``data/processed/clean_fingerprint.json``:

    - fingerprint matches -> the ``.tmp`` file replaces the old CSV;
    - no fingerprint stored yet, or ``update_fingerprint=True`` -> this build becomes the
      reference (Person 1 commits the JSON);
    - otherwise -> the ``.tmp`` files are deleted and ``DataMismatchError`` is raised.

    So a CSV at the final path is never half-written and never differs from the reference.
    On success the cleaning log is written to ``reports/cleaning/``. Returns the row counts.
    """
    from fake_news.data.integrity import check_fingerprint_values, fingerprint_file, load_expected, save_fingerprint_values

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    targets = {variant: getattr(config, attr) for variant, _, attr in _OUTPUTS}
    tmp = {variant: path.with_name(path.name + ".tmp") for variant, path in targets.items()}
    try:
        selector, hit_counter, conflicts = _stream_clean([(rm, tmp[variant]) for variant, rm, _ in _OUTPUTS])
        report = selector.report()
        for k, v in report.items():
            print(f"  {k:28s} {v:>7,}")

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

    write_cleaning_log(report, selector.dropped_rows(), hit_counter.table(conflicts))
    return report


def build_rule_subset(path, groups: tuple[str, ...], language_filter: bool) -> dict:
    """Clean the raw data with only the rules of ``groups`` into one CSV at ``path`` (markers removed).

    Reproduces earlier stages of the cleaning, e.g. the round-1 data that the first leak scan in
    notebook 02 looks at: ``groups=("R1",), language_filter=False``. This is not the shared dataset: there is no
    fingerprint check and no cleaning log. Returns the row counts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    selector, _, _ = _stream_clean([(True, path)], groups=groups, language_filter=language_filter)
    return selector.report()


def write_cleaning_log(report: dict, dropped: pd.DataFrame, marker_hits: pd.DataFrame) -> None:
    """Write the cleaning log to ``config.CLEANING_REPORT_DIR`` (deterministic: safe to commit)."""
    out = config.CLEANING_REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "rules_version": rules_version(),
        "n_rules": len(MARKERS),
        "rows": report,
        "row_filters": ROW_FILTERS,
        "dropped_by_reason_and_class": {
            reason: {"real": int((g["is_fake"] == 0).sum()), "fake": int((g["is_fake"] == 1).sum())}
            for reason, g in dropped.groupby("reason")
        },
    }
    (out / "cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    marker_hits.to_csv(out / "marker_hits.csv", index=False)
    dropped.to_csv(out / "dropped_rows.csv", index=False)
    print(f"Cleaning log written to {out}")


def _drop_rows_from_csv(path, doc_ids: set[int]) -> None:
    """Rewrite a cleaned CSV without the given doc_ids, chunk by chunk."""
    from fake_news.data.load import read_clean_csv_chunks

    out = path.with_name(path.name + ".filtered")
    first = True
    for chunk in read_clean_csv_chunks(path):
        chunk[~chunk["doc_id"].isin(doc_ids)].to_csv(out, index=False, header=first, mode="w" if first else "a")
        first = False
    os.replace(out, path)
