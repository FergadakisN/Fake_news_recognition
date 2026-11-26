# LSH neighbour features: retrieval design (chosen on val, random split)

The final classifier gets three **LSH neighbour features** per article: the similarity-weighted share of fake
labels among its similar training articles, the highest Jaccard similarity, and the number of such neighbours
(`fake_news.similarity.lsh_features`, built by `scripts/build_lsh_features.py`). Similarity is the Jaccard
similarity of word shingles. Candidates come from a train-only MinHash/LSH index (128 hashes, `config.SEED`) and
are scored with exact Jaccard.

The retrieval design (shingle size, banding) was chosen with the **neighbour voting analysis**
(`scripts/run_model_b.py`, an analysis tool, not a pipeline model). It uses the neighbours on their own as a
classifier:

- **Vote:** the similarity-weighted share of fake labels among the top `k` neighbours with similarity ≥ `min_sim`.
- **No neighbour:** the article is *uncovered* and gets the training fake rate (0.444, computed from train in code).

Everything below is on **val**. No test article was scored.
Results: `reports/lsh_features/voting_<regime>_*.csv|json`. (This analysis was called "Model B" before the
project plan changed.)

## Decision

| | Value |
|---|---|
| Shingles | **2 words** (the near-duplicate groups keep 3 words) |
| LSH banding | **64 bands × 2 rows** |
| `min_sim` | **0.10** (the features use every neighbour with J ≥ 0.10) |
| `k` | no limit in the features; in the vote, k = 5 (k has no effect on val, see below) |
| Prefilter | candidates must agree in ≥ 5 of 128 MinHash positions before the exact check (added later for speed: identical features on all 9,379 random-val articles, 2.3× faster) |
| Fallback | train fake rate, 0.4441 |
| Selection rule | highest val ROC-AUC of the vote among the LSH configurations (ties: F1 on covered, then speed) |

## Choices specific to the features

**No top-k limit: every neighbour with J ≥ 0.10 counts.**
- In the voting analysis `k` had no effect: ROC-AUC was identical for k = 1…50 (to 3 decimals), because covered
  articles have few neighbours above 0.10. Limiting k would not change `fake_neighbor_share` for almost any article.
- A cap would destroy `neighbor_count`, which is only informative if it can grow (with k = 5 it would be 0–5).
- `min_sim = 0.10` already bounds the neighbours. On random val the mean count is 0.52, the 99th percentile 11 and
  the maximum 40, so there is no runaway cost.

**Prefilter: `MIN_AGREE = 5` of 128 MinHash positions before the exact Jaccard check.**
- *Why it is needed:* the features query all 62,525 articles per split, not only val. With 2-word shingles and
  64 × 2 bands, an article gets ~480 LSH candidates, and on a val sample **only 0.11% of them have J ≥ 0.10**.
  Almost all the time went into the exact Jaccard of hopeless pairs (~22 ms per article, ~23 min per split).
- *Why it is safe:* two sets with Jaccard J agree in each MinHash position with probability J, so the number of
  agreeing positions is Binomial(128, J). A pair at exactly J = 0.10 expects 12.8 agreements, and has a probability
  below 1% of reaching only 4 or fewer. Pairs above 0.10 are even safer.
- *Measured:* on a 400-article val sample, keeping candidates with ≥ 5 agreements dropped 62% of the candidates and
  none of the 200 pairs with J ≥ 0.10. On **all 9,379 random-val articles**, the features with and without the
  prefilter are **identical** (0 of 4,903 neighbours lost), and queries take 9.4 ms instead of 21.7 ms (2.3× faster).
- The check is cheap: comparing two 128-value signatures costs far less than intersecting two shingle sets of
  ~500 elements each. Exact Jaccard is still computed for every candidate that passes.
- `MIN_AGREE = 0` turns the prefilter off.

## Comparison of retrieval designs (neighbour vote on val, best `k` / `min_sim` per configuration)

Main metrics are coverage, F1 on covered articles and ROC-AUC. Overall F1 is secondary, because uncovered articles all
get the fallback and so count as predicted real.

| Retrieval | Coverage | Fake / real coverage | F1 covered | ROC-AUC | F1 overall | Candidates / query (p90) | ms / query |
|---|---|---|---|---|---|---|---|
| 3-word, LSH 16×8 | 3.4% | 5.6% / 1.7% | 0.991 | 0.535 | 0.105 | 0.1 (0) | 0.02 |
| 3-word, LSH 32×4 | 5.2% | 8.1% / 2.9% | 0.988 | 0.552 | 0.148 | 0.2 (0) | 0.03 |
| 3-word, LSH 64×2 | 11.1% | 12.9% / 9.6% | 0.957 | 0.597 | 0.218 | 3.9 (8) | 0.26 |
| 3-word, exact search (reference, no LSH) | 12.8% | 14.0% / 11.9% | 0.941 | 0.608 | 0.230 | 17,893 | 3.0 |
| **2-word, LSH 64×2 (chosen)** | **18.6%** | 16.0% / 20.8% | 0.942 | **0.652** | 0.261 | 472 (1,332) | 32 |

## Why

- **Coverage is the bottleneck, and 2-word shingles raise it the most.** With 3-word shingles even exact search covers
  only 12.8%. Articles on the same story but in different words share almost no 3-word phrases. 2-word shingles also
  match articles that reuse pairs of words (names, phrases of the same event), and coverage rises to 18.6%.
- **AUC rises from 0.597 to 0.652** while F1 on covered articles barely drops (0.957 → 0.942). The extra neighbours
  are still informative.
- **The cost is acceptable here.** 472 candidates and 32 ms per query on average (p90: 1,332 candidates), against 4
  candidates and 0.26 ms for 3-word shingles. That is about 5 minutes for val + test. It stays under the limit of
  1,000 candidates per query, but it would not scale like the 3-word index.

## Caveats for the report

- **`min_sim` sits at the edge of the grid (0.10) for every configuration.** Lower thresholds were not tried (the
  range was fixed at 0.1–0.5). A higher threshold makes covered predictions more accurate (2-word: F1 covered 0.942 at
  0.10 → 0.988 at 0.40) but cuts coverage to 5%.
- **`k` does not matter.** AUC is identical for k = 1…50 at 3 decimals, because covered articles have few neighbours
  above `min_sim`. k = 5 won only on the tie-break.
- **The coverage gap between the classes flips.** With 3-word shingles fake articles are covered more often
  (12.9% vs 9.6%: fake stories get re-posted). With 2-word shingles real ones are (20.8% vs 16.0%), presumably because
  Reuters-style articles share fixed phrasing.
- **Under the random split, much of the high F1 on covered articles is near-duplicate matching.** Near-copies of a val
  article sit in train. The grouped split measures how much is left ("Random vs grouped split" below).

## The near-duplicate groups behind `split_grouped` and the exclusion rule

The groups have two tiers, merged with union-find (`data/processed/near_dup_groups.csv`, column `tier`):

1. **MinHash:** 3-word shingles, LSH 16 × 8, exact Jaccard ≥ 0.7 (near-copies with the same wording).
2. **Embeddings:** MiniLM vectors, exact top-10 cosine neighbours of every article. A pair is linked if
   - cosine ≥ **0.92**, or
   - 0.90 ≤ cosine < 0.92 **and** 3-word shingle containment ≥ **0.7** (the containment rule, below).

### Why cosine 0.92 and not 0.90

Below about 0.92, most embedding pairs are *different* articles about the same running news topic, and union-find
chains a whole topic into one group:

- **Topic chaining at 0.90.** One group held 35 different Reuters articles on the Hariri / Lebanon crisis from several
  weeks ("Lebanon PM Hariri resigns, assails Iran and Hezbollah", "Hariri to fly to Paris within 48 hours", "Hariri
  rescinds resignation"…), linked only by embeddings through hub articles (one with 14 links). Another 15 articles
  covered months of travel-ban court cases.
- At 0.92 the travel-ban chain is gone (largest group: 3 articles) and the Lebanon chain shrinks from 35 to 15.
  **0.92 reduces topic chaining but does not remove it completely:** a 15-article Hariri group remains.
- **Mixed fake/real groups: 80 at 0.90, 44 at 0.92** (13 with MinHash alone). Fake and real outlets cover the same
  events, so topic-level links create mixed groups much faster than copies do.
- Holding out whole topics is what `split_cluster` tests. The grouped split should remove copies and paraphrases of
  the *same story*, so the two regimes stay distinct.

| Cosine (no containment rule) | Groups (2+) | Articles | Largest | Mixed groups | Largest Hariri / travel-ban group |
|---|---|---|---|---|---|
| MinHash only | 1,002 | 2,445 | 40 | 13 | – |
| 0.90 | 1,827 | 4,500 | 40 | 80 | 35 / 15 |
| 0.92 | 1,402 | 3,412 | 40 | 44 | 15 / 3 |

### Truncated copies are their own category

Each embedding link MinHash did not have is described by its **containment**: the share of the *shorter* text's
3-word shingles that also occur in the other text (`paraphrases.link_type`; descriptive only).

| Type | Rule | Links (final) | What it is |
|---|---|---|---|
| truncated copy | containment ≥ 0.8, length ratio ≥ 1.5 | 189 | one text contains the other but is much longer (e.g. 1,662 vs 225 words, cosine 1.00, Jaccard 0.13) |
| edited copy | containment ≥ 0.8, similar length | 127 | a copy with edits, Jaccard just below 0.7 |
| partial overlap | 0.3 ≤ containment < 0.8 | 346 | shared passages: updated versions, long quotes |
| paraphrase | containment < 0.3 | 167 | the same story in other words |

**Truncated copies are copies MinHash misses because of length**, not paraphrases. Jaccard divides by the union,
which the extra length inflates, so a fully contained text can score far below 0.7, while MiniLM, which only reads
the opening, sees two identical texts. Examples: `truncated_copy_examples.csv` and `paraphrase_examples.csv`.

### The containment rule

Raising the threshold from 0.90 to 0.92 also dropped copies whose cosine was just below 0.92: **33 truncated-copy and
26 edited-copy links** (0.92 keeps 156 of the 189 truncated copies, 83%). To recover them without the topic chains, a
pair with 0.90 ≤ cosine < 0.92 is also linked if its containment is ≥ 0.7. Topic neighbours share almost no wording,
so they cannot pass it. Of the 243 links the rule admits:

- **all 33 truncated and all 26 edited copy links** are recovered;
- **25 other links** are partial overlaps with containment 0.70–0.79 (two versions of one China trade story, the
  "65 journalists" list from two sites, consecutive episodes of one podcast);
- 159 fall inside existing MinHash groups and change nothing; **no paraphrase passes** (containment < 0.3);
- **no chain comes back:** the largest groups are unchanged (40, 26, 20, 15, 13, 13), Hariri stays at 15 and travel
  ban at 3. Groups 1,402 → 1,433, articles 3,412 → 3,494, mixed groups 44 → 46.

Groups changed by the rule have the tier `embedding+containment` (37 groups, 79 articles) or `both+containment`
(27 groups, 133 articles: a MinHash group enlarged by such a link).

### Final groups

| | Groups (2+) | Articles | Mixed fake/real |
|---|---|---|---|
| MinHash tier alone | 1,002 | 2,445 (3.9%) | 13 |
| **final** | **1,433** | **3,494 (5.6%)** | **46** (101 articles) |
| tier `minhash` (unchanged) | 914 | 2,195 | 10 |
| tier `embedding` | 404 | 883 | 31 |
| tier `embedding+containment` | 37 | 79 | 2 |
| tier `both` | 51 | 204 | 3 |
| tier `both+containment` | 27 | 133 | 0 |

88 MinHash groups were enlarged or joined by embedding links. The largest group is still the 40-article comment-form
widget group (tier `minhash`). One false link comes from a shared author-bio header (`NOTE_cleaning_residue.md`).

## Random vs grouped split (val)

`split_grouped` keeps every near-duplicate group in one partition. It was first built from the MinHash groups alone
(J ≥ 0.7) and now uses the final groups above. The retrieval design is fixed across regimes (2-word shingles,
64 × 2). Only `k` and `min_sim` were retuned on grouped val: min_sim came out 0.10 every time, and k again made no
difference (k = 3 and k = 5 give identical results). So the differences below come from the split alone. (The val
sets contain different articles but have the same class balance, 44.4% fake.)

| Val set | Coverage | Fake / real coverage | F1 covered | ROC-AUC | F1 overall |
|---|---|---|---|---|---|
| random | 18.6% | 16.0% / 20.8% | 0.942 | 0.652 | 0.261 |
| grouped, MinHash groups only (earlier) | 17.4% | 13.1% / 20.8% | 0.918 | 0.639 | 0.211 |
| grouped, cosine 0.90 (rejected) | 15.9% | 11.6% / 19.3% | 0.902 | 0.625 | 0.187 |
| **grouped, final groups (0.92 + containment rule)** | **16.0%** | **11.8% / 19.3%** | **0.918** | **0.629** | **0.194** |

**Leakage in the random split: 394 val articles (4.2%) have a member of their final near-duplicate group in train**
(252 fake = 6.0% of the fake val articles; 142 real = 2.7%). With the MinHash groups alone it was 277 (3.0%). Under
the grouped split there are none, by construction: `lsh_features_sanity_grouped.csv` shows 0 val and 0 test articles
with a group member in train, and train and val look alike (83.5% vs 84.0% without a neighbour; fake share 0.913 vs
0.912 for fake articles).

Removing them costs mostly fake coverage (16.0% → 11.8%) and less real coverage (20.8% → 19.3%), and ROC-AUC drops
from 0.652 to 0.629. The drop stays moderate because cleaning had already removed exact duplicates and only 5.6% of
articles are in a group. Test is not scored.
