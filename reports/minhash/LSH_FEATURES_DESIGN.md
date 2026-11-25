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
  article sit in train. The grouped split measures how much is left (next section).

## Random vs grouped split (val)

`split_grouped` keeps every near-duplicate group (J ≥ 0.7, `near_dup_groups.csv`) in one partition. The design is
fixed across regimes (2-word shingles, 64 × 2). Only `k` and `min_sim` were retuned on grouped val, and they came
out the same (k = 5, min_sim = 0.10). So the differences below come from the split alone. (The two val sets contain
different articles but have the same class balance, 44.4% fake.)

| Val set | Coverage | Fake / real coverage | F1 covered | ROC-AUC | F1 overall |
|---|---|---|---|---|---|
| random | 18.6% | 16.0% / 20.8% | 0.942 | 0.652 | 0.261 |
| grouped | 17.4% | 13.1% / 20.8% | 0.918 | 0.639 | 0.211 |

Under the random split, **277 val articles (3.0%) have a near-duplicate in train**: 4.8% of the fake ones and 1.4% of
the real ones. Under the grouped split there are none, by construction. Removing them costs fake articles almost
3 points of coverage and leaves real ones unchanged. F1 on covered articles drops by 0.024 and ROC-AUC by 0.013. The
drop is moderate because cleaning had already removed exact duplicates, and only ~4% of articles are in near-duplicate
groups at all. Test is not scored.
