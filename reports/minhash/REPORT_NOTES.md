# Report notes: MinHash / LSH (near-duplicate groups, `split_grouped`, LSH neighbour features)

These are factual notes for writing the report section: the method, every parameter with its reason, the key numbers
and the limitations. Every number comes from a file in `reports/` and can be reproduced with the scripts listed at the
end. Notebook 04 (`notebooks/04-minhash-lsh.ipynb`) shows the same results with figures.

**Only train and val were used. Test was never scored.** All tuning happened on val.

## 1. Purpose

MinHash/LSH has two roles in the project:

1. **Leakage control.** Find near-duplicate articles and keep each group in one partition (`split_grouped`).
   Otherwise a copy in train and another copy in val/test lets a model recognize the text instead of judging it.
2. **Features.** For every article, find similar *training* articles with a train-only LSH index. Three numbers
   summarize their labels, and the final cluster-routed logistic regression uses them as inputs.

Data: 62,525 cleaned WELFake articles, 44.4% fake. Exact duplicates had already been removed in cleaning (8,264 rows).

## 2. Method

All MinHash/LSH code is our own, using NumPy and the standard library only (`src/fake_news/similarity/`).

| Step | What | Formula / detail |
|---|---|---|
| Shingles | an article becomes the set of its k-word shingles (words = `\w+`, lower case), each hashed with `zlib.crc32` | k = 3 for the groups, k = 2 for the features |
| Jaccard | similarity of two shingle sets | J(A,B) = \|A ∩ B\| / \|A ∪ B\| |
| MinHash | 128 hash functions h_i(x) = (a_i·x + b_i) mod P, P = 2^31 − 1; the signature is the minimum per function | P[min h(A) = min h(B)] = J(A,B); the share of equal positions estimates J with standard error √(J(1−J)/128) ≤ 0.044 |
| LSH banding | the signature is cut into b bands of r rows; articles with an identical band share a bucket; sharing ≥ 1 bucket makes a *candidate pair* | P(candidate \| s) = 1 − (1 − s^r)^b |
| Verify | the exact Jaccard of every candidate pair (MinHash is only used to find candidates) | |
| Groups | pairs above the threshold are graph edges; the groups are its connected components (union-find); group id = smallest doc_id | |
| Second tier | MiniLM sentence embeddings (L2-normalized, cosine = dot product), exact top-10 cosine neighbours per article; links merged with the MinHash links in the same union-find | |
| Containment | the share of the *shorter* text's 3-word shingles found in the other text | \|A ∩ B\| / min(\|A\|, \|B\|) |
| Features | train-only LSH index (2-word, 64 × 2); for every article the neighbours with J ≥ 0.1 among the training articles | see section 4 |
| Evaluation | exact Jaccard of all pairs with an inverted index (posting lists); recall, false-positive rate and S-curve of LSH against it | `lsh_eval.py` |

## 3. Parameter choices and reasons

### Near-duplicate groups, tier 1 (MinHash)

| Parameter | Value | Reason | Evidence |
|---|---|---|---|
| Shingle size | 3 words | Short enough that an edited copy still shares most shingles; long enough that unrelated articles share few | — |
| Hash functions | 128, seed `config.SEED` = 42 | Standard error ≤ 0.044; mean \|estimate − exact\| on the candidate pairs is 0.021 | `near_dup_summary.json` |
| Minimum shingles | 5 | Texts of about 6 words or fewer can reach a high Jaccard by chance; 79 articles get no signature and stay singletons (they are kept in the data, cleaning decision G7) | `near_dup_summary.json` |
| Banding | 16 bands × 8 rows | The S-curve threshold (1/16)^(1/8) ≈ 0.71 matches the Jaccard threshold. Measured recall is 0.966 at J ≥ 0.7, 0.998 at J ≥ 0.8 and 1.000 at J ≥ 0.9, with 3,954 candidate pairs out of 1.95 billion | `lsh_eval_retrieval.csv` |
| Jaccard threshold | 0.7 | Pairs between 0.7 and 0.8 are still the same article (new headline, intro or footer; encoding differences), so 0.8 would split real copies. For the grouped split a missed copy (a leak) costs more than a false group. Below 0.7, 16 × 8 is no longer reliable (recall 0.83 at J ≥ 0.6). Going from 0.8 to 0.7 adds 120 groups and about 320 articles, and the largest group stays at 40 | `threshold_comparison.csv`, `threshold_examples.csv` |

### Near-duplicate groups, tier 2 (embeddings)

| Parameter | Value | Reason | Evidence |
|---|---|---|---|
| Neighbours | exact top-10 cosine per article | Exact (no approximation) and fast enough: about 1 min for 62,525 × 384 | — |
| Cosine threshold | 0.92 | Below ~0.92 most pairs are *different* articles on the same running topic, and union-find chains whole topics. At 0.90 one group held 35 different Hariri/Lebanon articles and another 15 travel-ban articles; at 0.92 they shrink to 15 and 3. Mixed fake/real groups: 80 at 0.90, 44 at 0.92 | `embedding_threshold_comparison.csv`, `LSH_FEATURES_DESIGN.md` |
| Containment rule | pairs with 0.90 ≤ cosine < 0.92 are also linked if containment ≥ 0.7 | Raising the threshold from 0.90 to 0.92 dropped 33 truncated-copy and 26 edited-copy links. Topic neighbours share little wording and cannot pass the rule. It recovers all 59 copy links, adds 25 partial overlaps (containment 0.70–0.79), admits 0 paraphrases and brings no topic chain back | `two_tier_summary.json` |
| Link types (descriptive only) | truncated copy: containment ≥ 0.8 and length ratio ≥ 1.5; edited copy: containment ≥ 0.8; partial overlap: 0.3–0.8; paraphrase: < 0.3 | Shows that "truncated copies" are copies MinHash misses because of their length (Jaccard divides by the union, which the extra text inflates), not paraphrases | `truncated_copy_examples.csv`, `paraphrase_examples.csv` |

### `split_grouped`

70 / 15 / 15, groups assigned whole (each to the partition furthest below its target size, in random order with seed
42), stratified by the group's majority label. `split_random` was not changed.

### LSH neighbour features

| Parameter | Value | Reason | Evidence |
|---|---|---|---|
| Shingle size | 2 words | Coverage is the bottleneck. With 3-word shingles even exact search finds a neighbour for only 12.8% of random-val articles; with 2-word shingles LSH reaches 18.6%, and the vote's ROC-AUC rises from 0.597 to 0.652 | `LSH_FEATURES_DESIGN.md`, `voting_random_comparison.csv` |
| Banding | 64 × 2 | Proposes pairs down to J ≈ 0.1 (P = 0.47 at J = 0.1, 0.77 at 0.15, 0.989 recall at J ≥ 0.2) | `lsh_eval_feature_index.csv` |
| `min_sim` | 0.10 | The best value on val for every configuration (the grid was 0.1–0.5, so 0.10 is its lower edge) | `voting_*_tuning.csv` |
| Top-k | none (all neighbours with J ≥ 0.1) | k made no difference in the vote (identical AUC for k = 1…50), and a cap would make `neighbor_count` meaningless | `LSH_FEATURES_DESIGN.md` |
| Prefilter | a candidate needs ≥ 5 of 128 equal MinHash positions before the exact check | The number of agreeing positions is Binomial(128, J); at J = 0.1 (mean 12.8) the probability of ≤ 4 is < 1%. Measured: the same recall with and without it, and about 2.3× faster | `lsh_eval_feature_index.csv`, `LSH_FEATURES_DESIGN.md` |
| Fallback (no neighbour) | train fake rate, 0.4441 | The best guess without evidence; computed from train only | `lsh_features_*.json` |
| Exclusion rule | random: a training article ignores itself; grouped: it ignores its whole near-duplicate group | A training article is in the index; its features must mean the same as for a val article, which never finds itself and, under the grouped split, never has a group member in train | `lsh_features_sanity_*.csv` |

## 4. Key numbers

### Groups (`data/processed/near_dup_groups.csv`)

| | Groups (2+ articles) | Articles | Largest | Mixed fake/real groups |
|---|---|---|---|---|
| MinHash tier alone | 1,002 | 2,445 (3.9%) | 40 | 13 (28 articles) |
| **Final (MinHash + cosine 0.92 + containment)** | **1,433** | **3,494 (5.6%)** | **40** | **46 (101 articles)** |
| tier `minhash` | 914 | 2,195 | | 10 |
| tier `embedding` | 404 | 883 | | 31 |
| tier `embedding+containment` | 37 | 79 | | 2 |
| tier `both` | 51 | 204 | | 3 |
| tier `both+containment` | 27 | 133 | | 0 |

- **Links:** 2,274 from cosine ≥ 0.92 and 243 from the containment rule. 88 MinHash groups were enlarged or joined.
- **The four largest MinHash groups are scraping residue,** not re-posted stories: the same comment form (40 articles),
  promo footer (26), site sidebar (20) or "more headlines" box (13) under different titles. All their articles are
  fake (`NOTE_cleaning_residue.md`).
- **Mixed groups** are mostly Reuters articles re-posted by sites labeled fake. WELFake labels by source, so the same
  text carries both labels.

### Splits

| | train | val | test | Fake share |
|---|---|---|---|---|
| `split_random` (unchanged) | 43,767 | 9,379 | 9,379 | 44.4% |
| `split_grouped` | 43,766 | 9,380 | 9,379 | 44.4% |

- **Random-split leakage:** 394 val articles (4.2%) have a member of their final group in train: 252 fake (6.0% of
  fake val) and 142 real (2.7%). With MinHash groups alone it is 277 (3.0%).
- **Grouped split:** 0 val and 0 test articles have a group member in train.

### LSH neighbour features (`data/processed/features/lsh__{random,grouped}.csv`)

| | No neighbour (train / val) | fake_neighbor_share, articles with a neighbour (fake / real, val) |
|---|---|---|
| random | 80.6% / 81.4% | 0.945 / 0.038 |
| grouped | 83.5% / 84.0% | 0.912 / 0.041 |

Train and val look alike in both regimes, which is what the exclusion rule is for. Building the features takes about
9.4 ms per article (~10 min per regime), with ~474 LSH candidates per article before the prefilter.

### Neighbour vote (analysis tool, val)

| Val set | Coverage (fake / real) | F1 on covered | ROC-AUC | F1 overall |
|---|---|---|---|---|
| random | 18.6% (16.0 / 20.8) | 0.942 | 0.652 | 0.261 |
| grouped | 16.0% (11.8 / 19.3) | 0.918 | 0.629 | 0.194 |

The difference is the measured effect of near-duplicate leakage on this signal. Coverage drops mostly for fake
articles.

### Evaluation 1: retrieval quality against exact Jaccard (3-word shingles)

Ground truth: exact Jaccard of every pair, computed with an inverted index.
- **Sample:** 3,000 random articles (seed 42), all 4,498,500 pairs.
- **Full data:** all 62,446 articles with a signature, every pair with J ≥ 0.2 (10,691 of 1.95 billion pairs).

The sample holds only 20 pairs with J ≥ 0.5 and 6 with J ≥ 0.7: a pair is sampled only if both articles are, which
happens with probability ≈ 0.2%. Its intervals are therefore wide. The full data gives the precise numbers.

Recall (full data; 95% Wilson intervals all within ±0.012):

| Banding | J ≥ 0.5 | J ≥ 0.6 | J ≥ 0.7 | J ≥ 0.8 | J ≥ 0.9 | Candidate pairs | per article | FP rate at 0.7 |
|---|---|---|---|---|---|---|---|---|
| 16 × 8 (groups) | 0.607 | 0.832 | **0.966** | 0.998 | 1.000 | 3,954 | 0.13 | 15.2% |
| 32 × 4 | 0.973 | 0.999 | 1.000 | 1.000 | 1.000 | 7,621 | 0.24 | 54.4% |
| 64 × 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 184,352 | 5.90 | 98.1% |

- True pairs at each threshold: 6,435 / 4,434 / 3,472 / 2,997 / 2,457.
- Sample, 16 × 8: recall 4 of 6 pairs at J ≥ 0.7 (0.67, CI 0.30–0.90); 32 × 4 and 64 × 2 find all 6.
- False-positive rate = candidates with exact J below the threshold / all candidates. They are removed by the exact
  check, so they cost time, not quality.
- **Measured S-curve vs theory:** the candidate rate per 0.05 bin follows 1 − (1 − s^r)^b for every banding.
  - Full data: mean absolute difference 0.02 (16 × 8), 0.015 (32 × 4) and 0.001 (64 × 2); maximum 0.09.
  - The comparison uses the mean predicted probability of the pairs in each bin
    (`reports/figures/lsh_eval_s_curves.png`, `lsh_eval_s_curve.csv`).
- **The 118 pairs with J ≥ 0.7 that 16 × 8 missed** all lie between 0.70 and 0.85.
  - 78 of them end up in the same final group anyway, through other links (43 within MinHash groups alone).
  - **21 cross partitions of `split_grouped`**: a small residual leak (`lsh_eval_missed_pairs.csv`).

Runtime (3-word, full data, 62,446 articles):

| Method | Time |
|---|---|
| LSH 16 × 8: signatures 25.6 s + banding 1.2 s + verification 0.3 s | **27 s** |
| LSH 64 × 2 | 39 s |
| Exact search, inverted index (all pairs) | 182 s |
| Exact search, pair by pair (`np.intersect1d`, ~41 µs per pair; extrapolated) | ~79,000 s ≈ 22 h |

On the 3,000 sample all methods take 1–2 s (inverted index 1.9 s, LSH 1.4–1.5 s). The advantage of LSH grows with
data size.

### Evaluation 2: the feature index (2-word, 64 × 2, prefilter, J ≥ 0.1)

The 9,380 grouped-val articles were queried against the train-only index, exactly as the features are built, and
compared with exact search. Recall is counted over (val article, training article) pairs:

| | J ≥ 0.1 | J ≥ 0.2 | J ≥ 0.3 | J ≥ 0.5 | J ≥ 0.7 |
|---|---|---|---|---|---|
| true pairs | 5,694 | 1,963 | 1,134 | 684 | 33 |
| recall | **0.761** | 0.989 | 1.000 | 1.000 | 1.000 |

- **Val articles with at least one neighbour:** 16.0% with LSH (the features) vs 19.9% with exact search.
- **Time per val article:** LSH + prefilter 9.6 ms (486 candidates) vs exact inverted-index search 7.3 ms.
- **Sample (3,000):** recall 0.81 at J ≥ 0.1 and 1.0 above. The prefilter does not change recall and takes 0.65 ms
  instead of 1.46 ms.

### Evaluation 3: leakage audit (grouped split, val)

Val articles whose closest training article has a 2-word Jaccard ≥ t:

| | t = 0.3 | t = 0.5 |
|---|---|---|
| grouped, exact search | 250 (2.7%; fake 3.2%, real 2.2%) | 93 (1.0%; fake 1.2%, real 0.8%) |
| grouped, LSH features | 250 | 93 (identical to exact) |
| random, LSH features | 564 (6.0%; fake 8.8%, real 3.8%) | 400 (4.3%; fake 6.8%, real 2.2%) |

**What the 93 grouped pairs at J ≥ 0.5 are** (closest training article; `leakage_audit_pairs.csv`):
- *Similarity:* median 3-word Jaccard 0.59 (below the group threshold of 0.7), and 91% have 3-word containment ≥ 0.7.
- *Not the same story:* median embedding cosine 0.68. 96% of the pairs share their label.
- *Heuristic kinds:*
  - **40 share template text** (cosine < 0.8): different articles with the same footer, sidebar or headline box
    ("Chart of the Day", The Onion).
  - **31 are short texts** under 50 words, such as teasers built from one template.
  - **22 look like the same story** (cosine ≥ 0.8).
- So most of the remaining similarity is a site/template signal, not story leakage.

`fake_neighbor_share` alone on these subsets (val; ROC-AUC within each subset; uncovered articles keep the fallback):

| Regime | Subset | n | Coverage | F1 on covered | ROC-AUC |
|---|---|---|---|---|---|
| grouped | max J ≥ 0.5 | 93 | 100% | 0.960 | 0.954 |
| grouped | 0.3 ≤ max J < 0.5 | 157 | 100% | 0.912 | 0.901 |
| grouped | max J < 0.3 (rest) | 9,130 | 13.7% | 0.913 | 0.612 |
| grouped | rest with a neighbour (0.1 ≤ max J < 0.3) | 1,249 | 100% | 0.913 | 0.943 |
| grouped | all val | 9,380 | 16.0% | 0.918 | 0.629 |
| random | max J ≥ 0.5 | 400 | 100% | 0.989 | 0.986 |
| random | max J < 0.3 (rest) | 8,815 | 13.4% | 0.892 | 0.607 |
| random | all val | 9,379 | 18.6% | 0.942 | 0.652 |

Figures: `reports/figures/leakage_max_jaccard.png` (share of val with a train neighbour ≥ t, random vs grouped) and
`lsh_features_val_grouped.png` (feature distributions).

### Cluster split (`data/processed/features/lsh__cluster.csv`)

`split_cluster` holds out whole K-means topic clusters in val and test. The features were built exactly as for
grouped: the same train-only index, and training articles ignore their whole near-duplicate group. No near-duplicate
group is split across partitions in `split_cluster`. Test labels were not used, so the test rows show label-free
statistics only:

| | grouped val | cluster val | grouped test | cluster test |
|---|---|---|---|---|
| no neighbour | 84.0% | **95.2%** | 83.9% | 93.7% |
| mean `max_jaccard` | 0.034 | 0.010 | 0.034 | 0.015 |
| mean `neighbor_count` | 0.46 | 0.12 | 0.41 | 0.24 |
| coverage fake / real | 11.8% / 19.3% | 5.0% / 4.7% | – | – |
| F1 of `fake_neighbor_share` on covered articles | 0.918 | 0.937 | – | – |
| ROC-AUC of `fake_neighbor_share` (all articles) | 0.629 | **0.542** | – | – |

- **On unseen topics the features weaken sharply,** because they rarely fire: only 4.8% of cluster-val articles have
  a training article with J ≥ 0.1, against 16.0% under grouped. ROC-AUC falls from 0.629 to 0.542.
- **When a neighbour exists it is still reliable** (F1 0.937 on covered articles), so the signal is mostly
  same-topic similarity, and without the topic in train there is little left.
- **Train and val features do not look alike under this split.** 18.0% of training articles have a neighbour (their
  topics are in train), against 4.8% of val and 6.3% of test articles. The exclusion rule removes only the
  near-duplicate group, not the topic. A classifier trained on these features therefore sees them fire far more
  often in training than it will on held-out topics.

## 5. Limitations

1. **The feature index misses weak neighbours.** At 0.1 ≤ J < 0.2, 64 × 2 finds about three quarters of the pairs
   (recall 0.76 at J ≥ 0.1), so 16.0% of val articles get a neighbour instead of 19.9% with exact search. From J ≥ 0.3
   nothing is missed.
2. **At this data size, exact search is a real alternative.** The inverted index computes all exact pairs of the full
   data in 3 min (3-word) and is faster than LSH for the 2-word feature queries (7.3 vs 9.6 ms per article), because
   2-word shingles produce ~490 LSH candidates per article. LSH is the faster method for the groups (27 s vs 182 s).
   Its advantage grows with data size, but on 62k articles it is not decisive.
3. **Residual near-copy leakage in `split_grouped`.**
   - 21 pairs with 3-word J between 0.70 and 0.85 were missed by 16 × 8 and cross partitions.
   - 93 val articles (1.0%) still have a training article at 2-word J ≥ 0.5, mostly shared template text or short
     teasers.
4. **Topic chaining is reduced, not removed.** A 15-article Hariri group (different articles on one topic, linked by
   embeddings only) remains at cosine 0.92. 46 groups mix fake and real labels.
5. **MiniLM reads only the opening of an article** (256 word pieces), so embedding links judge the beginning of the
   text. A shared header can create a false link: the author-bio case in `NOTE_cleaning_residue.md` joins two MinHash
   groups (6 fake articles).
6. **Scraping residue drives the largest groups** (comment forms, footers, sidebars; all fake). Under the random split
   this text tells the label. The grouped split keeps each such group in one partition, but the same text can still
   appear in smaller groups or below the threshold (`NOTE_cleaning_residue.md`).
7. **The retrieval design was tuned with the neighbour vote on random val.** `min_sim` = 0.10 is the lower edge of the
   grid that was tried. Only `k` and `min_sim` were retuned on grouped val, and they did not change.
8. **Thresholds are judgement calls.** J ≥ 0.7, cosine ≥ 0.92 and containment ≥ 0.7 were set from examples and group
   statistics, not from labelled ground truth; there is no annotated set of "true duplicates".
9. **The 3,000-article sample is too small for recall at high similarity** (6 pairs at J ≥ 0.7). The full-data check
   fills this gap for 3-word shingles. For 2-word shingles the val-vs-train check plays that role.
10. **Embeddings fingerprint.** The shared `embeddings.npy` was built on Windows. Its fingerprint check fails on
    macOS/Linux because of CRLF vs LF line endings; `paraphrases.load_embeddings_checked` accepts both. The shared
    module is unchanged.
11. **Timings** are from one MacBook run, without repetitions, and vary by about ±10% between runs.

## 6. Reproduce

```bash
python scripts/find_near_duplicates.py              # MinHash tier (~1 min)
python scripts/find_paraphrases.py --write          # second tier + final groups
python scripts/build_grouped_split.py               # split_grouped in splits.csv
python scripts/build_lsh_features.py --regime random grouped cluster   # ~10 min each
python scripts/run_model_b.py --design fixed --regime grouped  # neighbour vote (analysis, val only)
python scripts/evaluate_lsh.py                      # evaluation (~35 min the first time, then ~4 min)
```

| File | Contents |
|---|---|
| `near_dup_summary.json`, `threshold_*.csv`, `largest_groups.csv` | MinHash tier |
| `two_tier_summary.json`, `two_tier_largest_groups.csv`, `embedding_*.csv`, `*_examples.csv` | second tier |
| `lsh_features_{random,grouped}.json`, `lsh_features_sanity_*.csv` | features |
| `lsh_eval_retrieval.csv`, `lsh_eval_runtime.csv`, `lsh_eval_s_curve.csv`, `lsh_eval_missed_pairs.csv` | evaluation 1 |
| `lsh_eval_feature_index.csv` | evaluation 2 |
| `leakage_audit_counts.csv`, `leakage_audit_scores.csv`, `leakage_audit_pairs.csv` | evaluation 3 |
| `lsh_eval_summary.json` | settings, timings and headline numbers of the evaluation |
| `../lsh_features/voting_*` | neighbour vote |
| `LSH_FEATURES_DESIGN.md`, `NOTE_cleaning_residue.md` | design note, cleaning residue |
