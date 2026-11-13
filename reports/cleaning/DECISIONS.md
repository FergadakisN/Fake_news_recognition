# Cleaning decisions (WELFake)

The record of what the cleaning removes, why, and how we know. All numbers can be regenerated
(see [How to reproduce](#how-to-reproduce)). The rules themselves, each with a description and
a real example, are in [`src/fake_news/data/markers.py`](../../src/fake_news/data/markers.py).

## Principle

> **Remove what identifies the outlet or the scraping process; keep what the author wrote.**

WELFake merges four source datasets (Kaggle, McIntire, Reuters, BuzzFeed). Real and fake
articles therefore come from different outlets and were scraped in different ways. Text that
identifies the source, such as datelines, bylines, newsletter boilerplate or embed residue,
reveals the label without saying anything about the news. A classifier trained on it learns
*who published the article*, not *what it says*.

## Process

| Step | What | Evidence |
|---|---|---|
| Round 1 | Manual inspection of article starts and ends, titles and known patterns → 15 rules (group **R1**) | `marker_hits.csv` (rules with group R1) |
| Leak scan (before) | On the round-1 data, training split only: every phrase in ≥ 50 articles that is ≥ 98% one class (counting only) | notebook 02, section 3 |
| Round 2 | Findings grouped into G1–G6 and decided by Person 1 (data owner) on 2026-11-12: **remove G1, G2, G3; keep G4, G5, G6**. Later the same day: very short texts (G7) and title labels (G8) kept | this file |
| Implementation | 29 rules added in three passes; after each pass the data was re-scanned to catch leftovers of the approved groups | `marker_hits.csv` (groups G1, G2) |
| Leak scan (after) | The same scan on the final data | notebook 02, section 7 |
| Freeze | Cleaning rules frozen; reference fingerprint and `split_random` regenerated | `cleaning_summary.json` (rules version `9a4c780111ed`) |

The leak scans only ever look at the **training** split. Validation and test data played no role in any decision.

## Decisions

| Group | Decision | What | Rules |
|---|---|---|---|
| **R1** | remove | Round-1 markers: Reuters datelines, outlet title suffixes, title tags `(VIDEO)`, image and Getty credits, URLs, emails, @handles, 21WIRE boilerplate, social bylines, agency credits | `title_tag`, `title_source_suffix`, `reuters_dateline`, `reuters_tag`, `agency_credits`, `21wire_boilerplate`, `scraper_boilerplate`, `social_byline`, `url`, `email`, `twitter_handle`, `image_credit`, `featured_image`, `photo_by_getty`, `getty_images` |
| **G1** | remove | Outlet boilerplate, mostly in real articles: NYT newsletter text and separators, Reuters corrections, graphic links, sponsor notes and tweet disclaimers, CNN/AP/AFP datelines, Breitbart promos, bios and contact lines, broken links | `nyt_briefing_title`, `nyt_briefing`, `nyt_separator`, `reuters_notes`, `reuters_tweet_disclaimer`, `agency_dateline`, `breitbart_promo`, `contact_lines`, `other_links`, `broken_links` |
| **G2** | remove | Scraping artifacts, mostly in fake articles: tweet/Instagram embed attributions, embedded JavaScript, blog footers, comment widgets, share/donate/"read more" lines, licenses, breadcrumbs, page-header dates and timestamps, media credits, bylines | `tweet_embed`, `pic_twitter_residue`, `embedded_code`, `21wire_acr`, `blog_footer`, `comment_widgets`, `share_donate`, `license_lines`, `daily_sheeple`, `breadcrumbs`, `post_dates`, `timestamps`, `header_credits`, `glued_handle`, `media_credit_inline`, `media_credit_end`, `comment_timestamp`, `byline`, `empty_brackets` |
| **G3** | drop rows | Non-English articles (Spanish, French, German, Russian, Arabic, …): 461 fake, 1 real | row filter `non_english` |
| **G4** | keep | Outlet house style: "Mr./Ms." (NYT), "said on Thursday" (Reuters), "U.S.", Reuters/Ipsos poll method sentences ("credibility interval…"), "Watch the video below" | none (documented limitation) |
| **G5** | keep | Topic and time: Rohingya, Catalonia, Brexit (real, 2017) vs Bundy, DAPL, "sheeple" (fake, 2016); dates inside sentences | none (measured by the cluster-held-out split) |
| **G6** | keep | Outlet names inside sentences: "told Reuters", "Breitbart reported", "As Mad World News reported" | none (bylines and promos are removed under G1/G2) |
| **G7** | keep | Very short texts (< 20 words). Decided 2026-11-12: they are the articles as published (teasers, stubs), not a source marker; dropping them would select on a property that correlates with the label | none |
| **G8** | keep | Title labels: 22 editorial labels at the start of titles ("WATCH:" 98% fake, "BREAKING:" 100% fake, "Factbox:" 0% fake; 2,216 training titles). Decided 2026-11-12: kept as part of the published headline; documented so their effect can be discussed in the report | none |

### Judgment calls

- **Dates** are removed only as page metadata ("Posted on <date> by X", "| <date>", "<date> 2 Comments", "<date> by: X Tags: …"). Inside a sentence they are content (G5).
- **Bylines** ("By Name Name") are removed at the start of the text or of a sentence, or when followed by a date. A capitalized "By" mid-sentence otherwise stays.
- **Media credits** at the end of a text are removed only when the last line *starts* with a credit word ("Image:", "H/T", "Via …"). An ordinary sentence ending in "… via Twitter." stays.
- **Language filter:** a text is non-English if > 30% of its letters are outside the Latin script, or if (with ≥ 20 words) its first 3,000 characters contain more Spanish/French/German/Italian/Portuguese function words than English ones. Texts under 20 words are kept.

## Effect

### Rows (`cleaning_summary.json`, `dropped_rows.csv`)

| | Rows | Real | Fake |
|---|---|---|---|
| Raw | 72,134 | 35,028 | 37,106 |
| Dropped: empty text | 883 | 1 | 882 |
| Dropped: non-English | 462 | 1 | 461 |
| Dropped: duplicate | 8,264 | 270 | 7,994 |
| Dropped: conflicting labels | 0 | 0 | 0 |
| **Clean** | **62,525** | **34,756** | **27,769** (44.4%) |

Most dropped rows are fake, so the cleaned data is less balanced than the raw file (44.4% vs 51.4% fake).

### Rules (`marker_hits.csv`)

For every rule: how many articles it changed, per class, in the raw data (`raw_real`, `raw_fake`)
and in the final cleaned data (`kept_real`, `kept_fake`). The largest ones in the cleaned data:

| Rule | Group | Real | Fake |
|---|---|---|---|
| `reuters_dateline` | R1 | 20,925 | 0 |
| `title_source_suffix` | R1 | 8,560 | 434 |
| `image_credit` | R1 | 0 | 6,530 |
| `title_tag` | R1 | 4 | 5,472 |
| `tweet_embed` | G2 | 656 | 5,066 |
| `url` | R1 | 254 | 3,620 |
| `pic_twitter_residue` | G2 | 435 | 3,092 |
| `share_donate` | G2 | 100 | 2,597 |
| `twitter_handle` | R1 | 2,462 | 2,522 |
| `featured_image` | R1 | 0 | 1,989 |
| `social_byline` | R1 | 1,935 | 284 |
| `media_credit_end` | G2 | 4 | 1,907 |

### Leak scans (training split only; `notebooks/02-data-cleaning.ipynb`)

The scan runs inside notebook 02. For `before`, the notebook rebuilds the round-1 data (the 15 R1 rules
only, no language filter) in a temporary folder.

| | Before (round-1 rules) | After (final rules) |
|---|---|---|
| Phrases in ≥ 50 articles that are ≥ 98% one class | 532 | 307 |
| … fake-only | 218 | 46 |
| … real-only | 314 | 261 |

What remains is almost entirely topic (G5: Rohingya, Catalonia, Bundy, …), style (G4) and the title
labels we keep (G8: "factbox"). How much the remaining differences between the classes matter for
classification is a question for the modelling phase; the grouped and cluster-held-out splits measure
the effect of duplicates and topic overlap.

### Known residue

Cleaning is never perfect. After the final scan, small markers below the scan's thresholds
remain, e.g. outlet names in page headers ("Alex Jones | Infowars.com") and a few credit
formats. They are documented here rather than chased further, so the rules stay readable and
the risk of removing real content stays low.

## Files

| File | Content |
|---|---|
| `cleaning_summary.json` | Row counts per step, dropped rows per reason and class, rules version |
| `marker_hits.csv` | Per rule: group, source, description, example, articles changed per class (raw and cleaned) |
| `dropped_rows.csv` | Every dropped `doc_id` with its reason (and the kept copy for duplicates) |

## How to reproduce

Everything in this folder except this file is generated by code, in this order:

```bash
python scripts/prepare_data.py                     # data, splits, fingerprint, cleaning log (~3 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # notebooks, figures, leak scans
```

Changing a rule changes the rules version, the fingerprint and possibly the rows. Person 1 then runs
`python scripts/prepare_data.py --rebuild --update-fingerprint --new-split`, re-runs the notebooks
and updates this file.
