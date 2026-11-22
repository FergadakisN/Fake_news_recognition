# Note for Person 1: cleaning residue found by MinHash/LSH

Found while building the near-duplicate groups (`scripts/find_near_duplicates.py`, threshold 0.7). Nothing in the
cleaning code was changed; this is for you to decide on. All counts are on the current cleaned data (62,525 articles).
To look at an example: `load_clean().set_index("doc_id").loc[<doc_id>, "content"]`.

## 1. Page widgets scraped as article text

The four largest near-duplicate groups are not re-posted stories. Each article has its own title, but the text after
it is the same page element. Every one of these articles is labeled fake, so the widget text tells the label.

| Group (`group_id`) | Size | What the text is | Phrase to search for | Articles with the phrase | Example doc_ids |
|---|---|---|---|---|---|
| 216 | 40 | Comment-policy form (~320 words) | "it s preferable that you choose a name so it becomes clear who said what" | 46, all fake | 216, 5694, 5959, 6097, 6567, 7306 |
| 1352 | 26 | Galactic Connection promo footer (~215 words) | "click here to learn more about alexandra s personalized essences" | 26, all fake | 1352, 3234, 3456, 6250, 7097, 7841 |
| 114 | 20 | Site sidebar: a long list of the site's other headlines (~4,300 words) | "channel list" (only in some members; the rest share the headline list) | 11, all fake | 114, 153, 3480, 5907, 6530, 8399 |
| 4949 | 13 | The Onion's "more headlines" box (~257 words) | "nation puts 2016 election into perspective" | 20, all fake | 4949, 5247, 7117, 7513, 12058, 17099 |

The `comment_widgets` rule (G2) does not catch the first one. For the grouped split this does no harm, because each
group stays in one partition. With the random split, the same widget text appears in train and test.

## 2. Mojibake (encoding residue)

There are **79 articles** (78 fake, 1 real; 55 of them in `train` of `split_random`) where UTF-8 punctuation was
decoded with the wrong code page and normalization did not fix it:

| Form | Example | Articles | Example doc_ids |
|---|---|---|---|
| `â€` + rest (cp1252) | "weâ€tmre trying" (= we're) | 26 | 448, 3840, 4090, 7497, 8620 |
| `â` + C1 control character (latin-1) | "arrival â\x80\x94 one" (= em dash) | 47 | 605, 1376, 2369, 7053, 8033 |
| lone `â` glued to a word | "whereâ reporters" | 21 | 605, 2369, 4454, 7497, 10435 |

(The forms overlap, so the rows add up to more than 79.) Searching for `â` alone over-counts: 125 articles contain it,
but some are real text, e.g. the Persian transliteration in doc 208 ("qorân be zabâné"). The patterns are `â€`,
`â[\x80-\x9f]` and `\wâ\s`.

Effect: in near-duplicate pairs one copy sometimes has the mojibake and the other doesn't (e.g. doc 14856 vs 60545,
"donâ tmt" vs "don t"), which lowers their Jaccard similarity. And because the residue is almost only in fake articles,
it is a small label signal.

## 3. Minor: apostrophe variants

Some copies of the same article drop the apostrophe ("theres", "hes") where others replace it with a space
("there s", "he s"), e.g. docs 41249 vs 53052. This only lowers similarity slightly and does not look worth a rule.
