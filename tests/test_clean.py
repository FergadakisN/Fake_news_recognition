import pandas as pd

from fake_news.data.clean import (
    clean_dataset,
    make_content,
    normalize,
    remove_text_markers,
    remove_title_markers,
)


def test_normalize_fixes_entities_quotes_and_glued_sentences():
    assert normalize("Trump&amp;Co said “hello” and   left") == 'Trump&Co said "hello" and left'
    assert normalize("it ended.Then he left") == "it ended. Then he left"
    assert normalize("on Twitter.Featured image") == "on Twitter. Featured image"
    assert normalize("U.S. officials") == "U.S. officials"  # abbreviations untouched


def test_reuters_dateline_and_tag_removed():
    assert remove_text_markers("WASHINGTON (Reuters) - The U.S. Senate voted.") == "The U.S. Senate voted."
    assert "(Reuters)" not in remove_text_markers("He told (Reuters) that it was fine.")


def test_links_emails_handles_removed():
    s = "Read more at https://example.com/a?b=1 or bit.ly/2jBh4LU. Mail x@nytimes. com. Ask @realDonaldTrump now."
    out = remove_text_markers(s)
    for gone in ["https", "bit.ly", "nytimes", "@realDonaldTrump"]:
        assert gone not in out
    assert "Ask" in out and "now" in out


def test_email_rule_keeps_next_sentence():
    assert remove_text_markers("Write to a@b.com. The president spoke.").endswith("The president spoke.")


def test_image_credits_and_bylines_removed():
    out = remove_text_markers("It was a long day. Featured image via Chip Somodevilla/Getty Images")
    assert out == "It was a long day."
    assert "Getty" not in remove_text_markers("Photo by Win McNamee/Getty Images. Next sentence.")
    assert "Twitter" not in remove_text_markers("Story ends. Follow Pam Key on Twitter @pamkeyNEN")


def test_agency_credits_removed():
    out = remove_text_markers("Trade talks resumed. (Reporting by Jane Doe; Editing by John Roe)")
    assert out == "Trade talks resumed."


def test_title_markers_removed():
    assert remove_title_markers("Trump Rants At Reporters (VIDEO)") == "Trump Rants At Reporters"
    assert remove_title_markers("Senate passes bill [TWEETS]") == "Senate passes bill"
    assert remove_title_markers("Senate passes bill - The New York Times") == "Senate passes bill"
    assert remove_title_markers("Wall talks stall - Breitbart") == "Wall talks stall"
    assert remove_title_markers("Trump says 2016 - report") == "Trump says 2016 - report"  # not an outlet


def test_make_content_lowercases_and_unifies_apostrophes():
    assert make_content("It's Over", "It s done") == "it s over. it s done"
    assert make_content("Breaking!", "Text") == "breaking! text"


def _raw(rows):
    return pd.DataFrame(rows, columns=["doc_id", "title", "text", "label"])


def test_clean_dataset_drops_empty_duplicates_and_conflicts():
    raw = _raw([
        (0, "A", "Same story here.", 0),
        (1, "A", "Same story here.", 0),          # exact duplicate -> dropped, keep doc 0
        (2, "B", None, 1),                        # missing text -> dropped
        (3, "C", "https://only-a-link.com/x", 1), # empty after marker removal -> dropped
        (4, "D", "Disputed text.", 0),
        (5, "D", "Disputed text.", 1),            # same content, both labels -> both dropped
        (6, None, "No title but text.", 1),
    ])
    out = clean_dataset(raw)
    assert out["doc_id"].tolist() == [0, 6]
    assert out.attrs["report"] == {
        "raw_rows": 7, "dropped_empty_text": 2, "dropped_non_english": 0, "dropped_conflicting_labels": 2,
        "dropped_duplicates": 1, "clean_rows": 2,
    }
    assert list(out.columns) == ["doc_id", "title", "text", "content", "n_words", "is_fake"]
    assert out.loc[out.doc_id == 6, "title"].item() == ""


def test_marker_variants_have_same_rows():
    raw = _raw([
        (10, "Senate vote (VIDEO)", "WASHINGTON (Reuters) - The Senate voted.", 0),
        (11, "Other", "Different text. Featured image via Getty Images", 1),
        (12, "Senate vote", "The Senate voted.", 0),  # duplicate of 10 once markers are removed
    ])
    clean, kept = clean_dataset(raw, remove_markers=True), clean_dataset(raw, remove_markers=False)
    assert clean["doc_id"].tolist() == kept["doc_id"].tolist() == [10, 11]
    assert "(Reuters)" in kept.loc[0, "text"] and "(Reuters)" not in clean.loc[0, "text"]
    assert "reuters" not in clean.loc[0, "content"]
