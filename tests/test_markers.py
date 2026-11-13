"""Round-2 cleaning rules (G1 outlet boilerplate, G2 scraping artifacts, G3 language filter)."""

import pytest

from fake_news.data.clean import is_non_english, normalize, remove_text_markers
from fake_news.data.markers import MARKERS, KEPT_SIGNALS, ROW_FILTERS, rules_version


def clean(s: str) -> str:
    return remove_text_markers(normalize(s))


@pytest.mark.parametrize("marker", MARKERS, ids=lambda m: m.name)
def test_every_documented_example_matches_its_rule(marker):
    assert marker.pattern.search(normalize(marker.example)), f"example of {marker.name} does not match its pattern"


def test_registry_is_complete():
    names = [m.name for m in MARKERS]
    assert len(names) == len(set(names)), "rule names must be unique"
    for m in MARKERS:
        assert m.field in {"text", "title"} and m.group in {"R1", "G1", "G2"}
        assert m.description and m.source
    assert set(ROW_FILTERS) == {"empty_text", "non_english", "duplicate", "conflicting_labels"}
    assert {k["group"] for k in KEPT_SIGNALS} == {"G4", "G5", "G6", "G7", "G8"}
    assert len(rules_version()) == 12


# ---- G2: embeds and code
def test_tweet_embed_attribution_removed_but_tweet_text_kept():
    out = clean("Russia stokes every divide. Ben Sasse (@BenSasse) September 28, 2017Spencer responded by writing")
    assert out == "Russia stokes every divide. Ben Sasse Spencer responded by writing"
    out = clean("My statement: pic. twitter. - Tom Steyer (@TomSteyer) May 29, 2017, Utah")
    assert "pic" not in out and "May 29" not in out and "Tom Steyer" in out
    out = clean("A photo posted by Mark Ruffalo (@markruffalo) on Oct 26, 2016 at 3:06pm PDT Ruffalo and Jones went")
    assert out == "Ruffalo and Jones went"


def test_glued_handle_and_empty_brackets():
    assert clean("(@JemaineClement) September 4, 2016@realDonaldTrump funny because") == "funny because"
    assert clean("Ben Sasse ( ) said") == "Ben Sasse said"


def test_embedded_code_removed():
    js = ("Debate:// <![CDATA[ (function(d, s, id) { var js, fjs = d.getElementsByTagName(s)[0]; "
          "if (d.getElementById(id)) return; js = d.createElement(s); js.id = id; "
          "js.src = \"//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.3\"; "
          "fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk')); // ]]>This scene")
    assert clean(js) == "Debate: This scene"


def test_site_furniture_removed():
    assert clean("He left. The post Notes From A Cubs Fan appeared first on Shadowproof .") == "He left."
    assert clean("Done. Post navigation") == "Done."
    assert clean("Ending war. This entry was posted in General . Bookmark the permalink .") == "Ending war."
    assert clean("Story. Please LIKE and SHARE this story on Facebook or Twitter.") == "Story."
    assert clean("Text. Republish our reports, analyses, breaking news and videos (click for details)") == "Text."


def test_media_credits_removed():
    assert clean("Here s the video via YouTube:President Obama spoke.") == "Here s :President Obama spoke."
    assert clean("He announced it via Twitter.") == "He announced it via Twitter."  # ordinary sentence: kept
    assert clean("Nobody does it. Image: Gage Skidmore via Flickr.") == "Nobody does it."
    assert clean("Pic via Instagram. She smiled.") == ". She smiled."


def test_comment_timestamps_and_bylines():
    assert clean("| February 18, 2016 at 7:33 am | Reply Trump's book came out") == "Trump's book came out"
    assert clean("By Daisy Luther Let's talk about Wikileaks.") == "Let's talk about Wikileaks."
    assert clean("By Jesse Lee Peterson President Donald Trump has issued").startswith("President Donald Trump")


def test_21wire_reposts_and_ads():
    s = "It ended. This article was originally published at New Junkie PostREAD MORE 2016 ELECTION NEWS AT: 21st Century Wire 2016 Files"
    assert clean(s) == "It ended."
    assert clean("Hosts. Join us for uncensored, uninterruptible talk radio, custom-made for philosophers. Next") == "Hosts. Next"


# ---- G1: outlet boilerplate
def test_reuters_notes_removed():
    assert clean('He spoke. (This version of the story corrects 17th paragraph to say Scrooge McDuck)') == "He spoke."
    assert clean("Votes fell. (For a graphic of the Czech election click tmsnrt.rs/2vO4hPW) Babis won") == "Votes fell. Babis won"
    s = ("It grew. SAP is the sponsor of this content. It was independently created by Reuters' editorial staff "
         "and funded in part by SAP, which otherwise has no role in this coverage.")
    assert clean(s) == "It grew."


def test_agency_datelines_removed():
    assert clean("LUCCA, Italy (AP) - U. S. Secretary of State Rex Tillerson issued") == "U. S. Secretary of State Rex Tillerson issued"
    assert clean("(CNN) For many Americans") == "For many Americans"


def test_nyt_briefing_removed():
    s = ("Happiness is easy. _____ Your Evening Briefing is posted at 6 p. m. Eastern. And don't miss Your Morning "
         "Briefing, posted weekdays at 6 a. m. Eastern, and Your Weekend Briefing, posted at 6 a. m. Sundays. "
         "Want to look back? Here's last night's briefing.")
    assert clean(s) == "Happiness is easy."


def test_breitbart_promos_and_bios_removed():
    assert clean("He voted. Breitbart News Daily airs on SiriusXM Patriot 125 weekdays from 6:00 a. m. to 9:00 a. m. Eastern.") == "He voted."
    s = ("It grows. Charlie Nash is a reporter for Breitbart Tech. You can follow him on Twitter @MrNashington "
         "or like his page at Facebook.")
    assert clean(s) == "It grows."
    s = "Unchallenged. You can follow Ben Kew on Facebook, on Twitter at @ben_kew, or email him at bkew@breitbart. com"
    assert clean(s) == "Unchallenged."


def test_content_is_kept():
    s = ("Mr. Trump said on Thursday that the U.S. would act. Officials told Reuters the plan was ready. "
         "The statement was posted on his website.")
    assert clean(s) == s  # house style (G4) and outlet names in sentences (G6) are kept


# ---- G3: language filter
@pytest.mark.parametrize("text, expected", [
    ("Политика Победив на президентских выборах, Дональд Трамп стал самой яркой звездой", True),
    ("اجتماع لمجلس الأمن من 28 أكتوبر 2016 مذكرة مفاهيمية عن رئاسة الروسية", True),
    ("Las imágenes libres de derechos más destacadas de la semana, el black friday y la lotería de navidad "
     "que todos los años se celebra en el país con una gran fiesta para las familias", True),
    ("The president said on Thursday that the bill, which passed the House, would be signed. He added that it "
     "was a good day for the country and its people.", False),
    ("Café owners in Zürich said the naïve plan was a disaster for everyone who works there and has a family", False),
    ("Watch:", False),  # too short to decide -> kept
])
def test_language_filter(text, expected):
    assert is_non_english(text) is expected


# ---- speed-up pre-checks must never change the output
@pytest.mark.parametrize("marker", [m for m in MARKERS if m.triggers], ids=lambda m: m.name)
def test_documented_example_contains_a_trigger(marker):
    low = normalize(marker.example).lower()
    assert any(t in low for t in marker.triggers)


def test_triggers_do_not_change_output_on_real_data():
    from fake_news import config
    from fake_news.data.load import load_raw

    if not config.RAW_CSV.exists():
        pytest.skip("raw data not downloaded")
    texts = load_raw()["text"].fillna("").sample(4000, random_state=0).map(normalize)
    for s in texts:
        assert remove_text_markers(s) == remove_text_markers(s, use_triggers=False)


# ---- round-2 residue rules (second pass)
def test_round2_residue_rules():
    assert clean("Loser.com. SAP is the sponsor of this coverage which is independently produced by the staff of "
                 "Reuters News Agency.") == "Loser.com."
    s = ("The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, "
         "and . The opinions expressed are his own. Reuters has not edited the statements or confirmed their "
         "accuracy. : - No deal was made")
    assert clean(s) == ": - No deal was made"
    assert clean("Good evening. Here's the latest. Senators voted.") == "Senators voted."
    assert clean("Good morning. (Want to get California Today by email? Sign up.) Let's turn") == "Let's turn"
    assert clean("Home » Headlines » World News » Breaking: Weiner cooperates") == "Breaking: Weiner cooperates"
    assert clean("Zone Matt Agorist October 27, 2016 2 Comments Cuyahoga Falls") == "Zone Matt Agorist Cuyahoga Falls"
    assert clean("Published: 23 mins ago (DAILYCALLER) - Tens of thousands") == "Tens of thousands"
    assert clean("Sun, 23 Oct 2016 00:00 UTC Heather Ann Thompson's book") == "Heather Ann Thompson's book"
    assert clean("Explained. For entire story: Infowars") == "Explained."
    assert clean("Margin By Michael Snyder, on November 9th, 2016 The 2016 election") == "Margin The 2016 election"
    assert clean("Pruitt said. Read more: NYP") == "Pruitt said."
    assert clean("Bob Price serves as associate editor for Breitbart Texas. He is a founding member of the "
                 "Breitbart Texas team.") == ""
    assert clean("on November 8, 2016, voters chose") == "on November 8, 2016, voters chose"  # date in a sentence: kept
    assert clean("What is your comment on this?") == "What is your comment on this?"  # content: kept


def test_round2_final_pass():
    from fake_news.data.clean import remove_title_markers
    assert clean("x. Good evening. Here's the latest. 1. The challenges grew.") == "x. 1. The challenges grew."
    assert remove_title_markers("U.S. Senate, Orlando, Donald Trump: Your Tuesday Evening Briefing") == "U.S. Senate, Orlando, Donald Trump"
    assert clean("Beware: reheating foods Thursday, October 27, 2016 by: Isabelle Z. Tags: reheating , toxic food , "
                 "foodborne illness Nothing beats a hot meal") == "Beware: reheating foods Nothing beats a hot meal"
    assert clean("the law By John W. Whitehead - October 27, 2016 \"Today the path\"") == "the law \"Today the path\""
    assert clean("Alex Jones | Infowars.com - October 27, 2016 Comments Alex Jones breaks down") == "Alex Jones | Infowars.com - Alex Jones breaks down"
