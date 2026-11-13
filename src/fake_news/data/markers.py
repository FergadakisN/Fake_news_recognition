"""Registry of every cleaning rule and cleaning decision, with its rationale and evidence.

This module is the single place that documents *what* the cleaning removes and *why*, so the
report can cite it. Each build also writes a log of how often every rule fired
(``reports/cleaning/``, see :func:`fake_news.data.clean.build_clean_dataset`), and the full
decision record is in ``reports/cleaning/DECISIONS.md``.

Principle: **remove what identifies the outlet or the scraping process; keep what the author
wrote.** WELFake merges several source datasets, so text that identifies the source
(datelines, bylines, newsletter boilerplate, embed residue, ...) reveals the label without
saying anything about the news itself.

How the rules were found
------------------------
- ``round1``: manual inspection of article starts/ends, titles and known patterns.
- ``round2``: systematic leak scan on the training split (``notebooks/02-data-cleaning.ipynb``): every phrase
  occurring in >= 50 articles that is >= 98% one class (counting only). Decisions approved by the
  group's data owner (Person 1) on 2026-11-12.

Groups
------
- ``R1``: round-1 source markers.
- ``G1``: round-2 outlet boilerplate (mostly in real articles: NYT, Reuters, CNN/AP/AFP, Breitbart).
- ``G2``: round-2 scraping artifacts (mostly in fake articles: embeds, site furniture, code).
- ``G3``: round-2 row filter: non-English articles are dropped (see :data:`ROW_FILTERS`).
- ``G4``-``G8``: signals deliberately **kept** (see :data:`KEPT_SIGNALS`).

Order matters: rules run top to bottom on the normalized text. Rules anchored on handles,
emails or links run before the generic handle/email/link rules remove those anchors.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Marker:
    name: str
    field: str        # "text" or "title"
    group: str        # R1, G1, G2 (see module docstring)
    source: str       # where the marker comes from (outlet or origin)
    description: str
    example: str      # a real (shortened) example from WELFake
    pattern: re.Pattern
    replacement: str = " "
    # Speed-up only: the rule runs only if the lower-cased text contains one of these substrings
    # (every possible match contains one). Empty = always run. Tests check the output is unchanged.
    triggers: tuple[str, ...] = ()


_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
          r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE = rf"{_MONTH}\s+\d{{1,2}},\s+\d{{4}}"
_TIME = r"\d{1,2}:\d{2}\s*[ap]\.?\s?m\.?(?:\s*(?:PDT|PST|EDT|EST|CDT|CST|UTC|GMT))?"
_SENT = r"(?:^|(?<=[.!?)\"]))\s*"  # start of text or of a sentence (cheap lookbehind)
# one capitalized name word (no apostrophe), not a function word or title
_NAME = (r"(?!(?:The|A|An|In|On|At|For|And|But|This|That|It|He|She|We|They|I|President|Senator|Governor|Mr|Mrs|Ms|Dr)\b)"
         r"[A-Z][A-Za-z-]*\.?")

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

TITLE_MARKERS: list[Marker] = [
    Marker("nyt_briefing_title", "title", "G1", "New York Times (real)", "Newsletter name used as title.",
           "U.S. Senate, Orlando, Donald Trump: Your Tuesday Evening Briefing",
           re.compile(r":?\s*\byour (?:\w+ )?(?:morning|evening|weekend) briefing\b:?|^new york tonight:\s*", re.I), ""),
    Marker("title_tag", "title", "R1", "fake sites", "All-caps tags in brackets.",
           "Trump Rants At Reporters (VIDEO)", re.compile(r"\s*[\[(][A-Z0-9][A-Z0-9 /&,!'+-]*[\])]"), ""),
    Marker("title_source_suffix", "title", "R1", "NYT, Breitbart, fake sites", "Outlet name appended to the title.",
           "Senate passes bill - The New York Times",
           re.compile(r"\s+[-|·—–]\s+(?:" + "|".join(map(re.escape, TITLE_SOURCES)) + r")\s*$"), ""),
]

TEXT_MARKERS: list[Marker] = [
    # ---- G2: embedded code and social-media embeds (before links/handles are removed)
    Marker("embedded_code", "text", "G2", "fake sites", "Scraped JavaScript (Facebook SDK loader, CDATA blocks).",
           "(function(d, s, id) { var js, fjs = d.getElementsByTagName(s)[0]; ... 'facebook-jssdk'));",
           re.compile(r"(?://\s*)?(?:<!\[CDATA\[\s*)?\(function\s*\(d,\s*s,\s*id\).{0,700}?['\"]facebook-jssdk['\"]\)\);?"
                      r"(?:\s*//\s*\]\]>)?|//\s*<!\[CDATA\[.{0,1500}?//\s*\]\]>")),
    Marker("tweet_embed", "text", "G2", "embedded tweets/Instagram (both classes, different formats)",
           "Attribution line of an embedded post: (@handle) + date [+ time]. The post text itself is kept.",
           "Ben Sasse (@BenSasse) September 28, 2017Spencer responded...",
           re.compile(rf"(?i:an? (?:photo|video|post) (?:posted|shared) by [^()]{{0,60}}?)?\(\s*@\w+\s*\)\s*(?:on\s+)?{_DATE}"
                      rf"(?:\s+at\s+{_TIME})?")),
    Marker("pic_twitter_residue", "text", "G2", "Breitbart (real)", "Broken pic.twitter.com links of embedded tweets.",
           "...not in this new bill from the House GOP pic. twitter. During...",
           re.compile(r"\bpic\.\s?twitter\.(?:\s?com/\S*)?", re.I)),
    # ---- R1: datelines and agency credits
    Marker("reuters_dateline", "text", "R1", "Reuters (real)", "Dateline at the start of Reuters articles.",
           "WASHINGTON (Reuters) - The U.S. Senate voted...", re.compile(r"^\s*.{0,120}?\(Reuters\)\s*-+\s*")),
    Marker("reuters_tag", "text", "R1", "Reuters (real)", "Any remaining (Reuters) tag.",
           "He told (Reuters) that...", re.compile(r"\(Reuters\)")),
    Marker("agency_dateline", "text", "G1", "CNN, AP, AFP (mostly real)", "Dateline/tag of other agencies.",
           "(CNN) For many Americans across the country...",
           re.compile(r"^\s*[^()]{0,80}?\((?:CNN|AP|AFP)\)\s*-*\s*|\((?:CNN|AP|AFP)\)"
                      r"|\((?i:naturalnews|dailycaller|daily caller)\)\s*-*")),
    Marker("agency_credits", "text", "R1", "Reuters, AP (real)", "Reporting/editing credits and the SAP sponsor note.",
           "(Reporting by Jane Doe; Editing by John Roe)",
           re.compile(r"\((?:additional )?(?:reporting|writing|editing|compiled)\s+by\s[^)]{0,250}\)"
                      r"|this article was funded in part by.{0,250}?creation or production\."
                      r"|(?:^|(?<=[.!?] ))[^.]{0,80}?\bcontributed to this report\.", re.I)),
    # ---- G1: outlet boilerplate (before links/emails/handles are removed)
    Marker("reuters_notes", "text", "G1", "Reuters (real)", "Correction notes, graphic links and sponsor disclaimers.",
           "(This version of the story corrects 17th paragraph ...) / For graphic on X, click tmsnrt.rs/2vO4hPW",
           re.compile(r"\((?:this (?:version of the )?story|refiles?|corrects?|updates? with)\b[^)]{0,250}\)"
                      r"|\(?for (?:a |an )?(?:graphic|interactive)s?\b[^)]{0,200}?(?:click|see)\b\s*:?\s*(?:here\b|\S+)\)?"
                      rf"|{_SENT}[\w&' -]{{1,40}}? is the sponsor of this content\..{{0,250}}?"
                      r"(?:no role in this coverage|involvement in its creation or production)\."
                      rf"|{_SENT}[\w&' -]{{1,40}}? is the sponsor of this coverage[^.]{{0,200}}\."
                      rf"|{_SENT}[^.]{{0,60}}?\bindependently (?:created|produced) by (?:the )?reuters\b[^.]{{0,200}}\.", re.I)),
    Marker("reuters_tweet_disclaimer", "text", "G1", "Reuters (real)", "Disclaimer above Reuters' Trump-tweet factboxes.",
           "The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, "
           "and . The opinions expressed are his own. Reuters has not edited the statements or confirmed their accuracy.",
           re.compile(r"the following statements were posted to the verified twitter accounts?.{0,250}?"
                      r"confirmed their accuracy\.", re.I)),
    Marker("nyt_briefing", "text", "G1", "New York Times (real)", "Newsletter boilerplate of the NYT 'Briefing' articles.",
           "Your Evening Briefing is posted at 6 p.m. Eastern. And don't miss Your Morning Briefing, ...",
           re.compile(r"your (?:morning|evening|weekend) briefing is posted.{0,120}?\b(?:eastern|sundays)\."
                      r"|and don't miss your (?:morning|evening|weekend) briefing.{0,200}?\b(?:eastern|sundays)\."
                      r"|want to look back\? here's [^.]{0,40}?briefing\."
                      r"|photographs may appear out of order for some readers\. viewing this version of the briefing should help\."
                      r"|what (?:did you|would you)[^?]{0,100}\?[^.]{0,120}?briefing@nytimes\.\s?com\.?"
                      r"|(?:let us know|contact us) at briefing@nytimes\.\s?com\.?"
                      r"|^\s*good (?:morning|evening)\.(?:\s*here's (?:the latest|what you need to know)[^.]{0,40}\.)?"
                      r"|\bgood (?:morning|evening)\.\s*here's (?:the latest|what you need to know)[^.]{0,40}\."
                      r"|\bgood evening on this \w+\.\s*this is new york tonight[^.]{0,120}\."
                      r"|\(?want to get [^?]{1,40}? by email\?[^)]{0,60}\)?", re.I)),
    Marker("breitbart_promo", "text", "G1", "Breitbart (real)", "Radio promos, author bios and book promos.",
           "Charlie Nash is a reporter for Breitbart Tech. / Breitbart News Daily airs on SiriusXM Patriot 125 ...",
           re.compile(r"(?i:breitbart news daily airs on siriusxm patriot.{0,80}?eastern\.?)"
                      r"|(?i:(?:broadcast live )?on siriusxm patriot(?: channel)? 125.{0,40}?eastern)"
                      rf"|{_SENT}(?:[A-Z][\w.'-]+\s){{1,3}}?[A-Z][\w.'-]+\s(?:is|was|serves as)\s(?:(?:a|an|the)\s)?[^.]{{0,120}}?"
                      r"\b(?:for|at)\s(?i:breitbart)\b[^.]{0,80}\."
                      r"|(?i:he was named one of the \"most influential\" people in news media in 2016\.)"
                      r"|(?i:his (?:new |latest )?book,?[^.]{0,150}?is available from regnery[^.]{0,40}\.)"
                      r"|(?i:he is also the political analyst for armed american radio\.)"
                      r"|(?i:reach (?:him|her) directly at)"
                      rf"|{_SENT}(?i:(?:he|she) is (?:also )?(?:a|an|the) [^.]{{0,80}}\bbreitbart\b[^.]{{0,60}}\.)")),
    Marker("21wire_acr", "text", "G2", "21st Century Wire (fake)", "Repost notes, 'READ MORE ... Files' lines and ACR radio ads.",
           "READ MORE 2016 ELECTION NEWS AT: 21st Century Wire 2016 Files / Join us for uncensored, uninterruptible talk radio...",
           re.compile(r"read more [^:]{0,80}?(?:news )?at:?\s*21st century wire[^.]{0,60}?files"
                      r"|join us for uncensored, uninterruptible talk radio[^.]{0,300}\.?"
                      r"|join acr hosts [^.]{0,250}\.?"
                      r"|(?:this article was )?originally published (?:at|on|by) [^.]{0,80}?(?=read more|\.|$)"
                      r"|continue this story at [^.]{0,60}?(?=read more|\.|$)", re.I)),
    # ---- R1: generic boilerplate
    Marker("21wire_boilerplate", "text", "R1", "21st Century Wire (fake)", "Subscription line and outlet name.",
           "SUPPORT OUR WORK BY SUBSCRIBING & BECOME A MEMBER @21WIRE.TV",
           re.compile(r"(?:support our work by )?subscrib(?:e|ing)\s*&\s*becom(?:e|ing) a member"
                      r"|tune in to the alternate current radio network[^.]{0,200}\.?"
                      r"|\b21st century wire\b|@?\s?\b21wire(?:\.tv)?\b", re.I)),
    Marker("scraper_boilerplate", "text", "R1", "various", "Share widgets and newsletter prompts.",
           "Get short URL / Share on Facebook Share on Twitter / Want to get this briefing by email?",
           re.compile(r"get short url|share on facebook(?: share on twitter)?|next prev swipe left/right"
                      r"|you are here: home|\(?want to get this briefing by email\?[^)]{0,60}\)?"
                      r"|for entire article click link|\bwe are change\b", re.I)),
    Marker("blog_footer", "text", "G2", "fake sites", "Blog-software footers and navigation.",
           "The post X appeared first on Shadowproof. / This entry was posted in General. Bookmark the permalink.",
           re.compile(r"the post\b.{0,250}?\bappeared first on\b[^.]{0,80}\.?"
                      r"|this entry was posted in [^.]{0,80}\.(?:\s*bookmark the permalink\s?\.?)?|bookmark the permalink\s?\.?"
                      r"|post navigation|related (?:posts|articles):?(?: no related posts)?", re.I)),
    Marker("comment_widgets", "text", "G2", "fake sites", "Comment forms and widgets.",
           "Leave a Reply Click here to get more info on formatting / FaceBook Comments You must be logged in to post a comment",
           re.compile(r"leave a (?:reply|comment)|click here to get more info on formatting"
                      r"|\(?1\)? leave the name field empty if you want to post as anonymous\.?|\badd (?:a )?comment\b"
                      r"|(?:add to the conversation using )?facebook comments|you must be logged in to post a comment(?: login)?"
                      r"|you can leave a response,? or trackback to this entry", re.I)),
    Marker("share_donate", "text", "G2", "fake sites", "Share, print, subscribe, donate and 'read more' calls to action.",
           "Please share this story / Republish our reports, analyses, breaking news and videos (click for details)",
           re.compile(r"\bprint this(?: article| page| post)?\b|\bprint email\b"
                      r"|(?:please )?(?:like and )?share this (?:story|article|post)\b[^.!]{0,60}[.!]?|please share\b[^.!]{0,80}[.!]?"
                      r"|(?:support the trump movement and )?help us fight liberal media bias\.?"
                      r"|republish our reports[^.(]{0,120}(?:\(click for details\))?|be sociable,? share!"
                      rf"|{_SENT}[^.]{{0,120}}\bdonate page\b[^.]{{0,60}}\.?|\bdonate!"
                      r"|by subscribing to [^.]{0,200}\.|help support us by subscribing[^.]{0,60}|privacy policy"
                      r"|(?:by )?clicking here\b:?"
                      r"|\bread more\b:?[^.!?]{0,80}(?:[.!?]|$)|\bfor (?:the )?entire (?:story|article)\b:?[^.]{0,60}", re.I)),
    Marker("license_lines", "text", "G2", "fake sites", "Republication licenses, disclaimers, obfuscated emails.",
           "You have permission to republish this article under a Creative Commons license with full attribution",
           re.compile(r"the views expressed herein are the views of the author exclusively[^.]{0,250}\."
                      r"|\[?email(?:\s|&#160;)protected\]?"
                      r"|this (?:work|article)\b[^.]{0,150}?licensed under (?:a )?creative commons[^.]{0,150}\.?"
                      r"|you have permission to republish this article[^.]{0,200}"
                      r"|under (?:a )?creative commons[ \w.]{0,30}?licen[cs]e(?:\s[\d.]+)?", re.I)),
    Marker("daily_sheeple", "text", "G2", "The Daily Sheeple (fake)", "Repost boilerplate of The Daily Sheeple.",
           "Delivered by The Daily Sheeple We encourage you to share and ( Click for details ). Contributed by X of The Daily Sheeple .",
           re.compile(r"delivered by the daily sheeple|we encourage you to share and \(?\s*click for details\s*\)?\s?\.?"
                      r"|contributed by [^.]{0,60}? of the daily sheeple\s?\.?", re.I)),
    Marker("breadcrumbs", "text", "G2", "fake sites", "Navigation breadcrumbs.",
           "Home » Headlines » World News » Breaking: ...", re.compile(r"(?:[\w&' -]{1,30}»\s*){1,6}")),
    Marker("post_dates", "text", "G2", "fake sites",
           "Publication dates in page-header context ('Posted on <date> by X', '| <date>', '<date> 2 Comments'). "
           "Dates inside sentences are kept (G5).",
           "Matt Agorist October 27, 2016 2 Comments",
           re.compile(rf"(?i:\bposted on\s+){_DATE}(?i:\s+by\s+\S+)?|\|\s*{_DATE}(?:\s+at\s+{_TIME})?(?:\s*\|\s*(?i:reply))?"
                      rf"|{_DATE}\s+(?:\d+\s+)?(?i:comments?)\b|\b\d[\d,]*\s+(?i:views)\s+(?={_MONTH})"
                      rf"|(?:(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s+)?{_DATE}\s+by:\s+[^:]{{0,40}}?\s+Tags:\s+"
                      r"(?:[\w'-]+(?:\s[\w'-]+){0,3}\s?,\s+){0,10}[\w'-]+(?:\s[a-z][\w'-]*){0,3}")),
    Marker("timestamps", "text", "G2", "fake sites", "Relative and UTC timestamps.",
           "Published: 23 mins ago / Sun, 23 Oct 2016 00:00 UTC",
           re.compile(r"(?:published:?\s*)?\b\d+\s+(?:mins?|minutes|hours?)\s+ago\b"
                      r"|\b(?:mon|tue|wed|thu|fri|sat|sun),?\s+\d{1,2}\s+[a-z]{3}\s+\d{4}\s+\d{1,2}:\d{2}\s*(?:utc|gmt)\b", re.I)),
    Marker("header_credits", "text", "G2", "fake sites", "Photo copyright lines and 'Fort Russ' translation headers.",
           "© Naif Rahma / Reuters / November 20, 2016 - Fort Russ - ... - translated by J. Arnoldski -",
           re.compile(r"©\s*(?:[A-Z][\w.-]*\s+){1,4}/\s*[A-Z][\w-]*"
                      r"|(?i:\bfort russ(?: news)?\b\s*-?|translated by\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3}\s*-?)")),
    Marker("contact_lines", "text", "G1", "Breitbart (real), various",
           "Author contact sentences ('You can follow/email ...'), including their handles and emails.",
           "You can follow him on Twitter @MrNashington or like his page at Facebook.",
           re.compile(r"\byou\s+can\s+(?:\w+\s+){0,3}?(?:follow|reach|email|contact|like|find)\b"
                      r"(?:[^.]|\.(?=\s?(?:com|org|net)\b)){0,200}(?:\.|$)"
                      r"|\bor\s+email\s+(?:the\s+author|him|her|them|us|me)\s+at\b[\s,.]*"
                      r"|(?<=[.!?])\s*and facebook\.", re.I)),
    Marker("social_byline", "text", "R1", "Breitbart (real), various", "'Follow X on Twitter' bylines.",
           "Follow Pam Key on Twitter @pamkeyNEN",
           re.compile(r"\b(?:follow|like) (?:\w+ ){0,3}on (?:twitter|facebook)\b(?: at)?[!.]?", re.I)),
    # ---- R1: links, emails, handles
    Marker("url", "text", "R1", "various (mostly fake)", "URLs and common short links.",
           "https://example.com/a / bit.ly/2jBh4LU",
           re.compile(r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+(?:com|org|net|ly|co|tv|me)/\S*", re.I)),
    Marker("other_links", "text", "G1", "Reuters (real), various", "Links with any other domain (tmsnrt.rs, reut.rs, ...).",
           "click tmsnrt.rs/2fjdh4j", re.compile(r"\b(?:[\w-]+\.)+[a-z]{2,6}/\S*", re.I)),
    Marker("broken_links", "text", "G1", "Breitbart (real), various", "Broken links such as 'https: //t.co/...'.",
           "#MayorsStand4All https: . - Mayor Adler", re.compile(r"\bhttps?:\s*(?://\S*)?", re.I)),
    Marker("email", "text", "R1", "NYT, Breitbart (real)", "Email addresses, including the scraped 'name@site. com' form.",
           "briefing@nytimes. com",
           re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.\s?(?:com|org|net|edu|gov|uk|tv|us|io|info)\b", re.I)),
    Marker("twitter_handle", "text", "R1", "various", "@handles.", "Ask @realDonaldTrump now.", re.compile(r"(?<!\w)@\w+")),
    Marker("glued_handle", "text", "G2", "embedded tweets (fake)", "@handles glued to the previous word.",
           "September 4, 2016@realDonaldTrump funny because...", re.compile(r"@\w+")),
    # ---- R1 + G2: image and media credits
    Marker("image_credit", "text", "R1", "fake sites", "Image credit at the end of the text.",
           "Featured image via Chip Somodevilla/Getty Images",
           re.compile(r"(?:featured\s+)?(?:image|photo|picture)s?\s+(?:via|credit|courtesy)\b.{0,120}$", re.I)),
    Marker("featured_image", "text", "R1", "fake sites", "Any remaining 'Featured image ...' credit.",
           "Featured image is a screenshot", re.compile(r"\bfeatured image\b[^.]{0,120}\.?", re.I)),
    Marker("photo_by_getty", "text", "R1", "fake sites", "'Photo by X/Getty Images' credits.",
           "Photo by Win McNamee/Getty Images.", re.compile(r"photo(?:graph)?\s+by\s+[^.]{0,60}?/\s*getty images\.?", re.I)),
    Marker("getty_images", "text", "R1", "fake sites", "Any remaining 'Getty Images'.", "Getty Images",
           re.compile(r"\bgetty images\b", re.I)),
    Marker("media_credit_inline", "text", "G2", "fake sites", "Inline media credits: '<media> via <Source>'.",
           "Here s the video via YouTube:",
           re.compile(r"(?i:\b(?:the\s+)?(?:video|image|photo|picture|pic|clip|footage|screenshot|segment|interview)s?"
                      r"\s+(?:below\s+)?via:?\s+)@?\w[\w@-]*(?:\.\w[\w@-]*)*(?:\s+[A-Z][\w-]*){0,3}"
                      r"|(?i:\bbelow via:?\s+)[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2}|(?i:\bvia flickr\b)"
                      r"|(?i:\b(?:image|photo|picture)\s+credits?\s*:?)")),
    Marker("media_credit_end", "text", "G2", "fake sites",
           "A final line that starts with a credit word: 'Image:', 'Photo credit:', 'H/T', 'Via ...'.",
           "Image: Gage Skidmore via Flickr.",
           re.compile(r"(?:^|(?<=[.!?:]))\s*(?i:(?:featured\s+)?(?:image|photo|picture|pic|video|screenshot)s?"
                      r"(?:\s+(?:credit|source))?\s*:|h/t\b|hat tip\b|via\b)[^.!?]{0,120}[.!]?\s*$")),
    # ---- G2: scraped comments, bylines; G1: separators; cleanup
    Marker("comment_timestamp", "text", "G2", "scraped comment sections (fake)", "Comment timestamps and 'Reply' links.",
           "| February 18, 2016 at 7:33 am | Reply",
           re.compile(rf"\|?\s*{_DATE}\s+at\s+{_TIME}(?:\s*\|\s*(?i:reply))?")),
    Marker("byline", "text", "G2", "fake sites",
           "'By Author Name' (2-3 name words) at the start of the text or of a sentence, or anywhere when "
           "followed by ', on <date>'.",
           "By Daisy Luther Let's talk about Wikileaks.",
           re.compile(rf"(?:^|(?<=[.!?\"]))\s*By\s+(?:{_NAME}\s+){{1,2}}{_NAME}(?![\w'])(?:,?\s+on\s+{_DATE})?"
                      rf"|\bBy\s+(?:{_NAME}\s+){{1,2}}{_NAME},?\s+on\s+{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
                      rf"|\bBy\s+(?:{_NAME}\s+){{1,2}}{_NAME}\s+-\s+{_DATE}")),
    Marker("nyt_separator", "text", "G1", "New York Times (real)", "Underscore separator lines.",
           "_____ 2. Mr. Trump, who was named...", re.compile(r"_{3,}")),
    Marker("empty_brackets", "text", "G2", "cleanup", "Brackets emptied by the rules above (e.g. removed handles).",
           "Ben Sasse ( ) September 28", re.compile(r"\(\s*[,.;:]?\s*\)|\[\s*\]")),
]

_TRIGGERS = {
    "embedded_code": ("facebook-jssdk", "cdata"),
    "tweet_embed": ("@",),
    "pic_twitter_residue": ("pic.",),
    "reuters_dateline": ("(reuters)",),
    "reuters_tag": ("(reuters)",),
    "agency_dateline": ("(cnn)", "(ap)", "(afp)", "(naturalnews)", "(dailycaller)", "(daily caller)"),
    "reuters_tweet_disclaimer": ("verified twitter",),
    "broken_links": ("http",),
    "breadcrumbs": ("»",),
    "post_dates": ("posted on", "|", "comment", "views", "tags:"),
    "timestamps": ("ago", "utc", "gmt"),
    "header_credits": ("©", "fort russ", "translated by"),
    "agency_credits": ("(reporting", "(additional", "(writing", "(editing", "(compiled", "funded in part",
                       "contributed to this report"),
    "reuters_notes": ("(this", "(refile", "(correct", "(update", "graphic", "interactive", "sponsor", "independently"),
    "nyt_briefing": ("briefing", "good morning", "good evening", "by email"),
    "breitbart_promo": ("siriusxm", "breitbart", "most influential", "regnery", "armed american radio", "directly at"),
    "21wire_acr": ("21st century wire", "uninterruptible", "acr hosts", "originally published", "continue this story"),
    "21wire_boilerplate": ("member", "alternate current", "21st century wire", "21wire"),
    "scraper_boilerplate": ("short url", "share on facebook", "swipe", "you are here", "briefing by email",
                            "click link", "we are change"),
    "blog_footer": ("appeared first", "posted in", "permalink", "post navigation", "related"),
    "comment_widgets": ("leave a", "more info on formatting", "name field", "comment", "trackback"),
    "share_donate": ("print", "share", "liberal media bias", "republish", "donate", "subscrib", "privacy policy",
                     "clicking here", "read more", "entire"),
    "license_lines": ("views expressed herein", "protected", "creative commons", "permission to republish"),
    "daily_sheeple": ("daily sheeple", "click for details"),
    "contact_lines": ("you can", "or email", "and facebook"),
    "social_byline": ("on twitter", "on facebook"),
    "url": ("http", "www.", "/"),
    "other_links": ("/",),
    "email": ("@",),
    "twitter_handle": ("@",),
    "glued_handle": ("@",),
    "image_credit": ("image", "photo", "picture"),
    "featured_image": ("featured image",),
    "photo_by_getty": ("getty",),
    "getty_images": ("getty",),
    "media_credit_inline": ("via", "credit"),
    "media_credit_end": (":", "h/t", "hat tip", "via"),
    "comment_timestamp": (":",),
    "nyt_separator": ("___",),
    "empty_brackets": ("(", "["),
}
TEXT_MARKERS = [replace(m, triggers=_TRIGGERS.get(m.name, ())) for m in TEXT_MARKERS]

MARKERS: list[Marker] = TITLE_MARKERS + TEXT_MARKERS

# Row filters, in the order they are applied (a dropped row gets the first matching reason).
ROW_FILTERS = {
    "empty_text": "Text is empty after normalization and marker removal (782 of the 783 originally empty texts are fake).",
    "non_english": "Not English (G3): >30% of letters outside the Latin script (Cyrillic, Arabic, ...), or, for texts "
                   "with >= 20 words, more Spanish/French/German/Italian/Portuguese function words than English ones. "
                   "Almost all such articles are fake, so they reveal the label.",
    "duplicate": "Exact duplicate of the cleaned content of an article with a lower doc_id (the lowest is kept).",
    "conflicting_labels": "The same cleaned content appears with both labels (all copies dropped).",
}

# Signals found by the round-2 leak scan and deliberately KEPT, with the reason (for the report).
KEPT_SIGNALS = [
    {"group": "G4", "name": "outlet house style",
     "examples": "'Mr./Ms./Mrs.' honorifics (NYT; 4,388 real vs 1,548 fake training articles), 'said on Thursday' "
                 "(Reuters), 'U.S.' vs 'US'",
     "reason": "Written by the author, not boilerplate. Documented as a limitation: part of the remaining signal "
               "is the house style of the outlets that make up each class."},
    {"group": "G5", "name": "topic and time",
     "examples": "Rohingya, Catalonia, Brexit (real, 2017) vs Bundy, DAPL, 'sheeple' (fake, 2016); dates in Oct-Nov 2016",
     "reason": "This is the topic overlap the research question is about; the cluster-held-out split measures it. "
               "Removing it would hide the effect we want to quantify."},
    {"group": "G6", "name": "outlet names inside sentences",
     "examples": "'told Reuters' (1,911 real vs 49 fake training articles), 'Breitbart reported'",
     "reason": "Part of the sentence the author wrote; bylines and promos outside the article body are removed (G1)."},
]

KEPT_SIGNALS.append(
    {"group": "G7", "name": "very short texts",
     "examples": "cleaned texts under 20 words: 32 real vs 353 fake",
     "reason": "Decided 2026-11-12: kept. They are the articles as published (teasers, stubs), not a source "
               "marker; dropping them would select on a property that correlates with the label."})

KEPT_SIGNALS.append(
    {"group": "G8", "name": "title labels",
     "examples": "'WATCH:' (98% fake), 'BREAKING:' (100% fake), 'Factbox:' (0% fake): 22 editorial labels, "
                 "2,216 training titles",
     "reason": "Decided 2026-11-12: kept as part of the published headline. Documented (notebooks/02-data-cleaning) "
               "so their effect can be discussed in the report, like house style (G4)."})

# Findings not yet decided (none at the moment).
PENDING: list[dict] = []


def rules_version() -> str:
    """Short hash of all rule patterns and their order: changes whenever any rule changes."""
    h = hashlib.sha256()
    for m in MARKERS:
        h.update(f"{m.field}|{m.name}|{m.pattern.pattern}|{m.pattern.flags}|{m.replacement}\n".encode())
    return h.hexdigest()[:12]
