"""The sentiment corpus, kept in one place and in clean UTF-8.

The texts live here rather than inside the analysis code or the experiment for two
reasons. The first is encoding. The copies that reached this repository had been
round-tripped through a lossy encoding at some point, so pound signs and typographic
apostrophes had been replaced by U+FFFD, the replacement character. A corpus with
U+FFFD in it silently changes what a tokeniser sees, and the damage is easy to
reintroduce by editing the text in the wrong editor, so the repaired text sits in one
module and a test asserts that no replacement character survives anywhere in it.

The second reason is provenance. Two of these texts are real market events rather than
teaching examples, and a sentiment score means nothing without knowing what was said,
by whom, and when. The provenance is recorded beside each text.

House style bans em and en dashes, so where the source used a dash as punctuation this
copy uses a comma. That changes no word and no sentiment-bearing token.
"""

from __future__ import annotations

#: The teaching example from the unit notebook: a short, invented MacBook Pro review
#: with one clear opinion per sentence. Useful because the right answer is obvious.
SHORT_REVIEW = (
    "I feel the latest laptop from Mac is really good overall. It has amazing resolution. The "
    "computer is really very sleek and can slide into bags easily. However, I feel the weight is "
    "a let down. The price is a bit expensive given the configurations (I would expect an SSD "
    "storage), however the processor seems really good."
)

#: A real MacBook review, from T3: https://www.t3.com/reviews/new-macbook-review
#: Long, mixed in sentiment and full of qualified praise, which is exactly the shape of
#: text that a single document-level polarity score flattens into meaninglessness.
LONG_REVIEW = (
    "MacBook review: The most portable MacBook ever is far from the most practical. The new "
    "MacBook crams more interesting tech than you can shake a USB (Type-C) stick at into its "
    "skinny 12-inch frame. It's a dazzling device for sure, one that would set tongues wagging "
    "down at the pub in a heartbeat, but whether it would slot into your life without causing a "
    "fuss is an entirely different matter. OS X 10.11 El Capitan features: what's new and what "
    "matters The thing is, the new MacBook isn't a MacBook Air with a Retina display, which is "
    "what many people on the street were expecting from Apple. While it borrows the MacBook Air's "
    "tapered design, along with the MacBook Pro with Retina's high-resolution display and skinny "
    "black bezel, its shallow keyboard, Intel Core M processor and single USB Type-C port "
    "position it as an entirely new category of laptop. I'm not convinced that power-hungry "
    "MacBook owners will rush out to trade their machines for one, but if you're looking for a "
    "laptop that's as portable as an iPad and runs OS X, the new MacBook is a luxurious, albeit "
    "flawed option. A fashion item with a designer price tag to match, it starts at £1,049 for "
    "the entry-level model with 256GB of storage, rising to £1,299 for the more powerful 512GB "
    "configuration. New MacBook: Size and build Like Dell's Windows-powered XPS 13, the new "
    "MacBook slims down the display's bezel to accommodate more screen. Despite being 12 inches, "
    "its footprint is almost identical to the 11-inch MacBook Air, and at 2.03 pounds (versus the "
    "Air's 2.38 pounds) barely registers when slung into a backpack for transportation. Measuring "
    "just 30 x 19.2 x 1.7cm (W x D x H), the new MacBook is a whisker longer and wider than the "
    "11-inch Air (28 x 19.7 x 1.31cm) while shaving off around half a millimetre in height. It's "
    "crafted out of a durable aluminium that stands up well to knocks and scrapes. New MacBook: "
    "Features One of Apple's more contentious decisions was to give the new MacBook just one USB "
    "Type-C port. Smaller than a standard USB port and reversible, USB Type-C has been hailed as "
    "the future because it provides power, a connection to an external display (through "
    "DisplayPort, HDMI or VGA) and a USB port. The bad news is that to simultaneously use two or "
    "more of the above you'll need to pick up a USB-C Multi Port adapter, which Apple sells for "
    "£65. Apple had to make a trade-off between portability and convenience, and if you're set on "
    "using wired peripherals or charging your smartphone on the move, sorry: coughing up for and "
    "carrying around an adapter is a necessary evil. Along with a switch to USB Type-C, Apple had "
    "to engineer an entirely new keyboard and trackpad to make the new MacBook catwalk-thin. The "
    "innovative Force Touch Trackpad, which enables a third 'Force' click by pressing down on the "
    "trackpad with a certain degree of pressure, can be used for anything from activating a Quick "
    "Look preview to annotating attachments or seeing a file's information. Where traditional "
    "keyboards use a scissor mechanism, which tends to wobble around the edges, the new MacBook's "
    "keyboard uses Apple's new butterfly hinge under each individually-backlit key. Made from a "
    "single, strong piece of material, it makes every keypress responsive wherever you hit it. "
    "While precise, it provides minimal travel and feels closer to typing on an on-screen "
    "keyboard than a traditional tactile one. If you're looking to buy the new MacBook to hammer "
    "out long documents on the regular, see if you can try its keyboard out first - or you may "
    "find yourself taking advantage of Apple's 14-day returns policy. The new MacBook's speakers "
    "provide surprisingly loud and clear sound at high volumes with punchy mid-range tones and, "
    "for their size, impressive low-end. New MacBook: Display The centrepiece of the new MacBook, "
    "its 2,304 x 1,440 pixel-resolution display is a sight to behold. At 226 pixels-per-inch "
    "(PPI), it goes toe-to-toe with the 13-inch MacBook Pro with Retina (227 PPI) - I'd go as far "
    "to say that the new MacBook's thinner black bezel makes it the more attractive of the two. "
    "It's an IPS variant, which means bold colours, deep blacks and excellent viewing angles; "
    "crank up the brightness and you can even read websites comfortably outdoors - although "
    "you'll have to put up with screen glare and reflections. By default, OS X sets the "
    "resolution to 1,280 x 800, which renders crisp text, smooth lines and sharp images but "
    "leaves little room on the desktop for apps and windows. Upping it to 1,440 x 900 strikes a "
    "better balance between usable space and image quality. New MacBook: Performance The new "
    "MacBook houses Intel's Core M processor, which brings a few advantages and disadvantages. On "
    "the plus side, it uses so little power that Apple didn't need to put a fan inside, lending "
    "to its thin-ness. No fan means no noise, and very little heat dissipation - even when you're "
    "pushing the machine to its limits. So, here's the not so good part. Core M, a chip designed "
    "for mobile devices and 2-in-1 hybrids, is a good deal slower than the Core-series processors "
    "found in today's MacBook Air and MacBook Pro with Retina models, and as such isn't "
    "well-suited to heavy computing tasks such as editing large image files, videos and 3D "
    "rendering. On the other hand, the new MacBook's 8GB of RAM and fast 128GB or 256GB SSD mean "
    "it's easily nippy enough for everyday tasks - such as surfing the web, editing small-medium "
    "image files, streaming video, writing documents and even some (very) light gaming thanks to "
    "its integrated HD 5300 graphics. New MacBook: Battery Rather than a traditional rectangular "
    "battery, Apple came up with a terraced, contoured battery design to better fit the new "
    "MacBook's slim dimensions. Rated at 39.7Whr, it fell just short of Apple's 9-hour batter "
    "life claim during T3's rigorous battery life test, clocking in at just over 7 hours when "
    "looping a 1080p video over Wi-Fi. That's still impressive when you consider that it's "
    "driving all of those pixels, but if you value battery life above all else then the 13-inch "
    "MacBook Air, which has the legs to go for up to 12 hours, is still the king. New MacBook: "
    "Verdict A genuinely unique laptop for those seeking a Retina display in the most compact and "
    "light chassis around, the fact remains that the new MacBook's many compromises mean it won't "
    "suit everyone. It has the ability to stun and confuse in equal measure, packing enough power "
    "to be used as your main machine, depending on what you do. But even then you'll have to put "
    "up with switching adapters when hooking up monitors and wired peripherals and re-train your "
    "fingers to adapt to the keyboard's lack of tactile feedback. If money is no object, and "
    "you're prepared to see past its deficiencies, Apple's gorgeous new MacBook glistens like "
    "gold (literally, if you opt for it in Gold, rather than Silver or Space Grey). For everyone "
    "else, it may be worth at least holding out for a successor with one more USB port to appear "
    "down the line. New MacBook: is this the way all laptops should be? Including just one USB "
    "Type-C port and tampering with the MacBook Air's near-perfect keyboard were daring, if "
    "unsurprising moves considering Apple's penchant for minimalism. Holding the new MacBook in "
    "the hand is like seeing into the future of laptop design; it's the MacBook Air all over "
    "again. Tablets are expected to sell poorly for a second consecutive year in 2015 as people "
    "look toward thin and light computers that let them get real work done, an area where the new "
    "MacBook excels. But - and it's a biggy - it feels like the new MacBook is ahead of its time "
    "and will prove one too many compromises for most people. Using an adapter is awkward "
    "compared to full-size USB ports, adds cost and detracts from the machine's slick design. "
    "It's likely that future laptop makers - including Apple - will at least do what Google did "
    "with the Chromebook Pixel 2 and include two USB Type-C ports to lessen the pain."
)

#: The "hack crash" tweet of 23 April 2013, posted from the compromised Associated
#: Press account. The S&P 500 fell around one per cent within minutes and recovered
#: once the tweet was denied. Discussed in
#: https://doi.org/10.1177/0263276415583139
HACK_CRASH_TWEET = "Breaking: Two Explosions in the White House and Barack Obama is injured"

#: Muddy Waters Research, 6 August 2019, trailing a short position that turned out to be
#: in Burford Capital. Burford's shares fell heavily the following day. Quoted in the
#: claimants' skeleton argument for trial in Burford v LSE:
#: https://www.burfordcapital.com/media/1712/20200401-burford-v-lse-claimants-skeleton-argument-for-trial.pdf
MUDDY_WATERS_TWEET = (
    "Muddy Waters is now in a blackout period until tomorrow 8 am London time when we will "
    "announce a new short position on an accounting fiasco that’s potentially insolvent and "
    "possibly facing a liquidity crunch. Investors are bulled up about this company. We’re not"
)

#: Aspect vocabulary for the MacBook reviews, taken from the headings of Apple's own
#: technical specification. Lower case, because the matching is done on lower-cased
#: tokens. "resolution" is not an Apple heading but appears in the short review, and is
#: kept so that the worked example in the notebook still has something to find.
MAC_ASPECTS = [
    "price",
    "display",
    "resolution",
    "processor",
    "storage",
    "memory",
    "graphics",
    "charging",
    "expansion",
    "keyboard",
    "trackpad",
    "wireless",
    "camera",
    "video",
    "audio",
    "battery",
    "power",
    "size",
    "weight",
]

#: Everything the week 7 experiment analyses, in the order it reports them.
CORPUS = {
    "short_review": SHORT_REVIEW,
    "long_review": LONG_REVIEW,
    "hack_crash_tweet": HACK_CRASH_TWEET,
    "muddy_waters_tweet": MUDDY_WATERS_TWEET,
}
