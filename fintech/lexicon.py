"""A finance-specific sentiment scorer, in the Loughran-McDonald style.

The rest of this repository scores text with TextBlob, whose lexicon is derived
from general English and, in the case of the pattern analyser, largely from film
and product reviews. `FINDINGS.md` argues that this is the wrong tool for market
text, and until now it argued it from two anecdotes. This module supplies the
alternative that makes the claim measurable, and `fintech.evaluation` does the
measuring.

Why this particular alternative. Loughran and McDonald (2011), "When Is a
Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks", Journal of
Finance 66(1), 35-65, took the general-purpose Harvard-IV psychological
dictionary that the finance literature was using at the time and showed that
roughly three quarters of the words it counts as negative are not negative in a
financial filing: "liability", "cost", "tax", "capital", "board", "foreign",
"vice". Their response was to build word lists from the filings themselves. That
paper is the standard reference for "use a finance dictionary, not a general
one", which is exactly the claim under test here, so its approach is the one
worth reproducing.

What this is not. It is not the Loughran-McDonald dictionary. The published
master dictionary runs to tens of thousands of entries and is distributed as a
download; nothing in this repository touches the network, so the lists below are
hand entered. They are a SUBSET, chosen for words that plausibly appear in
short market announcements, and they were written and frozen before the
evaluation in `fintech.evaluation` was ever run, so the accuracy figures are not
the product of tuning the lexicon against the test set. Any comparison reported
against TextBlob is therefore a comparison against a small hand-made subset and
not against the published dictionary, which would very likely do better.

Three deliberate limitations, all of which the evaluation exposes.

Loughran and McDonald count single words. So does this. A finance headline is
full of phrases whose direction lives in the pair rather than in either word:
"cut costs" is good news and "cut the dividend" is bad news, and a unigram
counter scores both from "cut". This is a real property of the method rather
than a shortcoming of this implementation, and it is left in place so that the
evaluation can show it.

The dictionary has other categories, uncertainty, litigious, constraining, and
strong and weak modal words. None of them is directional, so none of them is
used here. Only the positive and negative lists contribute.

Negation is handled, which the original word counting is not. Loughran and
McDonald discuss negated positives, and later work counts them, so a small
negation rule is included: a negator within the preceding few tokens flips the
sign of a sentiment word. That turns "margins have not improved" from positive
into negative, and "cleared of any wrongdoing" from negative into positive. It is
a window rule and it does not parse, so it will misfire on a negation that scopes
further than the window or stops sooner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Positive entries, hand entered from the Loughran-McDonald positive list.
#: Short and blunt by design: the published positive list is itself much smaller
#: than the negative one, because filings state good news in neutral language and
#: reserve strong words for trouble.
POSITIVE_WORDS = frozenset(
    {
        "able",
        "achieve",
        "achieved",
        "achievement",
        "achievements",
        "achieves",
        "advantage",
        "advantages",
        "attain",
        "attained",
        "beneficial",
        "benefit",
        "benefited",
        "benefits",
        "best",
        "better",
        "breakthrough",
        "collaboration",
        "conclusive",
        "constructive",
        "delight",
        "delighted",
        "desirable",
        "effective",
        "efficiency",
        "efficient",
        "empower",
        "enable",
        "enabled",
        "enables",
        "encouraged",
        "encouraging",
        "enhance",
        "enhanced",
        "enhancement",
        "enhancements",
        "enhances",
        "enhancing",
        "enjoy",
        "enjoyed",
        "excellence",
        "excellent",
        "exceptional",
        "exclusive",
        "favorable",
        "gain",
        "gained",
        "gaining",
        "gains",
        "good",
        "great",
        "greatest",
        "highest",
        "impressive",
        "improve",
        "improved",
        "improvement",
        "improvements",
        "improves",
        "improving",
        "incredible",
        "innovation",
        "innovative",
        "leadership",
        "lucrative",
        "opportunities",
        "opportunity",
        "optimistic",
        "outperform",
        "outperformed",
        "outperforming",
        "perfect",
        "pleased",
        "popular",
        "positive",
        "positively",
        "premier",
        "proactive",
        "profitability",
        "profitable",
        "progress",
        "prosperity",
        "rebound",
        "rebounded",
        "regain",
        "regained",
        "resolved",
        "reward",
        "rewarded",
        "satisfaction",
        "satisfied",
        "solved",
        "stabilize",
        "strength",
        "strengthen",
        "strengthened",
        "strengthening",
        "strengths",
        "strong",
        "stronger",
        "strongest",
        "succeed",
        "succeeded",
        "success",
        "successes",
        "successful",
        "successfully",
        "superior",
        "surpass",
        "surpassed",
        "transparency",
        "tremendous",
        "unmatched",
        "unparalleled",
        "upturn",
        "valuable",
        "versatile",
        "vibrant",
        "win",
        "winner",
        "winners",
        "winning",
        "worthy",
    }
)

#: Negative entries, hand entered from the Loughran-McDonald negative list. Much
#: longer than the positive list, in the same proportion as the published one.
NEGATIVE_WORDS = frozenset(
    {
        "abandon",
        "abandoned",
        "abandoning",
        "adverse",
        "adversely",
        "allegation",
        "allegations",
        "allege",
        "alleged",
        "alleges",
        "bankrupt",
        "bankruptcy",
        "breach",
        "breached",
        "breaches",
        "burden",
        "burdened",
        "cancel",
        "cancelled",
        "cancellation",
        "cease",
        "ceased",
        "claim",
        "claims",
        "closure",
        "complaint",
        "complaints",
        "compressed",
        "concede",
        "conceded",
        "correction",
        "criminal",
        "crisis",
        "critical",
        "damage",
        "damaged",
        "damages",
        "decline",
        "declined",
        "declines",
        "declining",
        "default",
        "defaulted",
        "defective",
        "deficiency",
        "deficiencies",
        "deficit",
        "delay",
        "delayed",
        "delays",
        "deteriorate",
        "deteriorated",
        "deterioration",
        "difficult",
        "difficulties",
        "difficulty",
        "dilution",
        "dilutive",
        "diminish",
        "diminished",
        "disappointing",
        "disappointment",
        "discontinued",
        "dismissed",
        "dispute",
        "disputes",
        "disruption",
        "doubt",
        "doubtful",
        "downgrade",
        "downgraded",
        "downturn",
        "erroneous",
        "error",
        "errors",
        "fail",
        "failed",
        "failing",
        "fails",
        "failure",
        "failures",
        "falling",
        "fraud",
        "fraudulent",
        "halt",
        "halted",
        "harm",
        "harmful",
        "impair",
        "impaired",
        "impairment",
        "impairments",
        "inability",
        "inadequate",
        "insolvency",
        "insolvent",
        "investigate",
        "investigated",
        "investigation",
        "investigations",
        "lawsuit",
        "lawsuits",
        "liquidation",
        "litigation",
        "lose",
        "loses",
        "losing",
        "loss",
        "losses",
        "lost",
        "misconduct",
        "misstatement",
        "misstatements",
        "negative",
        "negatively",
        "overstate",
        "overstated",
        "overstatement",
        "penalties",
        "penalty",
        "poor",
        "poorly",
        "problem",
        "problems",
        "punitive",
        "recall",
        "recession",
        "restated",
        "restatement",
        "restructuring",
        "resign",
        "resignation",
        "resigned",
        "severe",
        "severely",
        "shortfall",
        "shrunk",
        "slowdown",
        "slowed",
        "suspend",
        "suspended",
        "suspension",
        "terminate",
        "terminated",
        "termination",
        "unable",
        "uncollectible",
        "underperform",
        "underperformed",
        "unfavorable",
        "unprofitable",
        "unsuccessful",
        "warning",
        "warnings",
        "weak",
        "weaken",
        "weakened",
        "weakening",
        "weakness",
        "weaknesses",
        "withdrawn",
        "worse",
        "worsening",
        "worst",
        "wrongdoing",
    }
)

#: Words that reverse the sign of a sentiment word appearing shortly after them.
#: Not part of the published dictionary; see the module docstring for why one is
#: needed anyway.
NEGATORS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nor",
        "without",
        "cannot",
        "cant",
        "wont",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "doesnt",
        "dont",
        "didnt",
        "refused",
        "refuses",
        "denied",
        "denies",
        "avoided",
        "removed",
        "removes",
        "removing",
        "cleared",
        "ceased",
        "stopped",
    }
)

#: How many tokens after a negator stay inside its scope. Three is the usual
#: choice in the accounting literature that counts negated positives, and it
#: covers "not expected to meet", "no longer expects to need".
NEGATION_WINDOW = 3

#: A word must move the score by more than this before the text is called
#: positive or negative. It matches the threshold `fintech.sentiment` uses on
#: TextBlob polarity, so the two analysers are read off their scores the same
#: way. Since scores here are ratios of whole word counts the exact value only
#: has to separate zero from the smallest non-zero ratio.
DEFAULT_THRESHOLD = 0.05

_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class LexiconScore:
    """A scored text, with the evidence that produced the score.

    `terms` lists every matched word as (word, signed weight) in order of
    appearance, where the weight is -1 or +1 after any negation has been applied.
    It exists so that a disagreement with TextBlob can be read rather than
    guessed at.
    """

    text: str
    positive: int
    negative: int
    score: float
    terms: tuple[tuple[str, int], ...]

    @property
    def matched(self) -> int:
        """How many sentiment words the lexicon recognised at all."""

        return self.positive + self.negative

    def label(self, threshold: float = DEFAULT_THRESHOLD) -> str:
        """The three way call, on the same convention as TextBlob's polarity."""

        if self.score > threshold:
            return "positive"
        if self.score < -threshold:
            return "negative"
        return "neutral"


def tokenise(text: str) -> list[str]:
    """Lower-cased alphabetic tokens.

    Apostrophes are dropped rather than split on, so that "doesn't" becomes the
    single token "doesnt" and can be listed as a negator, and digits are dropped
    entirely because no entry in either word list contains one.
    """

    return _WORD.findall(text.lower().replace("'", "").replace("’", ""))


def word_polarity(word: str) -> int:
    """+1, -1 or 0 for a single word, before any negation is applied."""

    lowered = word.lower()
    if lowered in POSITIVE_WORDS:
        return 1
    if lowered in NEGATIVE_WORDS:
        return -1
    return 0


def is_negated(tokens: list[str], index: int, window: int = NEGATION_WINDOW) -> bool:
    """True when a negator sits within `window` tokens before `index`.

    An even number of negators inside the window leaves the sign alone, which is
    the honest reading of "not without difficulty" even though the rule gets
    there by counting rather than by understanding.
    """

    start = max(0, index - window)
    negators = sum(1 for token in tokens[start:index] if token in NEGATORS)
    return negators % 2 == 1


def score_text(text: str, window: int = NEGATION_WINDOW) -> LexiconScore:
    """Count the sentiment words in a text and reduce them to one number.

    The score is (positive - negative) / (positive + negative), so it lives in
    [-1, 1] and is independent of how long the text is. A text containing no
    word from either list scores 0.0, and so does a text whose positive and
    negative counts are equal. Those two cases are genuinely different and the
    score does not distinguish them, which is why `matched` is reported
    alongside: zero out of zero words recognised is ignorance, and one out of
    two cancelling is a judgement.
    """

    tokens = tokenise(text)
    terms: list[tuple[str, int]] = []
    positive = negative = 0
    for index, token in enumerate(tokens):
        polarity = word_polarity(token)
        if polarity == 0:
            continue
        if is_negated(tokens, index, window):
            polarity = -polarity
        terms.append((token, polarity))
        if polarity > 0:
            positive += 1
        else:
            negative += 1

    total = positive + negative
    score = 0.0 if total == 0 else (positive - negative) / total
    return LexiconScore(
        text=text,
        positive=positive,
        negative=negative,
        score=float(score),
        terms=tuple(terms),
    )


def polarity(text: str) -> float:
    """The score alone, for callers that want the same shape as TextBlob."""

    return score_text(text).score


def classify(text: str, threshold: float = DEFAULT_THRESHOLD) -> str:
    """negative, neutral or positive for one text."""

    return score_text(text).label(threshold)
