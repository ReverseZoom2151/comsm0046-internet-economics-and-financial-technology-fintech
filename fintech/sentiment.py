"""Document, sentence and aspect level sentiment analysis with TextBlob.

The point of this module is to make the week 7 material testable. The original
notebook mixed environment mutation (`!pip install`, `nltk.download`) with the
analysis itself, built its frames a row at a time inside a loop, and, most
importantly, confused two different index spaces when it split a sentence around
the aspects it mentions. All of that is fixed here, and the parts that can go
wrong are exposed as small functions with a single job so a test can pin them.

Three ideas are worth keeping straight.

Document level asks one question of a whole text and gets one pair of numbers
back. It is cheap and it is what most "sentiment driven" trading signals use. It
is also the level at which TextBlob is most badly wrong about the texts in
`fintech.reviews`, which is the teaching point.

Sentence level splits the text first, which needs the NLTK punkt tokenizer.
That corpus is a download, so `ensure_corpora` makes the requirement explicit
and fails with an instruction rather than a stack trace out of the tokeniser.

Aspect level tries to attribute sentiment to the thing being talked about. The
approach here is the notebook's: find aspect words, cut the sentence at the
midpoints between them, and score each piece from the polarity of its n-grams.
It is a rough heuristic and the docstrings say where it breaks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from textblob import TextBlob

from fintech.plotting import new_figure

#: NLTK resources TextBlob needs before it can split a text into sentences or
#: words. Modern NLTK looks for `punkt_tab`; older releases only know `punkt`.
#: Either one being present is enough for the installed version to work, so both
#: are tried and only a total absence is an error.
REQUIRED_CORPORA = ("punkt_tab", "punkt")

#: Default n-gram length. The notebook used 3 and the aspect heuristic is tuned
#: to it: with n=3 an aspect word sits at the centre of the n-gram that starts
#: one word before it.
DEFAULT_N = 3


class CorpusUnavailableError(RuntimeError):
    """The NLTK tokenizer data TextBlob needs is not installed."""


def corpora_available() -> bool:
    """True when at least one of the punkt tokenizers is already on disk.

    Deliberately does not touch the network. Tests use this to skip rather than
    to download.
    """

    try:
        import nltk.data
    except ImportError:  # pragma: no cover - nltk is a hard dependency
        return False

    for corpus in REQUIRED_CORPORA:
        try:
            nltk.data.find(f"tokenizers/{corpus}")
            return True
        except LookupError:
            continue
    return False


def ensure_corpora(download: bool = False, quiet: bool = True) -> None:
    """Check for the tokenizer corpora, optionally fetching them once.

    Idempotent in both modes. With `download=False` (the default, and what the
    library code uses) nothing is fetched and a missing corpus raises
    `CorpusUnavailableError` carrying the command that fixes it. With
    `download=True` the corpora are fetched if absent; NLTK itself skips a
    package that is already up to date, so calling this repeatedly is cheap.

    The notebook instead put `!pip install -U textblob` and a bare
    `nltk.download('punkt')` in code cells, so running it mutated the
    environment and any failure surfaced as a LookupError from deep inside the
    tokeniser. Installation is a setup step, not part of an analysis.
    """

    if corpora_available():
        return

    if download:
        import nltk

        for corpus in REQUIRED_CORPORA:
            try:
                nltk.download(corpus, quiet=quiet)
            except Exception:  # offline, or the package name is unknown here
                continue
        if corpora_available():
            return

    raise CorpusUnavailableError(
        "TextBlob needs an NLTK sentence tokenizer that is not installed. "
        "Run `python -m textblob.download_corpora` once, or "
        "`python -c \"import nltk; nltk.download('punkt_tab')\"`, or call "
        "fintech.sentiment.ensure_corpora(download=True) from a machine with "
        "network access. Document level analysis works without it; sentence "
        "and aspect level analysis do not."
    )


@dataclass(frozen=True)
class DocumentSentiment:
    """One text, one pair of numbers, plus enough context to read them."""

    name: str
    text: str
    polarity: float
    subjectivity: float

    @property
    def label(self) -> str:
        """The three way call a naive trading signal would make on the polarity."""

        if self.polarity > 0.05:
            return "positive"
        if self.polarity < -0.05:
            return "negative"
        return "neutral"


@dataclass
class TextAnalysis:
    """Everything the experiment reports about one text."""

    document: DocumentSentiment
    sentences: pd.DataFrame | None = None
    aspects: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


def analyse_document(text: str, name: str = "document") -> DocumentSentiment:
    """Document level polarity and subjectivity.

    Needs no corpus: TextBlob's pattern analyser averages over the sentiment
    bearing words it recognises and never has to split sentences. That is also
    its weakness. Words it does not recognise contribute nothing at all, so a
    text whose meaning lives entirely in unscored vocabulary comes back neutral.
    """

    blob = TextBlob(text)
    sentiment = blob.sentiment
    return DocumentSentiment(
        name=name,
        text=text,
        polarity=float(sentiment.polarity),
        subjectivity=float(sentiment.subjectivity),
    )


def sentence_sentiment_frame(text: str, aspects: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per sentence, with polarity, subjectivity and any aspects found.

    Columns: Sentence (the TextBlob sentence object), Text, Polarity,
    Subjectivity, and Aspects when an aspect list is supplied.
    """

    ensure_corpora()
    blob = TextBlob(text)
    rows = [
        {
            "Sentence": sentence,
            "Text": str(sentence),
            "Polarity": float(sentence.polarity),
            "Subjectivity": float(sentence.subjectivity),
        }
        for sentence in blob.sentences
    ]
    df = pd.DataFrame(rows, columns=["Sentence", "Text", "Polarity", "Subjectivity"])
    if aspects is not None:
        df = append_aspects_column(df, aspects)
    return df


def append_aspects_column(df: pd.DataFrame, aspects: Sequence[str] | None) -> pd.DataFrame:
    """Add an Aspects column listing the aspect words each sentence mentions.

    Order is order of appearance in the sentence, not alphabetical, because the
    aspect splitting downstream relies on it. Matching is case insensitive,
    which the notebook's version was not: an aspect word that opened a sentence
    was silently missed there.

    `aspects is None` leaves the frame alone. The notebook wrote `aspects!=None`,
    which asks pandas-style equality of a list against None and is only correct
    by accident.
    """

    if aspects is None:
        return df

    wanted = {a.lower() for a in aspects}
    df = df.copy()
    df["Aspects"] = [
        [str(word) for word in sentence.words if str(word).lower() in wanted]
        for sentence in df["Sentence"]
    ]
    return df


def ngram_polarities(sentence, n: int = DEFAULT_N) -> list[float]:
    """Polarity of every n-gram in a sentence, in order.

    A sentence of W words has W - n + 1 n-grams, and none at all when W < n.
    The empty list is a legitimate answer and the callers handle it.
    """

    return [float(TextBlob(" ".join(gram)).polarity) for gram in sentence.ngrams(n)]


def aspect_word_positions(words: Sequence[str], aspects: Sequence[str]) -> list[tuple[str, int]]:
    """Where each aspect word occurs, as (word, word index) in order of appearance."""

    wanted = {a.lower() for a in aspects}
    return [(str(w), i) for i, w in enumerate(words) if str(w).lower() in wanted]


def word_range_ends(positions: Sequence[int], n_words: int) -> list[int]:
    """Cut points in WORD index space: 0, the midpoints between aspects, W.

    Returns len(positions) + 1 boundaries, so slice a in [0, len(positions)) owns
    words [ends[a], ends[a + 1]).
    """

    ends = [0]
    for previous, current in zip(positions, positions[1:], strict=False):
        ends.append(round((previous + current) / 2))
    ends.append(n_words)
    # Aspects can be adjacent or repeated, which would otherwise produce a
    # boundary list that goes backwards and a slice with a negative length.
    for i in range(1, len(ends)):
        ends[i] = min(max(ends[i], ends[i - 1]), n_words)
    return ends


def word_to_ngram_index(word_index: int, n: int, n_ngrams: int) -> int:
    """Map a word index onto the index of the n-gram centred on that word.

    This is the bug the notebook had. `get_range_ends` measured aspect positions
    in words and then appended `len(sentence.ngrams(n))` as the final boundary,
    mixing two index spaces in one list, and the resulting boundaries were used
    to slice a list of n-gram polarities. For a sentence of W words there are
    only W - n + 1 n-grams, so word index w and n-gram index w are not the same
    place: n-gram g spans words g .. g + n - 1 and is centred on word
    g + (n - 1) // 2. Reading the mapping backwards gives this function. With
    the default n = 3 the correction is one position, and it grows with n.

    The result is clamped into [0, n_ngrams] so that boundaries near either end
    of the sentence stay usable as slice indices.
    """

    return min(max(word_index - (n - 1) // 2, 0), n_ngrams)


def ngram_range_ends(word_ends: Sequence[int], n: int, n_ngrams: int) -> list[int]:
    """Convert word space cut points into n-gram space slice boundaries.

    The first boundary is always 0 and the last is always n_ngrams, so the
    slices between them cover every n-gram exactly once. Interior boundaries are
    mapped with `word_to_ngram_index` and forced to be non decreasing, so a
    squeezed range yields an empty slice rather than a reversed one.
    """

    if not word_ends:
        return [0, n_ngrams]

    ends = [0]
    for boundary in word_ends[1:-1]:
        ends.append(max(word_to_ngram_index(boundary, n, n_ngrams), ends[-1]))
    ends.append(n_ngrams)
    for i in range(1, len(ends)):
        ends[i] = min(max(ends[i], ends[i - 1]), n_ngrams)
    return ends


def ngram_centres(n_ngrams: int, n: int) -> list[float]:
    """Word space x positions for a curve of n-gram polarities.

    The notebook plotted the n-gram curve against `range(len(sentence.words))`
    tick labels. The curve had W - n + 1 points and the axis had W labels, so
    every point was labelled with a word it did not cover and the last n - 1
    labels had no curve under them at all. Plotting each n-gram at the position
    of the word it is centred on puts the labels back under the right points.
    """

    offset = (n - 1) / 2
    return [i + offset for i in range(n_ngrams)]


def aspect_polarity(polarities: Sequence[float]) -> float:
    """Score one slice of n-gram polarities by summing its minimum and maximum.

    This is the notebook's "simple hack" made safe. It picks up the strongest
    opinion in either direction, which is why it beats a mean on short slices
    where most n-grams are neutral filler and would drag the average to zero.

    Two things had to be fixed. An empty slice raised ValueError out of min();
    it now scores 0.0, which is the honest answer when a range contains no
    n-grams. And min + max is not a polarity: two polarities in [-1, 1] sum into
    [-2, 2], so the result is clipped back into range.

    The limitation is real and clipping does not remove it. The statistic is not
    a mean, it is not comparable between slices of different lengths, and a
    slice holding one strongly positive and one strongly negative n-gram scores
    near zero, indistinguishable from a slice with no opinion in it at all.
    """

    if len(polarities) == 0:
        return 0.0
    return float(min(max(min(polarities) + max(polarities), -1.0), 1.0))


def split_sentence_by_aspects(
    sentence, aspects: Sequence[str], n: int = DEFAULT_N
) -> tuple[list[tuple[str, int]], list[int], list[int], list[float]]:
    """The whole aspect split for one sentence, in one place.

    Returns the aspect positions, the word space cut points, the n-gram space
    slice boundaries and the n-gram polarities, so a caller can plot them and a
    test can check that the two boundary lists agree.
    """

    words = list(sentence.words)
    found = aspect_word_positions(words, aspects)
    polarities = ngram_polarities(sentence, n)
    word_ends = word_range_ends([i for _, i in found], len(words))
    gram_ends = ngram_range_ends(word_ends, n, len(polarities))
    return found, word_ends, gram_ends, polarities


def aspect_polarities_for_sentence(
    sentence, aspects: Sequence[str], n: int = DEFAULT_N
) -> list[tuple[str, float]]:
    """One (aspect, polarity) pair per aspect mentioned in the sentence.

    A sentence shorter than n has no n-grams to score, so every aspect in it
    falls back to the polarity of the sentence as a whole. That is a coarse
    answer, but it is better than the alternatives of crashing or reporting a
    confident zero.
    """

    found, _word_ends, gram_ends, polarities = split_sentence_by_aspects(sentence, aspects, n)
    if not found:
        return []
    if not polarities:
        return [(word, float(sentence.polarity)) for word, _ in found]
    return [
        (word, aspect_polarity(polarities[gram_ends[i] : gram_ends[i + 1]]))
        for i, (word, _) in enumerate(found)
    ]


def aspect_sentiment_frame(text: str, aspects: Sequence[str], n: int = DEFAULT_N) -> pd.DataFrame:
    """Aspect level sentiment for a whole text, as one row per aspect mention.

    Columns: Aspect, Polarity, Text. Sentences that mention no aspect contribute
    no rows.

    Two things the notebook got wrong are fixed here. Its guard read
    `if not 'Activity' in df.columns`, testing a column name that does not exist
    anywhere, so the branch fired every time; the column it meant is Aspects.
    And it grew the frame with `pd.concat` inside the loop, which is quadratic
    and, with `index=[1]` on every row, produced a frame whose index was all
    ones. Rows are collected in a list and the frame is built once.
    """

    df = sentence_sentiment_frame(text, aspects)
    if "Aspects" not in df.columns:
        df = append_aspects_column(df, aspects)

    rows = []
    for _, row in df.iterrows():
        mentioned = row["Aspects"]
        if len(mentioned) == 1:
            rows.append({"Aspect": mentioned[0], "Polarity": row["Polarity"], "Text": row["Text"]})
        elif len(mentioned) > 1:
            for aspect, polarity in aspect_polarities_for_sentence(row["Sentence"], aspects, n):
                rows.append({"Aspect": aspect, "Polarity": polarity, "Text": row["Text"]})

    return pd.DataFrame(rows, columns=["Aspect", "Polarity", "Text"])


def mean_aspect_polarity(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """Average each aspect over its mentions, most positive first."""

    if aspect_df.empty:
        return pd.DataFrame(columns=["Aspect", "Polarity", "Mentions"])
    grouped = (
        aspect_df.groupby(aspect_df["Aspect"].str.lower())["Polarity"]
        .agg(["mean", "size"])
        .reset_index()
    )
    grouped.columns = ["Aspect", "Polarity", "Mentions"]
    return grouped.sort_values(by="Polarity", ascending=False).reset_index(drop=True)


def plot_sentence_sentiment(sentence, aspects: Sequence[str], n: int = DEFAULT_N):
    """Plot the n-gram polarity curve of one sentence with its aspects marked.

    The curve is drawn against word positions, one point per n-gram placed at
    the word it is centred on, so the word tick labels line up with the points
    that actually cover them. The vertical lines are the word space cut points
    between aspects, which is the same space the labels are in.
    """

    words = [str(w) for w in sentence.words]
    found, word_ends, gram_ends, polarities = split_sentence_by_aspects(sentence, aspects, n)
    scored = aspect_polarities_for_sentence(sentence, aspects, n)

    fig, ax = new_figure(width=max(8.0, 0.32 * len(words)), height=5.0)
    ax.plot(ngram_centres(len(polarities), n), polarities, marker="o", markersize=3)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_title("Sentence sentiment with aspects labelled")
    ax.set_xlabel(f"word position (curve is the polarity of each {n}-gram)")
    ax.set_ylabel("polarity")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=7)

    for (word, index), (_, polarity) in zip(found, scored, strict=False):
        ax.text(index, 0.05, f"{word} = {polarity:.2f}", rotation="vertical", color="red")
    for boundary in word_ends[1:-1]:
        ax.axvline(x=boundary, color="red", linewidth=0.8, linestyle="--")

    # Reported so a caller can assert on the mapping rather than read the plot.
    fig.gram_range_ends = gram_ends
    return fig


def analyse_text(
    text: str,
    name: str = "document",
    aspects: Sequence[str] | None = None,
    n: int = DEFAULT_N,
) -> TextAnalysis:
    """Document level always, sentence and aspect level when the corpus allows.

    Degrades rather than fails: without the tokenizer corpus the document level
    numbers still come back and a note records why the rest is missing.
    """

    analysis = TextAnalysis(document=analyse_document(text, name))
    try:
        ensure_corpora()
    except CorpusUnavailableError as exc:
        analysis.notes.append(str(exc))
        return analysis

    analysis.sentences = sentence_sentiment_frame(text, aspects)
    if aspects is not None:
        analysis.aspects = aspect_sentiment_frame(text, aspects, n)
    return analysis
