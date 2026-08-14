"""Tests for the sentiment analysis module and the corpus it reads.

Three kinds of test live here.

The corpus tests are regression tests. The texts reached this repository with
U+FFFD replacement characters where pound signs and typographic apostrophes had
been, and that damage is invisible in most diffs and easy to reintroduce. A test
that reads the source file as bytes catches it.

The index tests are the ones that matter for correctness. Word indices and
n-gram indices are different spaces and the notebook mixed them, so the mapping
between them is tested directly, on inputs built by hand rather than by the
tokeniser. That keeps them fast and, more usefully, keeps them runnable with no
NLTK corpus installed.

The end to end tests need the tokenizer corpus, which is a download, so they
skip when it is absent. Only the test marked `network` will fetch anything, and
that marker is deselected by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fintech import reviews
from fintech.sentiment import (
    DEFAULT_N,
    CorpusUnavailableError,
    analyse_document,
    analyse_text,
    aspect_polarities_for_sentence,
    aspect_polarity,
    aspect_sentiment_frame,
    aspect_word_positions,
    corpora_available,
    ensure_corpora,
    mean_aspect_polarity,
    ngram_centres,
    ngram_range_ends,
    plot_sentence_sentiment,
    sentence_sentiment_frame,
    word_range_ends,
    word_to_ngram_index,
)

REVIEWS_PATH = Path(reviews.__file__)

ALL_TEXTS = (
    reviews.SHORT_REVIEW,
    reviews.LONG_REVIEW,
    reviews.HACK_CRASH_TWEET,
    reviews.MUDDY_WATERS_TWEET,
)


class FakeSentence:
    """A sentence-shaped object that needs no tokenizer corpus.

    TextBlob cannot split words without a downloaded corpus, but the aspect
    splitting maths only needs a word list, so the tests supply one.
    """

    def __init__(self, words, polarity=0.0):
        self.words = list(words)
        self.polarity = polarity

    def ngrams(self, n=DEFAULT_N):
        return [self.words[i : i + n] for i in range(len(self.words) - n + 1)]


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


def test_corpus_source_file_is_clean_utf8():
    """No replacement characters anywhere in the corpus module, bytes and all."""

    raw = REVIEWS_PATH.read_bytes()
    text = raw.decode("utf-8")  # raises if the file is not valid UTF-8
    assert "�" not in text
    assert b"\xef\xbf\xbd" not in raw


@pytest.mark.parametrize("text", ALL_TEXTS)
def test_texts_contain_no_replacement_characters(text):
    assert "�" not in text


@pytest.mark.parametrize("text", ALL_TEXTS)
def test_texts_contain_no_em_or_en_dashes(text):
    """House style, and the repaired text has to keep to it as well."""

    assert "—" not in text
    assert "–" not in text


def test_repaired_characters_are_the_intended_ones():
    """The specific characters that had been lost are back."""

    assert "£1,049" in reviews.LONG_REVIEW
    assert "£1,299" in reviews.LONG_REVIEW
    assert "£65" in reviews.LONG_REVIEW
    assert "that’s potentially insolvent" in reviews.MUDDY_WATERS_TWEET
    assert reviews.MUDDY_WATERS_TWEET.endswith("We’re not")
    # The dash that joined these two clauses is now a comma.
    assert "in a heartbeat, but whether" in reviews.LONG_REVIEW


def test_corpus_mapping_matches_the_constants():
    assert reviews.CORPUS["short_review"] == reviews.SHORT_REVIEW
    assert reviews.CORPUS["muddy_waters_tweet"] == reviews.MUDDY_WATERS_TWEET
    assert len(reviews.CORPUS) == 4


def test_aspect_list_is_lower_case_and_unique():
    assert all(aspect == aspect.lower() for aspect in reviews.MAC_ASPECTS)
    assert len(set(reviews.MAC_ASPECTS)) == len(reviews.MAC_ASPECTS)


# --------------------------------------------------------------------------
# Index spaces: words against n-grams
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_ngram_count_is_the_invariant_the_mapping_rests_on(n):
    words = [f"w{i}" for i in range(12)]
    sentence = FakeSentence(words)
    assert len(sentence.ngrams(n)) == len(words) - n + 1


def test_word_to_ngram_index_centres_the_word():
    """n-gram g covers words g .. g + n - 1, so word w sits in n-gram w - (n-1)//2."""

    n, n_ngrams = 3, 10
    assert word_to_ngram_index(5, n, n_ngrams) == 4
    # Clamped at both ends rather than running off the list.
    assert word_to_ngram_index(0, n, n_ngrams) == 0
    assert word_to_ngram_index(50, n, n_ngrams) == n_ngrams


def test_word_to_ngram_offset_grows_with_n():
    assert word_to_ngram_index(6, 1, 20) == 6
    assert word_to_ngram_index(6, 3, 20) == 5
    assert word_to_ngram_index(6, 5, 20) == 4


def test_ngram_centres_line_up_with_the_words_they_cover():
    words = [f"w{i}" for i in range(10)]
    n = 3
    n_ngrams = len(words) - n + 1
    centres = ngram_centres(n_ngrams, n)
    assert len(centres) == n_ngrams
    assert centres[0] == 1.0
    assert centres[-1] == float(len(words) - 2)
    # Every point sits inside the word axis, which is what the old plot got wrong.
    assert all(0 <= c <= len(words) - 1 for c in centres)


def test_word_range_ends_are_midpoints_between_aspects():
    assert word_range_ends([1, 13, 16], 20) == [0, 7, 14, 20]


def test_word_range_ends_never_go_backwards_for_adjacent_aspects():
    ends = word_range_ends([3, 4, 5], 8)
    assert ends == sorted(ends)
    assert ends[0] == 0
    assert ends[-1] == 8


def test_ngram_range_ends_stay_in_ngram_space():
    words = list(range(20))
    n = 3
    n_ngrams = len(words) - n + 1
    ends = ngram_range_ends(word_range_ends([1, 13, 16], len(words)), n, n_ngrams)
    assert ends[0] == 0
    assert ends[-1] == n_ngrams
    assert ends == sorted(ends)
    assert all(0 <= e <= n_ngrams for e in ends)
    # This is the failure the notebook had: a word space boundary used directly
    # as an n-gram index runs past the end of the polarity list.
    assert word_range_ends([1, 13, 16], len(words))[-1] > n_ngrams


@pytest.mark.parametrize("positions", [[0], [0, 1], [0, 1, 2], [2, 9], [0, 5, 9], [9]])
def test_ngram_slices_cover_every_ngram_exactly_once(positions):
    words = [f"w{i}" for i in range(10)]
    n = DEFAULT_N
    n_ngrams = len(words) - n + 1
    ends = ngram_range_ends(word_range_ends(positions, len(words)), n, n_ngrams)
    assert len(ends) == len(positions) + 1
    covered = []
    for a in range(len(ends) - 1):
        covered.extend(range(ends[a], ends[a + 1]))
    assert covered == list(range(n_ngrams))


def test_aspect_word_positions_are_in_order_of_appearance_and_case_insensitive():
    words = ["Price", "is", "high", "but", "the", "processor", "sings"]
    assert aspect_word_positions(words, ["processor", "price"]) == [("Price", 0), ("processor", 5)]


# --------------------------------------------------------------------------
# Aspect polarity: safety and range
# --------------------------------------------------------------------------


def test_aspect_polarity_of_an_empty_slice_is_zero_not_an_exception():
    """The notebook called min() on an empty slice, which raises ValueError."""

    assert aspect_polarity([]) == 0.0


def test_aspect_polarity_stays_inside_the_polarity_range():
    assert aspect_polarity([1.0, 1.0, 0.9]) == 1.0
    assert aspect_polarity([-1.0, -1.0]) == -1.0
    assert -1.0 <= aspect_polarity([0.8, 0.9]) <= 1.0


def test_aspect_polarity_sums_min_and_max():
    assert aspect_polarity([0.0, 0.5, -0.2]) == pytest.approx(0.3)
    assert aspect_polarity([0.4]) == pytest.approx(0.8)


def test_aspect_polarity_cannot_tell_conflict_from_silence():
    """The documented limitation, pinned so nobody trusts the number too far."""

    assert aspect_polarity([0.6, -0.6]) == aspect_polarity([0.0, 0.0])


# --------------------------------------------------------------------------
# Degenerate sentences
# --------------------------------------------------------------------------


def test_sentence_shorter_than_n_falls_back_to_sentence_polarity():
    sentence = FakeSentence(["price", "bad"], polarity=-0.7)
    assert sentence.ngrams(3) == []
    assert aspect_polarities_for_sentence(sentence, ["price"]) == [("price", -0.7)]


def test_sentence_with_no_aspects_yields_nothing():
    sentence = FakeSentence(["the", "quick", "brown", "fox", "jumps"])
    assert aspect_polarities_for_sentence(sentence, ["price"]) == []


def test_adjacent_aspects_do_not_raise():
    sentence = FakeSentence(["the", "price", "storage", "memory", "are", "fine", "here"])
    scored = aspect_polarities_for_sentence(sentence, ["price", "storage", "memory"])
    assert [word for word, _ in scored] == ["price", "storage", "memory"]
    assert all(-1.0 <= p <= 1.0 for _, p in scored)


def test_empty_aspect_frame_averages_to_an_empty_frame():
    import pandas as pd

    empty = pd.DataFrame(columns=["Aspect", "Polarity", "Text"])
    assert mean_aspect_polarity(empty).empty


# --------------------------------------------------------------------------
# Document level, which needs no corpus at all
# --------------------------------------------------------------------------


def test_document_level_needs_no_corpus():
    result = analyse_document("This laptop is wonderful and the screen is excellent.", "test")
    assert -1.0 <= result.polarity <= 1.0
    assert 0.0 <= result.subjectivity <= 1.0
    assert result.polarity > 0
    assert result.label == "positive"


def test_document_labels_follow_the_polarity():
    assert analyse_document("The keyboard is awful and the price is terrible.").label == "negative"
    assert analyse_document("The device weighs 2.03 pounds.").label == "neutral"


def test_analyse_text_degrades_to_document_level_without_the_corpus(monkeypatch):
    monkeypatch.setattr("fintech.sentiment.corpora_available", lambda: False)
    analysis = analyse_text(reviews.HACK_CRASH_TWEET, name="hack_crash")
    assert analysis.sentences is None
    assert analysis.aspects is None
    assert analysis.notes and "download_corpora" in analysis.notes[0]


def test_ensure_corpora_reports_the_fix_rather_than_a_stack_trace(monkeypatch):
    monkeypatch.setattr("fintech.sentiment.corpora_available", lambda: False)
    with pytest.raises(CorpusUnavailableError) as excinfo:
        ensure_corpora()
    assert "download_corpora" in str(excinfo.value)


def test_ensure_corpora_is_a_no_op_when_the_corpus_is_present(monkeypatch):
    monkeypatch.setattr("fintech.sentiment.corpora_available", lambda: True)
    ensure_corpora()
    ensure_corpora(download=True)  # idempotent, and must not reach the network


def test_corpora_available_answers_without_touching_the_network():
    assert isinstance(corpora_available(), bool)


# --------------------------------------------------------------------------
# End to end, when the tokenizer corpus happens to be installed
# --------------------------------------------------------------------------


@pytest.fixture
def corpus_or_skip(nltk_available):
    if not nltk_available:
        pytest.skip("NLTK tokenizer corpus is not installed; run with --download once")


def test_short_review_has_the_expected_shape(corpus_or_skip):
    df = sentence_sentiment_frame(reviews.SHORT_REVIEW, reviews.MAC_ASPECTS)
    assert list(df.columns) == ["Sentence", "Text", "Polarity", "Subjectivity", "Aspects"]
    assert len(df) == 5
    assert df["Polarity"].between(-1.0, 1.0).all()
    assert df["Subjectivity"].between(0.0, 1.0).all()
    # The known short text: the resolution sentence is the most positive one and
    # the weight sentence is the only negative one.
    best = df.loc[df["Polarity"].idxmax(), "Text"]
    worst = df.loc[df["Polarity"].idxmin(), "Text"]
    assert "resolution" in best
    assert "weight" in worst
    assert df.loc[4, "Aspects"] == ["price", "storage", "processor"]


def test_aspect_frame_is_built_once_with_a_sane_index(corpus_or_skip):
    df = aspect_sentiment_frame(reviews.SHORT_REVIEW, reviews.MAC_ASPECTS)
    assert list(df.columns) == ["Aspect", "Polarity", "Text"]
    assert list(df.index) == list(range(len(df)))  # the notebook indexed every row 1
    assert df["Polarity"].between(-1.0, 1.0).all()
    scores = dict(zip(df["Aspect"], df["Polarity"], strict=True))
    assert set(scores) == {"resolution", "weight", "price", "storage", "processor"}
    assert scores["processor"] > 0
    assert scores["price"] < 0


def test_multi_aspect_sentence_splits_where_the_aspects_are(corpus_or_skip):
    df = sentence_sentiment_frame(reviews.SHORT_REVIEW, reviews.MAC_ASPECTS)
    sentence = df.loc[4, "Sentence"]
    scored = aspect_polarities_for_sentence(sentence, reviews.MAC_ASPECTS)
    assert [word for word, _ in scored] == ["price", "storage", "processor"]
    assert all(-1.0 <= p <= 1.0 for _, p in scored)


def test_plot_axis_and_curve_agree(corpus_or_skip):
    """The plot bug: W tick labels against a curve of W - n + 1 points."""

    import matplotlib.pyplot as plt

    df = sentence_sentiment_frame(reviews.SHORT_REVIEW, reviews.MAC_ASPECTS)
    sentence = df.loc[4, "Sentence"]
    words = list(sentence.words)
    fig = plot_sentence_sentiment(sentence, reviews.MAC_ASPECTS)
    try:
        line = fig.axes[0].lines[0]
        xdata = list(line.get_xdata())
        assert len(xdata) == len(words) - DEFAULT_N + 1
        assert xdata[0] == (DEFAULT_N - 1) / 2
        assert xdata[-1] == len(words) - 1 - (DEFAULT_N - 1) / 2
        assert len(fig.axes[0].get_xticklabels()) == len(words)
        assert fig.gram_range_ends[-1] == len(xdata)
    finally:
        plt.close(fig)


def test_long_review_aspects_stay_in_range(corpus_or_skip):
    df = aspect_sentiment_frame(reviews.LONG_REVIEW, reviews.MAC_ASPECTS)
    assert not df.empty
    assert df["Polarity"].between(-1.0, 1.0).all()
    means = mean_aspect_polarity(df)
    assert means["Polarity"].is_monotonic_decreasing


def test_tweets_are_analysable_at_every_level(corpus_or_skip):
    """No assertion about the values. The experiment reports those."""

    for text in (reviews.HACK_CRASH_TWEET, reviews.MUDDY_WATERS_TWEET):
        analysis = analyse_text(text, name="tweet")
        assert analysis.sentences is not None
        assert not analysis.sentences.empty
        assert -1.0 <= analysis.document.polarity <= 1.0


@pytest.mark.network
def test_ensure_corpora_can_fetch_the_corpus():
    ensure_corpora(download=True)
    assert corpora_available()
