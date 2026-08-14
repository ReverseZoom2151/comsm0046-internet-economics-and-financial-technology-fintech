"""Tests for the finance lexicon and for the labelled set it is measured on.

The lexicon is a lookup table plus a negation rule, so the tests that matter are
the ones that pin the rule: which words carry a sign, how far a negator reaches,
and what happens when nothing is recognised at all. The dataset tests are
integrity checks. A duplicated headline would inflate whichever analyser happens
to get it right, an empty text would be scored as neutral by both and count as a
free item, and a mistyped label would create a fourth class that every per-class
metric would then be computed against.
"""

from __future__ import annotations

import pytest

from fintech import lexicon
from fintech.evaluation import HEADLINES_PATH, LABELS, load_headlines


class TestWordLists:
    def test_the_two_lists_do_not_overlap(self):
        assert not (lexicon.POSITIVE_WORDS & lexicon.NEGATIVE_WORDS)

    def test_every_entry_is_lower_case_and_alphabetic(self):
        for word in lexicon.POSITIVE_WORDS | lexicon.NEGATIVE_WORDS | lexicon.NEGATORS:
            assert word.isalpha(), word
            assert word == word.lower(), word

    def test_the_negative_list_is_the_longer_one(self):
        # As in the published dictionary: filings state good news plainly and
        # save the vocabulary for trouble.
        assert len(lexicon.NEGATIVE_WORDS) > len(lexicon.POSITIVE_WORDS)

    @pytest.mark.parametrize("word", ["loss", "impairment", "fraud", "insolvent", "downgrade"])
    def test_known_negative_words_score_negative(self, word):
        assert lexicon.word_polarity(word) == -1

    @pytest.mark.parametrize("word", ["strong", "improved", "profitable", "gains", "win"])
    def test_known_positive_words_score_positive(self, word):
        assert lexicon.word_polarity(word) == 1

    @pytest.mark.parametrize("word", ["dividend", "shareholder", "quarter", "registrar"])
    def test_neutral_finance_words_score_zero(self, word):
        assert lexicon.word_polarity(word) == 0

    def test_word_polarity_is_case_insensitive(self):
        assert lexicon.word_polarity("LOSS") == lexicon.word_polarity("loss") == -1


class TestTokenise:
    def test_lower_cases_and_drops_punctuation(self):
        assert lexicon.tokenise("Profit, warning!") == ["profit", "warning"]

    def test_drops_digits_entirely(self):
        assert lexicon.tokenise("a 1.2 billion loss") == ["a", "billion", "loss"]

    def test_keeps_a_contraction_as_one_token(self):
        # "doesn't" has to survive as a single token because it is a negator.
        assert lexicon.tokenise("it doesn't improve") == ["it", "doesnt", "improve"]


class TestScoring:
    def test_a_text_with_no_recognised_word_scores_zero_and_matches_nothing(self):
        result = lexicon.score_text("The annual general meeting will be held on 14 May")
        assert result.score == 0.0
        assert result.matched == 0
        assert result.label() == "neutral"

    def test_a_single_negative_word_scores_minus_one(self):
        result = lexicon.score_text("The group reported a loss")
        assert result.score == -1.0
        assert (result.positive, result.negative) == (0, 1)
        assert result.label() == "negative"

    def test_a_single_positive_word_scores_plus_one(self):
        result = lexicon.score_text("Margins improved")
        assert result.score == 1.0
        assert result.label() == "positive"

    def test_the_score_is_a_ratio_not_a_count(self):
        # Three negatives to one positive is the same score as thirty to ten.
        result = lexicon.score_text("a loss, a deficit and a shortfall, plus one gain")
        assert result.positive == 1
        assert result.negative == 3
        assert result.score == pytest.approx(-0.5)

    def test_equal_counts_cancel_to_neutral_but_report_the_evidence(self):
        result = lexicon.score_text("strong sales and a weak order book")
        assert result.score == 0.0
        assert result.label() == "neutral"
        # Not the same state as recognising nothing, which is why matched exists.
        assert result.matched == 2

    def test_terms_are_reported_in_order_with_their_signs(self):
        result = lexicon.score_text("An impairment offset a strong gain")
        assert result.terms == (("impairment", -1), ("strong", 1), ("gain", 1))


class TestNegation:
    def test_a_negator_flips_the_following_word(self):
        assert lexicon.score_text("earnings improved").score == 1.0
        assert lexicon.score_text("earnings have not improved").score == -1.0

    def test_a_negation_with_nothing_to_negate_stays_neutral(self):
        # A documented miss rather than a success. "expected", "meet" and
        # "guidance" are all absent from the dictionary, so there is no sign for
        # "not" to flip and the lexicon calls a profit warning neutral. Negation
        # handling cannot rescue vocabulary the word lists do not contain.
        result = lexicon.score_text("Full year earnings are not expected to meet guidance")
        assert result.matched == 0
        assert result.label() == "neutral"

    def test_a_negated_negative_becomes_positive(self):
        result = lexicon.score_text("The company was cleared of any wrongdoing")
        assert result.terms == (("wrongdoing", 1),)
        assert result.label() == "positive"

    def test_negation_does_not_reach_beyond_the_window(self):
        window = lexicon.NEGATION_WINDOW
        inside = "not " + "very " * (window - 1) + "strong"
        outside = "not " + "very " * window + "strong"
        assert lexicon.score_text(inside).score == -1.0
        assert lexicon.score_text(outside).score == 1.0

    def test_two_negators_in_the_window_cancel(self):
        # "not without difficulty" is difficult, so the sign must survive both.
        assert lexicon.score_text("difficulty").score == -1.0
        assert lexicon.score_text("without difficulty").score == 1.0
        assert lexicon.score_text("not without difficulty").score == -1.0

    def test_is_negated_reads_only_backwards(self):
        tokens = ["strong", "not", "weak"]
        assert not lexicon.is_negated(tokens, 0)
        assert lexicon.is_negated(tokens, 2)

    def test_the_window_is_a_parameter(self):
        assert lexicon.score_text("not very very very strong", window=5).score == -1.0


class TestClassify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The auditor has attached a going concern warning", "negative"),
            ("Margins improved and the order book strengthened", "positive"),
            ("The half year results will be published on 3 September", "neutral"),
        ],
    )
    def test_classify_matches_the_label_of_the_score(self, text, expected):
        assert lexicon.classify(text) == expected
        assert lexicon.polarity(text) == lexicon.score_text(text).score

    def test_the_threshold_is_a_parameter(self):
        result = lexicon.score_text("gains, improvement and one loss")
        assert result.score == pytest.approx(1 / 3)
        assert result.label(threshold=0.05) == "positive"
        assert result.label(threshold=0.5) == "neutral"


class TestLabelledSet:
    @pytest.fixture(scope="class")
    def headlines(self):
        return load_headlines()

    def test_the_file_is_where_the_module_says_it_is(self):
        assert HEADLINES_PATH.is_file()

    def test_the_set_is_large_enough_to_measure_with(self, headlines):
        assert len(headlines) >= 60

    def test_every_label_is_from_the_allowed_set(self, headlines):
        assert set(headlines["label"]) <= set(LABELS)

    def test_all_three_classes_are_present_and_none_dominates(self, headlines):
        counts = headlines["label"].value_counts()
        assert set(counts.index) == set(LABELS)
        assert counts.min() >= 20
        # Nothing so unbalanced that always guessing one class scores well.
        assert counts.max() / len(headlines) < 0.5

    def test_no_text_is_empty(self, headlines):
        assert not headlines["text"].str.strip().eq("").any()

    def test_no_text_is_duplicated(self, headlines):
        assert not headlines["text"].duplicated().any()

    def test_no_text_carries_a_replacement_character_or_a_dash(self, headlines):
        # The same encoding damage that fintech.reviews guards against, plus the
        # house rule against em and en dashes.
        joined = "".join(headlines["text"])
        for bad in ("�", "—", "–"):
            assert bad not in joined

    def test_the_hard_cases_the_set_was_written_for_are_in_it(self, headlines):
        joined = " ".join(headlines["text"]).lower()
        for phrase in ("not expected to meet", "aggressive", "liabilities", "cut costs"):
            assert phrase in joined

    def test_a_malformed_file_is_refused(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("text,label\nsomething happened,upbeat\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown labels"):
            load_headlines(path)

    def test_a_duplicate_row_is_refused(self, tmp_path):
        path = tmp_path / "dupe.csv"
        path.write_text(
            "text,label\nprofit warning,negative\nprofit warning,negative\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_headlines(path)

    def test_a_missing_column_is_refused(self, tmp_path):
        path = tmp_path / "thin.csv"
        path.write_text("text\nprofit warning\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing columns"):
            load_headlines(path)
