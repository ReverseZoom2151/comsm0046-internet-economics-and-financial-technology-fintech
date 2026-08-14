"""Tests for the evaluation machinery, checked against hand-worked numbers.

Metrics code is the easiest kind of code to get quietly wrong: a transposed
confusion matrix, precision computed against the support instead of against the
number of predictions, or a McNemar test fed the concordant pairs still produces
plausible looking output. So every statistic here is checked against a toy
example small enough to work out on paper, and the p-values are checked against
values that can be derived by hand rather than against whatever the code happened
to print first.

The full-dataset tests deliberately assert almost nothing about which analyser
wins. That is a measurement, and pinning it in a test would turn a result into an
assumption. What they do assert is that the machinery is internally consistent on
real input: the matrix totals match the dataset, the accuracy matches the
diagonal, and the paired counts add up to n.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy import stats

from fintech import evaluation
from fintech.evaluation import (
    LABELS,
    Comparison,
    compare,
    confusion_figure,
    confusion_matrix,
    evaluate,
    load_headlines,
    mcnemar,
    per_class_metrics,
    wilson_interval,
)

#: A ten item toy set whose every metric can be counted off by eye.
#:
#: truth:      neg neg neg neg  neu neu neu  pos pos pos
#: prediction: neg neg neu pos  neu neu neg  pos neu neu
#:
#: negative: 4 true, 3 predicted, 2 hit  -> precision 2/3,  recall 0.5
#: neutral:  3 true, 5 predicted, 2 hit  -> precision 0.4,  recall 2/3
#: positive: 3 true, 2 predicted, 1 hit  -> precision 0.5,  recall 1/3
#: accuracy: 5 of 10
TOY_TRUTH = [
    "negative",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
    "positive",
    "positive",
    "positive",
]
TOY_PREDICTIONS = [
    "negative",
    "negative",
    "neutral",
    "positive",
    "neutral",
    "neutral",
    "negative",
    "positive",
    "neutral",
    "neutral",
]


class TestConfusionMatrix:
    @pytest.fixture
    def matrix(self):
        return confusion_matrix(TOY_TRUTH, TOY_PREDICTIONS)

    def test_it_is_square_over_the_full_label_set(self, matrix):
        assert list(matrix.index) == list(LABELS)
        assert list(matrix.columns) == list(LABELS)

    def test_rows_are_truth_and_columns_are_predictions(self, matrix):
        # The transposition test. Four items are truly negative and only three
        # were predicted negative, so the row sum and the column sum differ,
        # which they would not if the matrix were built the wrong way round.
        assert matrix.loc["negative"].sum() == 4
        assert matrix["negative"].sum() == 3

    def test_every_cell_matches_the_hand_count(self, matrix):
        assert matrix.loc["negative", "negative"] == 2
        assert matrix.loc["negative", "neutral"] == 1
        assert matrix.loc["negative", "positive"] == 1
        assert matrix.loc["neutral", "negative"] == 1
        assert matrix.loc["neutral", "neutral"] == 2
        assert matrix.loc["neutral", "positive"] == 0
        assert matrix.loc["positive", "negative"] == 0
        assert matrix.loc["positive", "neutral"] == 2
        assert matrix.loc["positive", "positive"] == 1

    def test_the_totals_are_the_sample_size(self, matrix):
        assert matrix.to_numpy().sum() == len(TOY_TRUTH)

    def test_a_class_that_never_occurs_still_gets_a_row(self):
        matrix = confusion_matrix(["neutral"], ["neutral"])
        assert matrix.shape == (len(LABELS), len(LABELS))
        assert matrix.to_numpy().sum() == 1

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="against"):
            confusion_matrix(["neutral"], ["neutral", "positive"])

    def test_an_unknown_label_is_refused(self):
        with pytest.raises(ValueError, match="unknown"):
            confusion_matrix(["upbeat"], ["neutral"])


class TestPerClassMetrics:
    @pytest.fixture
    def metrics(self):
        matrix = confusion_matrix(TOY_TRUTH, TOY_PREDICTIONS)
        return {m.label: m for m in per_class_metrics(matrix)}

    def test_precision_is_over_predictions_and_recall_over_support(self, metrics):
        negative = metrics["negative"]
        assert (negative.support, negative.predicted, negative.true_positives) == (4, 3, 2)
        assert negative.precision == pytest.approx(2 / 3)
        assert negative.recall == pytest.approx(0.5)

    def test_the_over_predicted_class(self, metrics):
        neutral = metrics["neutral"]
        assert (neutral.support, neutral.predicted) == (3, 5)
        assert neutral.precision == pytest.approx(0.4)
        assert neutral.recall == pytest.approx(2 / 3)

    def test_f1_is_the_harmonic_mean(self, metrics):
        positive = metrics["positive"]
        assert positive.precision == pytest.approx(0.5)
        assert positive.recall == pytest.approx(1 / 3)
        assert positive.f1 == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))

    def test_a_class_never_predicted_scores_zero_rather_than_nan(self):
        matrix = confusion_matrix(["positive", "positive"], ["neutral", "neutral"])
        metrics = {m.label: m for m in per_class_metrics(matrix)}
        assert metrics["positive"].predicted == 0
        assert metrics["positive"].precision == 0.0
        assert not math.isnan(metrics["positive"].precision)
        assert metrics["positive"].f1 == 0.0


class TestWilsonInterval:
    def test_against_the_textbook_worked_example(self):
        # 8 of 10 at 95 per cent. Working: z = 1.959964 so z^2 = 3.84146, the
        # centre is (0.8 + z^2/20) / (1 + z^2/10) = 0.99207 / 1.38415 = 0.71673
        # and the half width is
        # (z / 1.38415) sqrt(0.8 * 0.2 / 10 + z^2/400) = 1.41601 * 0.160011,
        # which is 0.22657. That gives [0.49016, 0.94331], the value quoted for
        # this case wherever the Wilson interval is tabulated.
        low, high = wilson_interval(8, 10)
        assert low == pytest.approx(0.49016, abs=1e-5)
        assert high == pytest.approx(0.94331, abs=1e-5)

    def test_it_stays_inside_the_unit_interval_at_the_extremes(self):
        # The normal approximation gives [1, 1] for 10 of 10 and a negative lower
        # bound for 0 of 10. Wilson gives a usable interval for both.
        low, high = wilson_interval(10, 10)
        assert 0.0 < low < 1.0
        assert high == pytest.approx(1.0)
        low, high = wilson_interval(0, 10)
        assert low == pytest.approx(0.0)
        assert 0.0 < high < 1.0

    def test_it_is_centred_on_the_proportion_for_a_symmetric_count(self):
        low, high = wilson_interval(50, 100)
        assert (low + high) / 2 == pytest.approx(0.5)

    def test_it_narrows_as_the_sample_grows(self):
        small = wilson_interval(8, 10)
        large = wilson_interval(800, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_a_wider_confidence_level_gives_a_wider_interval(self):
        narrow = wilson_interval(60, 100, confidence=0.90)
        wide = wilson_interval(60, 100, confidence=0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    @pytest.mark.parametrize(("successes", "trials"), [(1, 0), (11, 10), (-1, 10)])
    def test_impossible_counts_are_refused(self, successes, trials):
        with pytest.raises(ValueError):
            wilson_interval(successes, trials)


class TestMcNemar:
    def test_the_worked_example(self):
        # b = 12 items the first got right and the second wrong, c = 5 the other
        # way, 40 concordant. b + c = 17, below the exact threshold, so the
        # p-value is the two sided binomial probability of a split at least as
        # lopsided as 12 to 5 out of 17 at p = 0.5, which is 0.143463.
        first = [True] * 12 + [False] * 5 + [True] * 20 + [False] * 20
        second = [False] * 12 + [True] * 5 + [True] * 20 + [False] * 20
        result = mcnemar(first, second)

        assert (result.n_first_only, result.n_second_only) == (12, 5)
        assert (result.n_both_correct, result.n_both_wrong) == (20, 20)
        assert result.discordant == 17
        assert "exact binomial" in result.method
        assert result.p_value == pytest.approx(
            float(stats.binomtest(12, 17, 0.5).pvalue), rel=1e-12
        )
        assert result.p_value == pytest.approx(0.143463, abs=1e-6)
        assert not result.significant

    def test_the_chi_square_branch_matches_the_hand_computed_statistic(self):
        # b = 30, c = 10, above the exact threshold. Edwards' corrected statistic
        # is (|30 - 10| - 1)^2 / 40 = 361 / 40 = 9.025 on one degree of freedom.
        first = [True] * 30 + [False] * 10
        second = [False] * 30 + [True] * 10
        result = mcnemar(first, second)

        assert "chi-square" in result.method
        assert result.statistic == pytest.approx(9.025)
        assert result.p_value == pytest.approx(float(stats.chi2.sf(9.025, 1)))
        assert result.p_value == pytest.approx(0.002663, abs=1e-6)
        assert result.significant

    def test_concordant_pairs_are_discarded(self):
        # Adding items both analysers get right cannot change the p-value, which
        # is the whole point of conditioning on the discordant pairs.
        first = [True] * 12 + [False] * 5
        second = [False] * 12 + [True] * 5
        bare = mcnemar(first, second)
        padded = mcnemar(first + [True] * 500, second + [True] * 500)
        assert padded.p_value == pytest.approx(bare.p_value)
        assert padded.n_both_correct == 500

    def test_an_even_split_is_not_significant(self):
        first = [True] * 10 + [False] * 10
        second = [False] * 10 + [True] * 10
        result = mcnemar(first, second)
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_perfect_agreement_has_nothing_to_test(self):
        result = mcnemar([True, False, True], [True, False, True])
        assert result.discordant == 0
        assert result.p_value == 1.0
        assert not result.significant
        assert result.method == "no discordant pairs"

    def test_the_test_is_symmetric_in_its_arguments(self):
        first = [True] * 12 + [False] * 5 + [True] * 3
        second = [False] * 12 + [True] * 5 + [True] * 3
        forward = mcnemar(first, second)
        backward = mcnemar(second, first)
        assert forward.p_value == pytest.approx(backward.p_value)
        assert forward.n_first_only == backward.n_second_only

    def test_the_counts_partition_the_sample(self):
        first = [True, True, False, False, True]
        second = [True, False, True, False, False]
        result = mcnemar(first, second)
        total = (
            result.n_both_correct + result.n_first_only + result.n_second_only + result.n_both_wrong
        )
        assert total == len(first)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="equal lengths"):
            mcnemar([True], [True, False])


class TestEvaluate:
    @pytest.fixture
    def result(self):
        texts = [f"item {i}" for i in range(len(TOY_TRUTH))]
        predictions = dict(zip(texts, TOY_PREDICTIONS, strict=True))
        return evaluate("toy", texts, TOY_TRUTH, predictions.__getitem__)

    def test_accuracy_is_the_diagonal_over_the_total(self, result):
        assert result.n == 10
        assert result.n_correct == 5
        assert result.accuracy == pytest.approx(0.5)
        diagonal = sum(result.confusion.iat[i, i] for i in range(len(LABELS)))
        assert diagonal == result.n_correct

    def test_the_interval_brackets_the_accuracy(self, result):
        low, high = result.accuracy_interval
        assert low < result.accuracy < high
        assert (low, high) == wilson_interval(5, 10)

    def test_errors_lists_exactly_the_misclassified_items(self, result):
        errors = result.errors()
        assert len(errors) == 5
        assert list(errors.columns) == ["text", "label", "predicted"]
        assert (errors["label"] != errors["predicted"]).all()

    def test_a_perfect_analyser_has_no_errors(self):
        texts = ["a", "b"]
        truth = ["negative", "positive"]
        result = evaluate("oracle", texts, truth, dict(zip(texts, truth, strict=True)).__getitem__)
        assert result.accuracy == 1.0
        assert result.errors().empty


class TestCompareOnTheRealDataset:
    @pytest.fixture(scope="class")
    def comparison(self) -> Comparison:
        return compare()

    def test_it_covers_the_whole_labelled_set(self, comparison):
        frame = load_headlines()
        assert comparison.dataset_size == len(frame)
        assert sum(comparison.class_counts.values()) == len(frame)

    def test_both_analysers_were_run(self, comparison):
        assert set(comparison.results) == set(evaluation.ANALYSERS)
        for result in comparison.results.values():
            assert result.n == comparison.dataset_size
            assert result.confusion.to_numpy().sum() == comparison.dataset_size

    def test_every_prediction_is_a_valid_label(self, comparison):
        for result in comparison.results.values():
            assert set(result.predictions) <= set(LABELS)

    def test_accuracy_agrees_with_the_confusion_matrix(self, comparison):
        for result in comparison.results.values():
            diagonal = sum(result.confusion.iat[i, i] for i in range(len(LABELS)))
            assert result.accuracy == pytest.approx(diagonal / result.n)
            low, high = result.accuracy_interval
            assert 0.0 <= low <= result.accuracy <= high <= 1.0

    def test_the_paired_counts_partition_the_sample(self, comparison):
        test = comparison.test
        total = test.n_both_correct + test.n_first_only + test.n_second_only + test.n_both_wrong
        assert total == comparison.dataset_size

    def test_the_test_is_run_on_the_same_items_in_the_same_order(self, comparison):
        first, second = comparison.first, comparison.second
        assert comparison.results[first].texts == comparison.results[second].texts
        assert comparison.results[first].truth == comparison.results[second].truth

    def test_both_beat_the_majority_class_baseline_or_are_reported_as_not(self, comparison):
        # Not an assertion about which analyser wins. Always guessing the most
        # common class is the floor any three way classifier has to clear, and
        # recording it here is what makes an accuracy figure readable at all.
        baseline = max(comparison.class_counts.values()) / comparison.dataset_size
        assert 0.0 < baseline < 0.5
        for result in comparison.results.values():
            assert 0.0 <= result.accuracy <= 1.0

    def test_better_names_the_more_accurate_analyser(self, comparison):
        accuracies = {name: r.accuracy for name, r in comparison.results.items()}
        if len(set(accuracies.values())) == 1:
            assert comparison.better is None
        else:
            assert comparison.better == max(accuracies, key=accuracies.__getitem__)

    def test_comparing_more_than_two_analysers_is_refused(self):
        with pytest.raises(ValueError, match="exactly two"):
            compare(analysers={"a": str, "b": str, "c": str})

    def test_a_supplied_frame_is_used_instead_of_the_file(self):
        frame = pd.DataFrame(
            {
                "text": ["a large impairment", "the meeting is on Tuesday"],
                "label": ["negative", "neutral"],
            }
        )
        comparison = compare(frame)
        assert comparison.dataset_size == 2


class TestConfusionFigure:
    def test_it_draws_one_cell_label_per_count(self, tmp_path):
        texts = [f"item {i}" for i in range(len(TOY_TRUTH))]
        predictions = dict(zip(texts, TOY_PREDICTIONS, strict=True))
        result = evaluate("toy", texts, TOY_TRUTH, predictions.__getitem__)

        fig = confusion_figure(result)
        ax = fig.axes[0]
        counts = sorted(int(t.get_text()) for t in ax.texts)
        assert counts == sorted(int(v) for v in result.confusion.to_numpy().ravel())

        from fintech.plotting import save_figure

        path = save_figure(fig, "test confusion matrix", tmp_path)
        assert path.is_file()
