"""The pipeline must choose its omnibus test from the assumptions, not in advance.

The regression the whole module exists to prevent is the notebook behaviour:
Shapiro-Wilk runs, its answer is ignored, and a one-way ANOVA runs regardless.
So the load-bearing tests here are that data1 routes to ANOVA and data2 routes
to Kruskal-Wallis, because data2's columns both fail the normality test that
ANOVA depends on.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fintech.datasets import load_dataset  # noqa: E402
from fintech.hypothesis_tests import (  # noqa: E402
    DegenerateDataError,
    all_figures,
    analyse,
    bar_figure,
    holm_correction,
    levene_equal_variance,
    save_figures,
    shapiro_normality,
    tukey_post_hoc,
    welch_anova,
)


@pytest.fixture(scope="module")
def data1():
    return load_dataset("data1")


@pytest.fixture(scope="module")
def data2():
    return load_dataset("data2")


def normal_equal_variance_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(10.0, 2.0, 40),
            "b": rng.normal(10.4, 2.0, 40),
            "c": rng.normal(10.2, 2.0, 40),
        }
    )


# --- the point of the exercise -------------------------------------------------


def test_data1_routes_to_anova(data1):
    result = analyse(data1, name="data1")
    assert result.all_normal
    assert result.variance.equal_variance
    assert result.omnibus.test == "one-way ANOVA"
    assert not result.omnibus.significant
    assert result.omnibus.p_value == pytest.approx(0.0889, abs=5e-4)


def test_data2_routes_to_kruskal_wallis_because_normality_fails(data2):
    result = analyse(data2, name="data2")
    assert not result.all_normal
    assert [n.condition for n in result.normality if not n.normal] == ["x", "y"]
    assert result.omnibus.test == "Kruskal-Wallis H"
    assert result.omnibus.significant
    assert result.omnibus.p_value == pytest.approx(0.00243, abs=5e-5)
    assert "normality" in result.omnibus.reason


def test_data2_is_not_analysed_with_anova(data2):
    """The notebooks ran ANOVA on data2 after their own test rejected normality."""

    result = analyse(data2, name="data2")
    assert "ANOVA" not in result.omnibus.test


def test_data2_conclusion_survives_the_corrected_method(data2):
    """The method was unjustified; the answer happens to be the same either way."""

    anova_p = stats.f_oneway(data2["x"], data2["y"]).pvalue
    result = analyse(data2, name="data2")
    assert anova_p < 0.05
    assert result.omnibus.significant


def test_a_normal_equal_variance_dataset_routes_to_anova():
    frame = normal_equal_variance_frame()
    result = analyse(frame, name="synthetic")
    assert result.all_normal
    assert result.variance.equal_variance
    assert result.omnibus.test == "one-way ANOVA"


def test_unequal_variance_routes_to_welch_anova():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "a": rng.normal(10.0, 1.0, 40),
            "b": rng.normal(10.0, 8.0, 40),
            "c": rng.normal(10.0, 1.0, 40),
        }
    )
    result = analyse(frame, name="heteroscedastic")
    assert result.all_normal
    assert not result.variance.equal_variance
    assert result.omnibus.test == "Welch ANOVA"


# --- the statistics agree with scipy computed directly -------------------------


def test_normality_matches_scipy(data1):
    for normality in shapiro_normality(data1):
        statistic, p_value = stats.shapiro(data1[normality.condition])
        assert normality.statistic == pytest.approx(statistic)
        assert normality.p_value == pytest.approx(p_value)
        assert normality.p_value_corrected >= normality.p_value


def test_levene_matches_scipy(data1):
    variance = levene_equal_variance(data1)
    statistic, p_value = stats.levene(*[data1[c] for c in data1.columns], center="median")
    assert variance.statistic == pytest.approx(statistic)
    assert variance.p_value == pytest.approx(p_value)
    assert variance.p_value == pytest.approx(0.5935, abs=5e-4)


def test_anova_matches_scipy(data1):
    result = analyse(data1, name="data1")
    statistic, p_value = stats.f_oneway(*[data1[c] for c in data1.columns])
    assert result.omnibus.statistic == pytest.approx(statistic)
    assert result.omnibus.p_value == pytest.approx(p_value)
    assert result.omnibus.df_between == 2
    assert result.omnibus.df_within == 42


def test_kruskal_matches_scipy(data2):
    result = analyse(data2, name="data2")
    statistic, p_value = stats.kruskal(*[data2[c] for c in data2.columns])
    assert result.omnibus.statistic == pytest.approx(statistic)
    assert result.omnibus.p_value == pytest.approx(p_value)


def test_welch_anova_on_two_groups_equals_welch_t_test():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 25)
    b = rng.normal(1.0, 4.0, 25)
    f_statistic, p_value, df_between, df_within = welch_anova([a, b])
    t_result = stats.ttest_ind(a, b, equal_var=False)
    assert f_statistic == pytest.approx(t_result.statistic**2)
    assert p_value == pytest.approx(t_result.pvalue)
    assert df_between == 1
    assert df_within == pytest.approx(t_result.df)


def test_two_group_anova_is_flagged_as_a_t_test():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x": rng.normal(0.0, 1.0, 30), "y": rng.normal(0.3, 1.0, 30)})
    result = analyse(frame, name="pair")
    assert result.omnibus.test == "one-way ANOVA"
    assert "t squared" in result.omnibus.equivalent_to
    t_result = stats.ttest_ind(frame["x"], frame["y"])
    assert result.omnibus.statistic == pytest.approx(t_result.statistic**2)


def test_holm_correction_is_monotone_and_bounded():
    adjusted = holm_correction([0.01, 0.04, 0.5])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.08)
    assert adjusted[2] == pytest.approx(0.5)
    assert (holm_correction([0.9, 0.9]) <= 1.0).all()


# --- post-hoc ------------------------------------------------------------------


def test_tukey_runs_on_three_groups():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "a": rng.normal(10.0, 1.0, 30),
            "b": rng.normal(10.0, 1.0, 30),
            "c": rng.normal(15.0, 1.0, 30),
        }
    )
    post_hoc = tukey_post_hoc(frame)
    assert post_hoc.test == "Tukey HSD"
    assert len(post_hoc.comparisons) == 3
    pairs = {(c["group1"], c["group2"]): c["reject"] for c in post_hoc.comparisons}
    assert pairs[("a", "b")] is False
    assert pairs[("a", "c")] is True
    assert pairs[("b", "c")] is True


def test_significant_three_group_analysis_gets_a_post_hoc():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "a": rng.normal(10.0, 1.0, 30),
            "b": rng.normal(10.0, 1.0, 30),
            "c": rng.normal(14.0, 1.0, 30),
        }
    )
    result = analyse(frame, name="separated")
    assert result.omnibus.significant
    assert result.post_hoc.test == "Tukey HSD"


def test_no_post_hoc_when_the_omnibus_is_not_significant(data1):
    result = analyse(data1, name="data1")
    assert result.post_hoc.test is None
    assert "not significant" in result.post_hoc.reason


def test_no_post_hoc_for_two_groups(data2):
    result = analyse(data2, name="data2")
    assert result.post_hoc.test is None
    assert "two conditions" in result.post_hoc.reason


# --- degenerate inputs are refused, not silently turned into NaN ---------------


def test_a_single_group_is_refused():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(DegenerateDataError, match="at least 2 conditions"):
        analyse(frame)


def test_a_constant_column_is_refused():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0] * 4})
    with pytest.raises(DegenerateDataError, match="constant"):
        analyse(frame)


def test_too_few_observations_is_refused():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(DegenerateDataError, match="fewer than the 3"):
        analyse(frame)


def test_scipy_reports_nonsense_for_degenerate_input():
    """Why the refusal matters: scipy answers rather than complaining.

    Shapiro-Wilk on a constant column returns either p = 1.0, which reads as
    perfect normality for a column that has no distribution at all, or NaN.
    Which of the two depends on the scipy version, so this asserts only that the
    answer is unusable rather than pinning a number that moves between releases.
    ANOVA over two constant groups returns NaN, and a NaN p-value compares false
    against any threshold, so the caller concludes "not significant" from a test
    that never ran.

    Either way the result is worse than an error, because it is silent. That is
    what DegenerateDataError exists to prevent.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant = stats.shapiro([5.0, 5.0, 5.0, 5.0])
        with np.errstate(invalid="ignore"):
            anova = stats.f_oneway([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])

    assert np.isnan(constant.pvalue) or constant.pvalue == pytest.approx(1.0)
    assert not (constant.pvalue < 0.05), "a constant column must not read as non-normal"
    assert np.isnan(anova.pvalue)
    assert not (anova.pvalue < 0.05)


def test_a_non_dataframe_is_rejected():
    with pytest.raises(TypeError):
        analyse([[1.0, 2.0], [3.0, 4.0]])


# --- figures -------------------------------------------------------------------


def test_figures_are_written_without_deprecation_warnings(data1, tmp_path):
    """`sns.barplot(ci='sd')` was removed in seaborn 0.13; `errorbar='sd'` replaces it."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        figures = all_figures(data1, name="data1")

    # Three conditions: a QQ plot and a histogram each, plus bar, box and violin.
    assert len(figures) == 9
    paths = save_figures(figures, tmp_path)
    assert len(paths) == len(figures)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for figure in figures.values():
        plt.close(figure)


def test_the_bar_chart_is_labelled_as_a_standard_deviation(data1):
    """The notebook titled a standard deviation plot "Confidence Interval"."""

    figure = bar_figure(data1, name="data1")
    title = figure.axes[0].get_title().lower()
    assert "standard deviation" in title
    assert "confidence interval" not in title
    plt.close(figure)
