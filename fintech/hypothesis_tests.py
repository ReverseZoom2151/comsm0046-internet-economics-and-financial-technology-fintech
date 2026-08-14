"""An assumption-aware hypothesis-testing pipeline for the week 5 activity.

The activity notebooks run Shapiro-Wilk for normality and then run a one-way
ANOVA regardless of the answer. Nothing branches on the normality result, so
the test that gets run is fixed in advance and the preceding test is decorative.
For data2 that is not a stylistic complaint: both of its columns fail
Shapiro-Wilk (p = 0.0118 and p = 0.0303), and the notebook then runs a
parametric test whose central assumption its own previous cell has just
rejected. The conclusion happens to survive, because Kruskal-Wallis agrees that
the difference is significant, but the method used to reach it was not
justified by the data.

Three further gaps in the notebook version:

* ANOVA also assumes equal variance across groups and the notebooks never test
  it. Levene's test says the assumption does hold here (p = 0.594 for data1,
  p = 0.683 for data2), but that was luck rather than a check.
* ANOVA is an omnibus test. For data1's three conditions a significant result
  would not say which conditions differ, and there is no post-hoc anywhere.
* Shapiro-Wilk is run once per column with no correction for multiple
  comparisons, and at n = 15 and n = 20 it has so little power that failing to
  reject normality is close to a foregone conclusion. This module applies a
  Holm correction and reports the sample sizes so the weakness of the evidence
  for normality is visible rather than implied.

So the pipeline here tests the assumptions first and then chooses the omnibus
test from the answers: one-way ANOVA when the data are normal with equal
variances, Welch's ANOVA when they are normal with unequal variances, and
Kruskal-Wallis when normality fails. Every result carries the reason the test
was chosen. Results are returned as structured objects rather than printed, so
they can be asserted on in tests, which is how the choice of test for each
dataset is now pinned down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from fintech.plotting import new_figure, save_figure

#: Shapiro-Wilk needs at least three points, and a group with no spread breaks
#: every test here, so these are the limits the pipeline refuses below.
MIN_OBSERVATIONS = 3
MIN_GROUPS = 2

DEFAULT_ALPHA = 0.05


class DegenerateDataError(ValueError):
    """Raised when the input cannot support a hypothesis test at all.

    scipy will happily return NaN for a single group, a constant column or two
    observations. A NaN p-value silently compares false against any threshold,
    so the caller concludes "not significant" from a test that never ran. We
    refuse the input instead.
    """


@dataclass(frozen=True)
class NormalityResult:
    """Shapiro-Wilk for one condition, with the Holm-corrected p-value."""

    condition: str
    statistic: float
    p_value: float
    p_value_corrected: float
    n: int
    normal: bool


@dataclass(frozen=True)
class VarianceResult:
    """Levene's test for equality of variance across all conditions."""

    statistic: float
    p_value: float
    equal_variance: bool


@dataclass(frozen=True)
class OmnibusResult:
    """The omnibus test that was chosen, and why."""

    test: str
    statistic: float
    p_value: float
    df_between: float | None
    df_within: float | None
    significant: bool
    reason: str
    equivalent_to: str | None = None


@dataclass(frozen=True)
class PostHocResult:
    """Tukey HSD, or the reason no post-hoc was run."""

    test: str | None
    comparisons: tuple[dict[str, object], ...] = ()
    reason: str = ""
    summary: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the pipeline concluded about one dataset."""

    name: str
    conditions: tuple[str, ...]
    n_per_condition: dict[str, int]
    summary: pd.DataFrame
    normality: tuple[NormalityResult, ...]
    variance: VarianceResult
    omnibus: OmnibusResult
    post_hoc: PostHocResult
    alpha: float = DEFAULT_ALPHA
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_normal(self) -> bool:
        return all(result.normal for result in self.normality)


def _validate(frame: pd.DataFrame) -> list[np.ndarray]:
    """Turn a dataframe into one clean array per condition, or refuse it."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame with one column per condition")

    if frame.shape[1] < MIN_GROUPS:
        raise DegenerateDataError(
            f"need at least {MIN_GROUPS} conditions to compare, got {frame.shape[1]}; "
            "a single group has nothing to be tested against"
        )

    groups: list[np.ndarray] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        values = series.to_numpy(dtype=float)

        if values.size < MIN_OBSERVATIONS:
            raise DegenerateDataError(
                f"condition {column!r} has {values.size} usable observations, "
                f"fewer than the {MIN_OBSERVATIONS} needed for a normality test"
            )
        if np.ptp(values) == 0.0:
            raise DegenerateDataError(
                f"condition {column!r} is constant (every value is {values[0]!r}); "
                "a group with zero variance makes every test statistic undefined"
            )
        groups.append(values)

    return groups


def holm_correction(p_values) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    The notebooks run one normality test per condition and read each p-value
    against 0.05 as if it were the only test performed. With three conditions
    that inflates the chance of spuriously rejecting normality somewhere.
    """

    raw = np.asarray(list(p_values), dtype=float)
    m = raw.size
    if m == 0:
        return raw

    order = np.argsort(raw)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * raw[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def shapiro_normality(
    frame: pd.DataFrame, alpha: float = DEFAULT_ALPHA
) -> tuple[NormalityResult, ...]:
    """Shapiro-Wilk per condition, judged on Holm-corrected p-values."""

    groups = _validate(frame)
    statistics, p_values = [], []
    for values in groups:
        statistic, p_value = stats.shapiro(values)
        statistics.append(float(statistic))
        p_values.append(float(p_value))

    corrected = holm_correction(p_values)
    return tuple(
        NormalityResult(
            condition=str(column),
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=float(adjusted),
            n=int(values.size),
            normal=bool(adjusted >= alpha),
        )
        for column, values, statistic, p_value, adjusted in zip(
            frame.columns, groups, statistics, p_values, corrected, strict=True
        )
    )


def levene_equal_variance(frame: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> VarianceResult:
    """Levene's test, the assumption the notebooks never checked.

    Levene is used rather than Bartlett because Bartlett is itself sensitive to
    non-normality, and non-normality is exactly the case we need to survive.
    """

    groups = _validate(frame)
    statistic, p_value = stats.levene(*groups, center="median")
    return VarianceResult(
        statistic=float(statistic),
        p_value=float(p_value),
        equal_variance=bool(p_value >= alpha),
    )


def welch_anova(groups) -> tuple[float, float, float, float]:
    """Welch's heteroscedastic one-way ANOVA: F, p, df_between, df_within.

    scipy has no Welch ANOVA, only Welch's t-test for two groups, so the
    statistic is computed directly. For two groups it reduces to that t-test,
    with F equal to t squared.
    """

    arrays = [np.asarray(values, dtype=float) for values in groups]
    k = len(arrays)
    n = np.array([values.size for values in arrays], dtype=float)
    means = np.array([values.mean() for values in arrays])
    variances = np.array([values.var(ddof=1) for values in arrays])

    weights = n / variances
    total_weight = weights.sum()
    grand_mean = float((weights * means).sum() / total_weight)

    numerator = float((weights * (means - grand_mean) ** 2).sum()) / (k - 1)
    lam = float((((1.0 - weights / total_weight) ** 2) / (n - 1.0)).sum())
    denominator = 1.0 + (2.0 * (k - 2) / (k**2 - 1.0)) * lam

    f_statistic = numerator / denominator
    df_between = float(k - 1)
    df_within = (k**2 - 1.0) / (3.0 * lam)
    p_value = float(stats.f.sf(f_statistic, df_between, df_within))
    return float(f_statistic), p_value, df_between, df_within


def choose_omnibus_test(
    frame: pd.DataFrame,
    normality: tuple[NormalityResult, ...],
    variance: VarianceResult,
    alpha: float = DEFAULT_ALPHA,
) -> OmnibusResult:
    """Pick the omnibus test the assumptions actually license, and run it."""

    groups = _validate(frame)
    k = len(groups)
    failed = [result.condition for result in normality if not result.normal]
    two_group_note = (
        "with two conditions this omnibus test is equivalent to the corresponding two-sample test"
    )

    if failed:
        statistic, p_value = stats.kruskal(*groups)
        reason = (
            f"Shapiro-Wilk rejects normality for {', '.join(failed)} after a Holm "
            f"correction, so a parametric ANOVA is not justified; Kruskal-Wallis "
            f"makes no normality assumption."
        )
        return OmnibusResult(
            test="Kruskal-Wallis H",
            statistic=float(statistic),
            p_value=float(p_value),
            df_between=float(k - 1),
            df_within=None,
            significant=bool(p_value < alpha),
            reason=reason,
            equivalent_to="Mann-Whitney U" if k == 2 else None,
        )

    if not variance.equal_variance:
        statistic, p_value, df_between, df_within = welch_anova(groups)
        reason = (
            f"all conditions pass the normality test, but Levene's test rejects equal "
            f"variance (p = {variance.p_value:.4g}), so the classical ANOVA's pooled "
            f"variance is wrong; Welch's ANOVA does not pool."
        )
        return OmnibusResult(
            test="Welch ANOVA",
            statistic=statistic,
            p_value=p_value,
            df_between=df_between,
            df_within=df_within,
            significant=bool(p_value < alpha),
            reason=reason,
            equivalent_to="Welch's t-test" if k == 2 else None,
        )

    statistic, p_value = stats.f_oneway(*groups)
    total_n = sum(values.size for values in groups)
    reason = (
        f"all conditions pass Shapiro-Wilk and Levene's test does not reject equal "
        f"variance (p = {variance.p_value:.4g}), so both assumptions of the classical "
        f"one-way ANOVA hold."
    )
    return OmnibusResult(
        test="one-way ANOVA",
        statistic=float(statistic),
        p_value=float(p_value),
        df_between=float(k - 1),
        df_within=float(total_n - k),
        significant=bool(p_value < alpha),
        reason=reason if k > 2 else f"{reason} {two_group_note}.",
        equivalent_to="independent two-sample t-test (F = t squared)" if k == 2 else None,
    )


def tukey_post_hoc(frame: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> PostHocResult:
    """Tukey HSD across every pair of conditions.

    An omnibus test says only that the conditions are not all alike. With three
    conditions that leaves three possible pairwise stories, and the notebooks
    stop before distinguishing them.
    """

    groups = _validate(frame)
    labels: list[str] = []
    values: list[float] = []
    for column, group in zip(frame.columns, groups, strict=True):
        labels.extend([str(column)] * group.size)
        values.extend(group.tolist())

    result = pairwise_tukeyhsd(endog=np.asarray(values), groups=np.asarray(labels), alpha=alpha)
    table = result.summary()
    comparisons = tuple(
        {
            "group1": str(row[0]),
            "group2": str(row[1]),
            "mean_difference": float(row[2]),
            "p_value": float(row[3]),
            "lower": float(row[4]),
            "upper": float(row[5]),
            "reject": str(row[6]).strip().lower() == "true",
        }
        for row in (list(map(str, data)) for data in table.data[1:])
    )
    return PostHocResult(
        test="Tukey HSD",
        comparisons=comparisons,
        reason="the omnibus test is significant across more than two conditions",
        summary=str(table),
    )


def analyse(
    frame: pd.DataFrame, name: str = "dataset", alpha: float = DEFAULT_ALPHA
) -> AnalysisResult:
    """Run the whole assumption-aware pipeline over one dataset."""

    groups = _validate(frame)
    conditions = tuple(str(column) for column in frame.columns)
    normality = shapiro_normality(frame, alpha=alpha)
    variance = levene_equal_variance(frame, alpha=alpha)
    omnibus = choose_omnibus_test(frame, normality, variance, alpha=alpha)

    k = len(groups)
    if k == MIN_GROUPS:
        post_hoc = PostHocResult(
            test=None,
            reason=(
                "only two conditions, so the omnibus test already identifies the one "
                "pairwise difference and a post-hoc would add nothing"
            ),
        )
    elif not omnibus.significant:
        post_hoc = PostHocResult(
            test=None,
            reason=(
                f"the omnibus test is not significant at alpha = {alpha}, so there is no "
                "overall difference to decompose into pairwise ones"
            ),
        )
    elif not all(result.normal for result in normality):
        post_hoc = PostHocResult(
            test=None,
            reason=(
                "Tukey HSD assumes normality, which these data fail; Dunn's test would "
                "be the rank-based alternative"
            ),
        )
    else:
        post_hoc = tukey_post_hoc(frame, alpha=alpha)

    smallest = min(values.size for values in groups)
    if all(result.normal for result in normality):
        power_note = (
            f"Shapiro-Wilk has low power at n = {smallest}, so failing to reject normality "
            "is weak evidence of it rather than a demonstration"
        )
    else:
        power_note = (
            f"Shapiro-Wilk has low power at n = {smallest}, so rejecting normality here "
            "means the departure was large enough to show up despite that"
        )
    notes = [
        power_note,
        (
            f"{len(normality)} normality tests were run on this dataset; the p-values "
            "reported as corrected use a Holm step-down adjustment"
        ),
    ]
    if k == MIN_GROUPS:
        notes.append(
            "with two conditions a one-way ANOVA reduces exactly to an independent "
            "two-sample t-test, where F = t squared"
        )

    return AnalysisResult(
        name=name,
        conditions=conditions,
        n_per_condition={
            condition: int(values.size)
            for condition, values in zip(conditions, groups, strict=True)
        },
        summary=frame.describe(),
        normality=normality,
        variance=variance,
        omnibus=omnibus,
        post_hoc=post_hoc,
        alpha=alpha,
        notes=tuple(notes),
    )


def qq_figures(frame: pd.DataFrame, name: str = "dataset") -> dict[str, object]:
    """One QQ plot per condition, against a standard normal."""

    _validate(frame)
    standardised = (frame - frame.mean()) / frame.std()
    figures: dict[str, object] = {}
    for column in frame.columns:
        fig, ax = new_figure(5.0, 5.0)
        sm.qqplot(standardised[column].to_numpy(dtype=float), line="45", ax=ax)
        ax.set_title(f"QQ plot, {name} condition {column}")
        figures[f"{name}_qq_{column}"] = fig
    return figures


def histogram_figures(frame: pd.DataFrame, name: str = "dataset") -> dict[str, object]:
    """One histogram with a kernel density estimate per condition."""

    _validate(frame)
    figures: dict[str, object] = {}
    for column in frame.columns:
        fig, ax = new_figure(6.0, 4.0)
        sns.histplot(frame[column], kde=True, ax=ax)
        ax.set_title(f"Histogram, {name} condition {column}")
        ax.set_xlabel("value")
        figures[f"{name}_histogram_{column}"] = fig
    return figures


def bar_figure(frame: pd.DataFrame, name: str = "dataset"):
    """Group means with standard deviation error bars, honestly labelled.

    The notebooks call `sns.barplot(data=df, ci='sd')`. The `ci` keyword was
    removed in seaborn 0.13 in favour of `errorbar`, so that call raises a
    FutureWarning, and the title claimed a confidence interval while the
    argument asked for a standard deviation. Those are different quantities:
    the standard deviation describes the spread of the observations, the
    confidence interval describes the uncertainty in the mean. The title now
    says what is actually drawn.
    """

    _validate(frame)
    fig, ax = new_figure(6.0, 4.0)
    sns.barplot(data=frame, errorbar="sd", ax=ax)
    ax.set_title(f"Mean per condition with standard deviation, {name}")
    ax.set_xlabel("condition")
    ax.set_ylabel("value")
    return fig


def box_figure(frame: pd.DataFrame, name: str = "dataset"):
    """Inter-quartile ranges per condition."""

    _validate(frame)
    fig, ax = new_figure(6.0, 4.0)
    sns.boxplot(data=frame, ax=ax)
    ax.set_title(f"Inter-quartile ranges, {name}")
    ax.set_xlabel("condition")
    ax.set_ylabel("value")
    return fig


def violin_figure(frame: pd.DataFrame, name: str = "dataset"):
    """Full distribution shape per condition."""

    _validate(frame)
    fig, ax = new_figure(6.0, 4.0)
    sns.violinplot(data=frame, ax=ax)
    ax.set_title(f"Distribution shape by condition, {name}")
    ax.set_xlabel("condition")
    ax.set_ylabel("value")
    return fig


def all_figures(frame: pd.DataFrame, name: str = "dataset") -> dict[str, object]:
    """Every figure the activity asks for, keyed by the name to save it under."""

    figures: dict[str, object] = {}
    figures.update(qq_figures(frame, name))
    figures.update(histogram_figures(frame, name))
    figures[f"{name}_bar_mean_sd"] = bar_figure(frame, name)
    figures[f"{name}_box"] = box_figure(frame, name)
    figures[f"{name}_violin"] = violin_figure(frame, name)
    return figures


def save_figures(figures: dict[str, object], directory: Path | str) -> list[Path]:
    """Write figures to disk through the project's sanitising saver."""

    return [save_figure(fig, title, directory) for title, fig in figures.items()]
