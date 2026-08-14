"""Measuring two sentiment analysers on the same labelled headlines.

`FINDINGS.md` claims that a general-purpose sentiment analyser is unsuited to
market text, and supports it with two anecdotes: the Associated Press hack tweet
of April 2013, which TextBlob scores at polarity 0.0000, and the Muddy Waters
tweet of August 2019, which it scores at +0.0341 the evening before its target
lost more than half its value. Both are real and both are striking. Neither is a
measurement, and two texts chosen because they are memorable are the worst
possible sample to generalise from. This module turns the claim into a number
with an interval around it.

**The labelled set is author written.** `data/headlines.csv` holds short market
announcements written for this repository, each labelled negative, neutral or
positive from the point of view of the asset being discussed. They are not a
published benchmark, they were not sampled from a newswire, and nobody else has
labelled them, so there is no inter-annotator agreement to quote. That bounds
what any accuracy figure here can claim: it measures the analysers on this set,
written by someone who knew both analysers existed, and it does not estimate
their accuracy on real newsflow. What it can do honestly is compare two
analysers on identical items, which is the comparison the anecdotes were being
used to make.

The set deliberately includes the cases that separate a finance lexicon from a
general one: negation ("not expected to meet guidance", "no longer expects to
need further funding"), words that are positive in general English and negative
in a financial context ("aggressive", and the "liability" of Loughran and
McDonald's title), words that are negative in general English and positive in a
financial one ("short interest has fallen", "cut costs ahead of schedule"), and
a block of genuinely neutral procedural announcements, because an analyser that
finds an opinion in a notice of meeting is as wrong as one that misses a profit
warning.

**Why McNemar's test.** The two analysers are run over the same items, so their
per-item outcomes are paired and heavily correlated: an easy headline is easy for
both, and any test that treats the two accuracies as independent samples ignores
that correlation and gets the standard error wrong. McNemar's test conditions on
exactly the items where the two disagree, counting how many one gets right and
the other wrong in each direction, and asks whether that split is compatible with
a fair coin. Items both get right and items both get wrong carry no information
about which analyser is better and are correctly discarded. With a handful of
discordant pairs the chi-square approximation is poor, so an exact binomial test
is used below a documented count and the chi-square with continuity correction
above it.

Accuracy is reported with a Wilson score interval rather than the textbook
normal approximation, which misbehaves near 0 and 1 and can produce a lower bound
below zero at the sample sizes involved here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

from fintech import lexicon
from fintech.plotting import new_figure
from fintech.sentiment import analyse_document

#: The label set, in the order used for every table and figure. Ordered from
#: negative to positive rather than alphabetically so that a confusion matrix
#: reads like a scale and the two serious errors, calling bad news good and good
#: news bad, sit in the two far corners.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

#: Where the labelled set lives, resolved from this file rather than from the
#: process working directory, for the reason given in `fintech.datasets`.
HEADLINES_PATH = Path(__file__).resolve().parent.parent / "data" / "headlines.csv"

#: Below this many discordant pairs, McNemar uses an exact binomial test. The
#: usual rule of thumb is 25; the chi-square approximation to the binomial is
#: unreliable below it.
EXACT_TEST_THRESHOLD = 25

DEFAULT_CONFIDENCE = 0.95


def load_headlines(path: Path | str | None = None) -> pd.DataFrame:
    """Read the labelled set, refusing anything malformed.

    Columns are `text` and `label`. Validation is not defensive decoration: a
    stray blank line or a mistyped label would quietly become a class of its own
    and every per-class metric below it would be computed against a label set
    that no longer matches `LABELS`.
    """

    frame = pd.read_csv(Path(path) if path is not None else HEADLINES_PATH)

    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"headlines file is missing columns: {sorted(missing)}")

    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()

    if frame["text"].eq("").any():
        raise ValueError("headlines file contains an empty text")
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"headlines file contains unknown labels: {unknown}")
    duplicated = frame["text"].duplicated()
    if duplicated.any():
        repeated = frame.loc[duplicated, "text"].tolist()
        raise ValueError(f"headlines file contains duplicate texts: {repeated}")

    return frame.reset_index(drop=True)


def textblob_label(text: str) -> str:
    """TextBlob's three way call, through the same threshold the rest uses."""

    return analyse_document(text).label


def lexicon_label(text: str) -> str:
    """The finance lexicon's three way call."""

    return lexicon.classify(text)


#: The analysers under comparison, keyed by the name used in every report.
ANALYSERS: dict[str, Callable[[str], str]] = {
    "textblob": textblob_label,
    "finance_lexicon": lexicon_label,
}


def predict(texts: Sequence[str], analyser: Callable[[str], str]) -> list[str]:
    """Run one analyser over every text, in order."""

    return [analyser(text) for text in texts]


def wilson_interval(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The normal approximation, p +/- z sqrt(p(1-p)/n), is the one taught first and
    it is wrong in exactly the place accuracy figures live: near 1, where it
    produces an upper bound above 1, and at small n, where its coverage is well
    below nominal. The Wilson interval solves the score equation instead and
    stays inside [0, 1] by construction.
    """

    if trials <= 0:
        raise ValueError("a confidence interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes ({successes}) must lie in [0, {trials}]")

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half_width = (z / denominator) * ((p * (1 - p) / trials + z**2 / (4 * trials**2)) ** 0.5)
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall and F1 for one class, with the counts behind them."""

    label: str
    support: int
    predicted: int
    true_positives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class AnalyserResult:
    """Everything measured about one analyser on one labelled set."""

    name: str
    texts: tuple[str, ...]
    truth: tuple[str, ...]
    predictions: tuple[str, ...]
    correct: tuple[bool, ...]
    accuracy: float
    accuracy_interval: tuple[float, float]
    confusion: pd.DataFrame
    per_class: tuple[ClassMetrics, ...]
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def n(self) -> int:
        return len(self.truth)

    @property
    def n_correct(self) -> int:
        return sum(self.correct)

    def errors(self) -> pd.DataFrame:
        """The misclassified items, with what was said and what was expected."""

        rows = [
            {"text": text, "label": true, "predicted": predicted}
            for text, true, predicted, ok in zip(
                self.texts, self.truth, self.predictions, self.correct, strict=True
            )
            if not ok
        ]
        return pd.DataFrame(rows, columns=["text", "label", "predicted"])


def confusion_matrix(
    truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str] = LABELS
) -> pd.DataFrame:
    """Counts of true label (rows) against predicted label (columns).

    Every label in `labels` gets a row and a column even when it never occurs, so
    that the matrix is square and comparable between analysers.
    """

    if len(truth) != len(predictions):
        raise ValueError(f"{len(truth)} true labels against {len(predictions)} predictions")

    matrix = pd.DataFrame(0, index=list(labels), columns=list(labels), dtype=int)
    matrix.index.name = "true"
    matrix.columns.name = "predicted"
    for true, predicted in zip(truth, predictions, strict=True):
        if true not in matrix.index:
            raise ValueError(f"unknown true label {true!r}")
        if predicted not in matrix.columns:
            raise ValueError(f"unknown predicted label {predicted!r}")
        matrix.loc[true, predicted] += 1
    return matrix


def per_class_metrics(matrix: pd.DataFrame) -> tuple[ClassMetrics, ...]:
    """Precision, recall and F1 per class, read straight off the matrix.

    A class the analyser never predicts has an undefined precision, zero over
    zero. It is reported as 0.0 rather than as NaN, and `predicted` is carried
    alongside so that the difference between "never guessed it" and "guessed it
    and was always wrong" stays visible.
    """

    results = []
    for label in matrix.index:
        true_positives = int(matrix.loc[label, label])
        support = int(matrix.loc[label].sum())
        predicted = int(matrix[label].sum())
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append(
            ClassMetrics(
                label=str(label),
                support=support,
                predicted=predicted,
                true_positives=true_positives,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
            )
        )
    return tuple(results)


def evaluate(
    name: str,
    texts: Sequence[str],
    truth: Sequence[str],
    analyser: Callable[[str], str],
    confidence: float = DEFAULT_CONFIDENCE,
) -> AnalyserResult:
    """Score one analyser over a labelled set and compute every metric."""

    predictions = predict(texts, analyser)
    correct = [p == t for p, t in zip(predictions, truth, strict=True)]
    matrix = confusion_matrix(truth, predictions)
    accuracy = sum(correct) / len(correct) if correct else 0.0
    return AnalyserResult(
        name=name,
        texts=tuple(texts),
        truth=tuple(truth),
        predictions=tuple(predictions),
        correct=tuple(correct),
        accuracy=float(accuracy),
        accuracy_interval=wilson_interval(sum(correct), len(correct), confidence),
        confusion=matrix,
        per_class=per_class_metrics(matrix),
        confidence=confidence,
    )


@dataclass(frozen=True)
class McNemarResult:
    """McNemar's test on two classifiers scored over the same items."""

    n_both_correct: int
    n_first_only: int
    n_second_only: int
    n_both_wrong: int
    statistic: float
    p_value: float
    method: str
    significant: bool
    alpha: float = 0.05

    @property
    def discordant(self) -> int:
        """The pairs the test actually uses."""

        return self.n_first_only + self.n_second_only


def mcnemar(
    first_correct: Sequence[bool],
    second_correct: Sequence[bool],
    alpha: float = 0.05,
    exact_threshold: int = EXACT_TEST_THRESHOLD,
) -> McNemarResult:
    """Test whether two paired classifiers differ, on the discordant pairs only.

    `first_correct[i]` and `second_correct[i]` must refer to the same item. The
    two counts that matter are b, the items the first got right and the second
    got wrong, and c, the reverse. Under the null hypothesis that the two
    analysers are equally likely to be the one that is right when they disagree,
    b is binomial(b + c, 0.5).

    With b + c below `exact_threshold` the p-value comes from that binomial
    directly. Above it the chi-square statistic (|b - c| - 1)^2 / (b + c) is used
    with one degree of freedom; the -1 is Edwards' continuity correction, which
    is needed because a discrete count is being compared against a continuous
    distribution. With no discordant pairs at all the two analysers agreed
    everywhere and there is nothing to test, which is reported as p = 1.0 rather
    than as a division by zero.
    """

    if len(first_correct) != len(second_correct):
        raise ValueError(
            f"paired test needs equal lengths, got {len(first_correct)} and {len(second_correct)}"
        )

    both = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if a and b)
    first_only = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if a and not b)
    second_only = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if b and not a)
    neither = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if not a and not b)

    discordant = first_only + second_only
    if discordant == 0:
        return McNemarResult(
            n_both_correct=both,
            n_first_only=0,
            n_second_only=0,
            n_both_wrong=neither,
            statistic=0.0,
            p_value=1.0,
            method="no discordant pairs",
            significant=False,
            alpha=alpha,
        )

    if discordant < exact_threshold:
        p_value = float(stats.binomtest(first_only, discordant, 0.5).pvalue)
        statistic = float(min(first_only, second_only))
        method = f"exact binomial on {discordant} discordant pairs"
    else:
        statistic = float((abs(first_only - second_only) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, 1))
        method = f"chi-square with continuity correction on {discordant} discordant pairs"

    return McNemarResult(
        n_both_correct=both,
        n_first_only=first_only,
        n_second_only=second_only,
        n_both_wrong=neither,
        statistic=statistic,
        p_value=p_value,
        method=method,
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


@dataclass(frozen=True)
class Comparison:
    """Two analysers, one labelled set, and the paired test between them."""

    dataset_size: int
    class_counts: dict[str, int]
    results: dict[str, AnalyserResult]
    test: McNemarResult
    first: str
    second: str

    @property
    def better(self) -> str | None:
        """The more accurate analyser, or None when they tie."""

        a, b = self.results[self.first].accuracy, self.results[self.second].accuracy
        if a == b:
            return None
        return self.first if a > b else self.second


def compare(
    frame: pd.DataFrame | None = None,
    analysers: dict[str, Callable[[str], str]] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = 0.05,
) -> Comparison:
    """Evaluate every analyser on the labelled set and test the difference.

    Exactly two analysers are compared, because McNemar's test is a paired test
    between two classifiers. Comparing more would need a correction for the
    multiple pairwise tests, and there is no third analyser here to need it.
    """

    frame = load_headlines() if frame is None else frame
    analysers = dict(ANALYSERS) if analysers is None else analysers
    if len(analysers) != 2:
        raise ValueError(f"McNemar compares exactly two analysers, got {len(analysers)}")

    texts = frame["text"].tolist()
    truth = frame["label"].tolist()
    results = {
        name: evaluate(name, texts, truth, analyser, confidence)
        for name, analyser in analysers.items()
    }
    first, second = list(results)
    test = mcnemar(results[first].correct, results[second].correct, alpha=alpha)

    return Comparison(
        dataset_size=len(frame),
        class_counts={label: int((frame["label"] == label).sum()) for label in LABELS},
        results=results,
        test=test,
        first=first,
        second=second,
    )


def confusion_figure(result: AnalyserResult, title: str | None = None):
    """Draw one confusion matrix, counts annotated in the cells.

    Colour carries no information a reader should rely on here, so the count is
    written into every cell. The diagonal is what the analyser got right.
    """

    matrix = result.confusion
    fig, ax = new_figure(width=5.5, height=5.0)
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    ax.set_title(title or f"{result.name}: {result.accuracy:.1%} accurate on n = {result.n}")

    largest = matrix.to_numpy().max()
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            count = int(matrix.iat[row, column])
            ax.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > largest / 2 else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="headlines")
    return fig
