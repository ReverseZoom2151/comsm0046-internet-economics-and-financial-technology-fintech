"""Measure TextBlob against a finance lexicon on labelled market headlines.

Run with `python -m experiments.sentiment_evaluation`.

`experiments/week7_sentiment.py` shows what TextBlob says about two tweets that
moved markets. It is an illustration, and two texts cannot support a claim about
an analyser. This script runs both analysers over every headline in
`data/headlines.csv`, reports accuracy with a confidence interval, prints the
confusion matrix and the per-class precision and recall, tests the difference
between the two with McNemar's paired test, and lists the items each one gets
wrong so that the failure modes can be read rather than assumed.

Nothing here asserts an expected number. It prints what the analysers say, and
the errors it lists are whatever they turn out to be.

No network access is needed: document level polarity does not require the NLTK
tokenizer corpora, and the finance lexicon is embedded in `fintech.lexicon`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fintech.evaluation import compare, confusion_figure
from fintech.plotting import save_figure

FIGURES = Path(__file__).resolve().parent.parent / "figures"

RULE = "=" * 78


def _heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def _report_analyser(result) -> None:
    low, high = result.accuracy_interval
    print(f"\n{result.name}")
    print(
        f"  accuracy {result.accuracy:.1%} "
        f"({result.n_correct}/{result.n}), "
        f"{result.confidence:.0%} Wilson interval [{low:.1%}, {high:.1%}]"
    )
    print("\n  confusion matrix (rows are the true label, columns the prediction):")
    for line in result.confusion.to_string().splitlines():
        print(f"    {line}")
    print(
        f"\n  {'class':<10}{'support':>8}{'predicted':>11}{'precision':>11}{'recall':>9}{'F1':>7}"
    )
    for metrics in result.per_class:
        print(
            f"  {metrics.label:<10}{metrics.support:>8}{metrics.predicted:>11}"
            f"{metrics.precision:>11.3f}{metrics.recall:>9.3f}{metrics.f1:>7.3f}"
        )


def _report_errors(result, limit: int) -> None:
    errors = result.errors()
    if errors.empty:
        print(f"\n  {result.name} misclassified nothing")
        return
    print(f"\n  {result.name} misclassified {len(errors)} of {result.n}, showing up to {limit}:")
    for _, row in errors.head(limit).iterrows():
        print(f"    said {row['predicted']:<9} for {row['label']:<9} {row['text'][:76]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--errors",
        type=int,
        default=12,
        help="how many misclassified headlines to list per analyser (default 12)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip writing the confusion matrix figures",
    )
    args = parser.parse_args(argv)

    comparison = compare()

    _heading("Labelled set")
    print(f"{comparison.dataset_size} author-written market headlines")
    for label, count in comparison.class_counts.items():
        print(f"  {label:<10}{count:>4}")
    print(
        "\nThese are written for this repository, not sampled from a newswire and not a\n"
        "published benchmark, so the accuracies below describe the analysers on this set\n"
        "rather than on real newsflow. The paired comparison between them is the part\n"
        "that carries across."
    )

    _heading("Per analyser")
    for result in comparison.results.values():
        _report_analyser(result)

    _heading("McNemar's paired test")
    test = comparison.test
    first, second = comparison.first, comparison.second
    print(f"comparing {first} against {second} on the same {comparison.dataset_size} items\n")
    print(f"  both correct        {test.n_both_correct:>4}")
    print(f"  only {first:<16}{test.n_first_only:>4}")
    print(f"  only {second:<16}{test.n_second_only:>4}")
    print(f"  both wrong          {test.n_both_wrong:>4}")
    print(f"\n  method     {test.method}")
    print(f"  statistic  {test.statistic:.4f}")
    print(f"  p-value    {test.p_value:.6g}")
    print(f"  significant at alpha = {test.alpha}: {'yes' if test.significant else 'no'}")
    if comparison.better is None:
        print("  the two analysers are equally accurate on this set")
    else:
        print(f"  more accurate on this set: {comparison.better}")

    _heading("What each one gets wrong")
    for result in comparison.results.values():
        _report_errors(result, args.errors)

    if not args.no_figures:
        _heading("Figures")
        for result in comparison.results.values():
            title = f"Confusion matrix {result.name} on labelled headlines"
            path = save_figure(confusion_figure(result), title, FIGURES)
            print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
