"""Run both week 5 datasets through the assumption-aware pipeline.

This module replaces `week5_activity_dataset1.ipynb` and
`week5_activity_dataset2.ipynb` outright. Those two notebooks are 24 of their
27 cells byte-identical: the only difference between 272KB and 207KB of stored
JSON is which `pd.read_csv` line is commented out. Duplicating a whole document
to change one argument means every fix has to be made twice, and in practice
one copy drifts.

Selecting the dataset is a loop here, so both are analysed in one run, and the
chosen omnibus test is printed together with the reason the assumptions
license it. Figures go to figures/ rather than to `plt.show()` calls buried in
loops, so a headless run produces artefacts instead of nothing.

Run it with:

    python -m experiments.week5_hypothesis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fintech.datasets import available_datasets, dataset_spec, load_dataset  # noqa: E402
from fintech.hypothesis_tests import (  # noqa: E402
    DEFAULT_ALPHA,
    AnalysisResult,
    all_figures,
    analyse,
    save_figures,
)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def report(result: AnalysisResult) -> str:
    """The analysis written out as the text the notebooks never quite produced."""

    lines = [
        "=" * 78,
        f"Dataset {result.name}: {len(result.conditions)} conditions "
        f"({', '.join(result.conditions)}), "
        f"n = {', '.join(str(result.n_per_condition[c]) for c in result.conditions)}",
        "=" * 78,
        "",
        "Summary statistics",
        "-" * 78,
        result.summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "",
        "Assumption 1, normality (Shapiro-Wilk, Holm corrected)",
        "-" * 78,
    ]
    for normality in result.normality:
        verdict = "normal" if normality.normal else "NOT normal"
        lines.append(
            f"  {normality.condition}: W = {normality.statistic:.4f}, "
            f"p = {normality.p_value:.4f}, Holm p = {normality.p_value_corrected:.4f} "
            f"-> {verdict}"
        )

    lines += [
        "",
        "Assumption 2, equal variance (Levene, median centred)",
        "-" * 78,
        f"  W = {result.variance.statistic:.4f}, p = {result.variance.p_value:.4f} -> "
        f"{'equal variances' if result.variance.equal_variance else 'UNEQUAL variances'}",
        "",
        "Omnibus test chosen from the assumptions",
        "-" * 78,
        f"  test: {result.omnibus.test}",
        f"  why:  {result.omnibus.reason}",
        f"  statistic = {result.omnibus.statistic:.4f}, p = {result.omnibus.p_value:.5f}",
    ]
    if result.omnibus.df_between is not None:
        within = "n/a" if result.omnibus.df_within is None else f"{result.omnibus.df_within:.3f}"
        lines.append(f"  df between = {result.omnibus.df_between:.0f}, df within = {within}")
    if result.omnibus.equivalent_to:
        lines.append(f"  note: equivalent to {result.omnibus.equivalent_to}")
    verdict = "reject" if result.omnibus.significant else "fail to reject"
    lines.append(f"  conclusion: {verdict} the null of equal conditions at alpha = {result.alpha}")

    lines += ["", "Post-hoc", "-" * 78]
    if result.post_hoc.test is None:
        lines.append(f"  none run: {result.post_hoc.reason}")
    else:
        lines.append(f"  {result.post_hoc.test}: {result.post_hoc.reason}")
        for comparison in result.post_hoc.comparisons:
            mark = "differ" if comparison["reject"] else "no difference"
            lines.append(
                f"    {comparison['group1']} vs {comparison['group2']}: "
                f"mean difference = {comparison['mean_difference']:.4f}, "
                f"p = {comparison['p_value']:.4f}, "
                f"95% CI [{comparison['lower']:.4f}, {comparison['upper']:.4f}] -> {mark}"
            )

    lines += ["", "Caveats", "-" * 78]
    lines += [f"  - {note}" for note in result.notes]
    lines.append("")
    return "\n".join(lines)


def run(
    figure_dir: Path | str = FIGURE_DIR,
    alpha: float = DEFAULT_ALPHA,
    make_figures: bool = True,
) -> dict[str, AnalysisResult]:
    """Analyse every activity dataset and optionally write its figures."""

    results: dict[str, AnalysisResult] = {}
    for name in available_datasets():
        frame = load_dataset(name)
        result = analyse(frame, name=name, alpha=alpha)
        results[name] = result
        print(report(result))
        print(f"  {dataset_spec(name).description}\n")

        if make_figures:
            figures = all_figures(frame, name=name)
            paths = save_figures(figures, figure_dir)
            for figure in figures.values():
                plt.close(figure)
            print(f"  wrote {len(paths)} figures to {Path(figure_dir)}\n")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR), help="where to write the figures")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="significance threshold")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args(argv)

    run(figure_dir=args.figure_dir, alpha=args.alpha, make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
