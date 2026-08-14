"""Command line interface.

One entry point for the three strands of the coursework: the market simulation,
the hypothesis testing, and the sentiment analysis. Every command takes a seed
where randomness is involved, and nothing is written to disk unless a directory
is named, so a command is safe to run from a script.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

DEFAULT_SEED = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fintech",
        description=(
            "Market simulation, hypothesis testing and sentiment analysis for "
            "internet economics and financial technology."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- market ------------------------------------------------------------
    market = sub.add_parser("market", help="Vernon Smith 1962 market experiments")
    market.add_argument(
        "scenario",
        nargs="?",
        default="baseline",
        choices=["baseline", "arrival", "traders", "mix", "shock", "all"],
        help="which experiment to run",
    )
    market.add_argument("--seed", type=int, default=DEFAULT_SEED)
    market.add_argument("--sessions", type=int, default=10, help="market sessions per scenario")
    market.add_argument("--seconds", type=float, default=600.0, help="simulated seconds")
    market.add_argument("--periods", type=int, default=10, help="trading periods per session")
    market.add_argument("--save-figures", metavar="DIRECTORY", default=None)

    # -- statistics --------------------------------------------------------
    stats = sub.add_parser("stats", help="hypothesis testing over the course datasets")
    stats.add_argument("dataset", nargs="?", default="all", help="data1, data2, or all")
    stats.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    stats.add_argument("--save-figures", metavar="DIRECTORY", default=None)

    # -- sentiment ---------------------------------------------------------
    sentiment = sub.add_parser("sentiment", help="document, sentence and aspect level sentiment")
    sentiment.add_argument(
        "text",
        nargs="?",
        default="all",
        help="a corpus name (short_review, long_review, hack_crash_tweet, "
        "muddy_waters_tweet), 'all', or your own text",
    )
    sentiment.add_argument(
        "--download",
        action="store_true",
        help="download the NLTK tokenizer if it is missing",
    )

    # -- profitability -----------------------------------------------------
    profit = sub.add_parser("profit", help="which trading strategy makes the most money")
    profit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    profit.add_argument("--sessions", type=int, default=20, help="market sessions per arm")
    profit.add_argument("--each", type=int, default=10, help="traders per strategy per side")
    profit.add_argument("--seconds", type=float, default=600.0)
    profit.add_argument(
        "--head-to-head",
        action="store_true",
        help="also run every pair of strategies against each other",
    )

    # -- sentiment evaluation ----------------------------------------------
    evaluate = sub.add_parser(
        "evaluate", help="score the labelled headlines with both sentiment analysers"
    )
    evaluate.add_argument("--save-figures", metavar="DIRECTORY", default=None)

    sub.add_parser("datasets", help="list the available course datasets")

    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine readable JSON instead of a formatted report",
    )

    return parser


def _run_market(args) -> int:
    from . import smith

    common = dict(
        seed=args.seed,
        n_sessions=args.sessions,
        end_time=args.seconds,
        n_periods=args.periods,
    )
    scenarios = {
        "baseline": lambda: [smith.chart1_baseline(**common)],
        "arrival": lambda: smith.arrival_mode_comparison(**common),
        "traders": lambda: smith.trader_count_comparison(**common),
        "mix": lambda: smith.trader_mix_comparison(**common),
        "shock": lambda: [smith.shock_scenario(**common)],
    }
    if args.scenario == "all":
        results = [r for make in scenarios.values() for r in make()]
    else:
        results = scenarios[args.scenario]()

    print(smith.summary_table(results).to_string(index=False))

    if args.save_figures:
        for result in results:
            print(f"wrote {smith.plot_alpha(result, directory=args.save_figures)}")
    return 0


def _run_stats(args) -> int:
    from experiments.week5_hypothesis import report

    from . import datasets, hypothesis_tests

    names = datasets.available_datasets() if args.dataset == "all" else (args.dataset,)
    for name in names:
        frame = datasets.load_dataset(name)
        result = hypothesis_tests.analyse(frame, name=name, alpha=args.alpha)
        print(report(result))
        print()
        if args.save_figures:
            figures = hypothesis_tests.all_figures(frame, name)
            for path in hypothesis_tests.save_figures(figures, args.save_figures):
                print(f"wrote {path}")
    return 0


#: The significance tests need at least three observations per strategy, and a
#: sample of three is not worth simulating for, so the floor is set higher.
MIN_PROFIT_SESSIONS = 5


def _run_profit(args) -> int:
    from . import profitability

    if args.sessions < MIN_PROFIT_SESSIONS:
        print(
            f"profit needs at least {MIN_PROFIT_SESSIONS} sessions: each session gives one "
            f"observation per strategy, and the comparison cannot run on fewer than three.",
        )
        return 2

    study = profitability.profit_study(
        seed=args.seed,
        n_sessions=args.sessions,
        n_each=args.each,
        end_time=args.seconds,
    )

    if getattr(args, "json", False):
        print(_as_json(study))
        return 0

    from experiments.strategy_profit import print_profit

    print_profit(study)
    return 0


def _run_evaluate(args) -> int:
    from . import evaluation

    comparison = evaluation.compare()

    if getattr(args, "json", False):
        print(_as_json(comparison))
        return 0

    print(f"{comparison.dataset_size} labelled headlines: {comparison.class_counts}")
    print()
    for name, result in comparison.results.items():
        low, high = result.accuracy_interval
        print(f"  {name:16} accuracy {result.accuracy:6.1%}  95% CI [{low:.1%}, {high:.1%}]")
        for metric in result.per_class:
            print(
                f"      {metric.label:9} precision {metric.precision:.3f}  "
                f"recall {metric.recall:.3f}  n={metric.support}"
            )
    test = comparison.test
    print()
    print(
        f"  McNemar: {test.n_first_only} for {comparison.first} only, "
        f"{test.n_second_only} for {comparison.second} only, "
        f"p = {test.p_value:.3g}"
        f"{'' if test.significant else '  not significant'}"
    )

    if args.save_figures:
        from .plotting import save_figure

        for name, result in comparison.results.items():
            figure = evaluation.confusion_figure(result)
            path = save_figure(figure, f"Confusion matrix {name}", args.save_figures)
            print(f"wrote {path}")
    return 0


def _as_json(value) -> str:
    """Render a result as JSON, so a run can feed something downstream.

    Dataclasses become objects, tuples become arrays, and anything the encoder
    does not recognise falls back to its string form rather than raising, since
    the point is to get the numbers out rather than to round trip the objects.
    """

    import dataclasses
    import json

    def default(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return str(obj)

    return json.dumps(value, default=default, indent=2, sort_keys=True)


def _run_sentiment(args) -> int:
    from . import reviews, sentiment

    if args.download:
        sentiment.ensure_corpora(download=True)

    if args.text == "all":
        items = list(reviews.CORPUS.items())
    elif args.text in reviews.CORPUS:
        items = [(args.text, reviews.CORPUS[args.text])]
    else:
        items = [("input", args.text)]

    for name, text in items:
        print(sentiment.analyse_text(text, name=name, aspects=reviews.MAC_ASPECTS))
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "datasets":
        from . import datasets

        for name in datasets.available_datasets():
            spec = datasets.dataset_spec(name)
            print(f"  {name:8} {spec.n_conditions} conditions, n={spec.n_per_condition}")
            print(f"           {spec.description}")
        return 0

    if args.command == "market":
        return _run_market(args)
    if args.command == "stats":
        return _run_stats(args)
    if args.command == "sentiment":
        return _run_sentiment(args)
    if args.command == "profit":
        return _run_profit(args)
    if args.command == "evaluate":
        return _run_evaluate(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
