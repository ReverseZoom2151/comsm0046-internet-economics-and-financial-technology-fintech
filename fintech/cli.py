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

    sub.add_parser("datasets", help="list the available course datasets")

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
