"""Who makes the money, and which convergence claims survive a test.

    python -m experiments.strategy_profit

Two questions the coursework does not ask, answered on the same simulator and
the same seeds as the rest of the market strand.

The first is profit. `fintech.profitability` runs the four trading algorithms
against each other and then in every two-strategy head to head, because a profit
figure from a mixed market is a figure for a strategy against those particular
opponents rather than a property of the strategy.

The second is whether the convergence differences reported in FINDINGS.md are
real. They were point estimates of pooled alpha with no spread attached.
`fintech.smith.convergence_claims` scores each session separately and puts the
claims through `fintech.hypothesis_tests`, the pipeline this repository already
owns and had only ever pointed at somebody else's data.

Everything is seeded, so the same command prints the same tables twice.
"""

from __future__ import annotations

import argparse

import pandas as pd

from fintech.profitability import DEFAULT_SEED, ProfitStudy, profit_study
from fintech.smith import ClaimFamily, convergence_claims


def _table(frame: pd.DataFrame) -> str:
    with pd.option_context("display.width", 160):
        return frame.to_string(index=False, float_format=lambda value: f"{value:.4g}")


def print_profit(study: ProfitStudy) -> None:
    mixed = study.mixed
    print("\n=== Profit by trading strategy ===")
    print(f"\nMixed market: {mixed.description}, {mixed.n_sessions} sessions, seed {mixed.seed}")
    print("mean profit per trader, one sample per session\n")
    print(_table(mixed.summary_frame()))

    omnibus = mixed.analysis.omnibus
    print(f"\n{omnibus.test}: statistic {omnibus.statistic:.4g}, p = {omnibus.p_value:.4g}")
    print(f"  chosen because {omnibus.reason}")
    if omnibus.significant:
        ranking = " > ".join(mixed.ranking)
        print(f"  the four strategies do not all earn the same; ranking {ranking}")
    else:
        print("  no detectable difference between the four, so the ranking above is not evidence")

    post_hoc = mixed.analysis.post_hoc
    if post_hoc.test:
        print(f"\n{post_hoc.test} on the mixed market:")
        print(post_hoc.summary)
    else:
        print(f"\nNo post-hoc: {post_hoc.reason}")

    print("\nHead to head, each pair alone in a market together, Holm corrected as a family\n")
    print(_table(study.head_to_head.to_frame()))
    wins = study.head_to_head.wins()
    print("\nsignificant head-to-head wins: " + ", ".join(f"{k} {v}" for k, v in wins.items()))
    if study.agrees:
        print(f"{mixed.winner} wins the mixed market and every head to head it fought.")
    else:
        print(
            f"{mixed.winner} tops the mixed market but does not win every head to head, "
            "which is the population-mix confound showing up rather than being argued away."
        )

    print("\nCaveats that travel with these numbers:")
    for note in study.notes:
        print(f"  - {note}")


def print_claims(family: ClaimFamily) -> None:
    print("\n=== Convergence claims, tested ===")
    print(
        f"per-session Smith alpha, {len(family.claims)} claims Holm corrected together "
        f"at alpha = {family.alpha}\n"
    )
    print(_table(family.to_frame()))

    for claim in family.claims:
        print(f"\n{claim.claim}")
        print(
            f"  {claim.better}: mean alpha {claim.mean_better:.3f} over n = {claim.n_better}; "
            f"{claim.worse}: {claim.mean_worse:.3f} over n = {claim.n_worse}"
        )
        print(
            f"  {claim.test}, p = {claim.p_value:.4g}, Holm p = {claim.p_value_corrected:.4g}"
            f" -> {claim.verdict}"
        )
        print(f"  test chosen because {claim.analysis.omnibus.reason}")

    supported = [claim.claim for claim in family.supported]
    failed = [claim.claim for claim in family.unsupported]
    print(f"\n{len(supported)} of {len(family.claims)} claims survive.")
    for text in failed:
        print(f"  fails: {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # The default is deliberately smaller than the settings FINDINGS.md
    # publishes. Reproducing that table needs --sessions 40 --each 20, which
    # takes over fifteen minutes because the head to head arm runs six pairs;
    # the exact command is in the "Reproducing all of this" section there.
    #
    # Note that the default is not merely a faster version of the same answer:
    # the profit comparison gives p = 0.057 at twenty sessions and p = 0.00103
    # at forty, so a default run does not reach the published conclusion. That
    # is the point being made about sample size rather than a caveat to it.
    parser.add_argument("--sessions", type=int, default=20, help="independent runs per condition")
    parser.add_argument("--seconds", type=float, default=600.0, help="simulated session length")
    parser.add_argument("--periods", type=int, default=10, help="trading periods per session")
    parser.add_argument("--each", type=int, default=10, help="traders of each type, per side")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    parser.add_argument("--profit-only", action="store_true")
    parser.add_argument("--claims-only", action="store_true")
    args = parser.parse_args(argv)

    print("Strategy profitability and convergence significance, on the Bristol Stock Exchange")
    print(
        f"seed {args.seed}, {args.sessions} sessions per condition, "
        f"{args.seconds:.0f} simulated seconds"
    )

    if not args.claims_only:
        print_profit(
            profit_study(
                seed=args.seed,
                n_sessions=args.sessions,
                n_each=args.each,
                end_time=args.seconds,
                alpha=args.alpha,
            )
        )

    if not args.profit_only:
        print_claims(
            convergence_claims(
                seed=args.seed,
                n_sessions=args.sessions,
                end_time=args.seconds,
                n_periods=args.periods,
                n_each=args.each,
                alpha=args.alpha,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
