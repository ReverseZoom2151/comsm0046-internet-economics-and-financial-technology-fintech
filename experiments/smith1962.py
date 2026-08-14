"""Run the Smith 1962 replication end to end and print what it measured.

    python -m experiments.smith1962

Every number printed here comes from a seeded run, so the same command gives the
same table twice. Figures go to figures/; nothing is written to the working
directory, because each market session runs and cleans up inside a temporary
directory of its own.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from fintech.smith import (
    DEFAULT_SEED,
    ScenarioResult,
    arrival_mode_comparison,
    chart1_baseline,
    plot_alpha,
    plot_transactions,
    shock_scenario,
    summary_table,
    trader_count_comparison,
    trader_mix_comparison,
)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def _print_scenario(result: ScenarioResult, full: bool = False) -> None:
    """Print one scenario, in full or as a single line of alpha per period.

    The comparisons deliberately overlap: the periodic arm of the arrival
    comparison is the baseline, and the 40-a-side arm of the count comparison is
    the homogeneous arm of the mix comparison. Printing every period table for
    all eight would be the same numbers three times over, so only the two
    scenarios with something to show get the full table.
    """

    print(f"\n{result.name}  ({result.description})")
    print(
        f"  equilibrium {result.equilibrium:.1f}, {result.n_transactions} trades "
        f"over {result.n_sessions} sessions, {result.n_silent_periods} silent "
        f"{'period' if result.n_silent_periods == 1 else 'periods'}"
    )
    if full:
        frame = result.to_frame()
        frame = frame.assign(period=frame["period"] + 1)
        with pd.option_context("display.width", 120):
            print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    else:
        profile = " ".join(
            "  . " if math.isnan(period.alpha) else f"{period.alpha:5.1f}"
            for period in result.periods
        )
        print(f"  alpha by period: {profile}")
    print(
        f"  alpha: first traded period {result.alpha_first:.2f}, final traded period "
        f"{result.alpha_last:.2f} on {result.final_trades} trades, "
        f"ratio {result.convergence_ratio:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sessions", type=int, default=10, help="independent runs per scenario")
    parser.add_argument("--seconds", type=float, default=600.0, help="simulated session length")
    parser.add_argument("--periods", type=int, default=10, help="trading periods per session")
    parser.add_argument("--figures", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)

    common = {
        "seed": args.seed,
        "n_sessions": args.sessions,
        "end_time": args.seconds,
        "n_periods": args.periods,
    }

    print("Vernon Smith (1962) Chart 1, replicated on the Bristol Stock Exchange")
    print(
        f"seed {args.seed}, {args.sessions} sessions per scenario, "
        f"{args.seconds:.0f} simulated seconds, {args.periods} trading periods"
    )

    baseline = chart1_baseline(**common)
    _print_scenario(baseline, full=True)

    arrivals = arrival_mode_comparison(**common)
    for result in arrivals:
        _print_scenario(result)

    counts = trader_count_comparison(**common)
    for result in counts:
        _print_scenario(result)

    mixes = trader_mix_comparison(**common)
    for result in mixes:
        _print_scenario(result)

    shock = shock_scenario(**common)
    _print_scenario(shock, full=True)

    every = [baseline, *arrivals, *counts, *mixes, shock]
    print("\nSummary")
    with pd.option_context("display.width", 160):
        print(
            summary_table(every).to_string(index=False, float_format=lambda value: f"{value:.2f}")
        )

    written = [plot_transactions(result, args.figures) for result in (baseline, shock)]
    written.append(plot_alpha([baseline, *arrivals], args.figures, "arrival mode"))
    written.append(plot_alpha(counts, args.figures, "trader count"))
    written.append(plot_alpha(mixes, args.figures, "trader mix"))

    print("\nFigures written:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
