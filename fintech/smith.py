"""Smith's 1962 Chart 1 replication, as named experiments that report numbers.

The demo notebook plots six scatter charts and invites the reader to agree that
prices converge. That is not a measurement, and with an unseeded random module
behind it the picture is different on every run. Smith supplied the measurement
himself: the coefficient of convergence alpha, the root mean square deviation of
transaction prices from the theoretical equilibrium, expressed as a percentage
of that equilibrium. His finding is that alpha falls from one trading period to
the next, so alpha per period is what these experiments compute, and whether it
falls is a question with an answer rather than an assumption.

Everything here is parameterised and takes a seed, so a reported number can be
reproduced. Figures are a by-product, not the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fintech.market import PooledTape, SessionResult, pool_tapes, run_sessions
from fintech.plotting import new_figure, save_figure
from fintech.schedules import (
    CHART1_RANGE,
    OrderSchedule,
    TraderPopulation,
    all_zip,
    equilibrium_at,
    fixed_schedule,
    mixed_population,
    shock_schedule,
)

DEFAULT_SEED = 100
DEFAULT_SESSION_SECONDS = 600.0
DEFAULT_PERIODS = 10


@dataclass(frozen=True)
class PeriodStats:
    """One trading period's worth of transactions, scored against equilibrium."""

    index: int
    start: float
    end: float
    equilibrium: float
    n_transactions: int
    mean_price: float
    alpha: float

    @property
    def traded(self) -> bool:
        return self.n_transactions > 0


@dataclass(frozen=True, eq=False)
class ScenarioResult:
    """A named, repeatable market scenario and what its prices did."""

    name: str
    description: str
    seed: int
    n_sessions: int
    equilibrium: float
    periods: tuple[PeriodStats, ...]
    sessions: tuple[SessionResult, ...]
    pooled: PooledTape

    @property
    def n_transactions(self) -> int:
        return self.pooled.n_transactions

    @property
    def traded_periods(self) -> tuple[PeriodStats, ...]:
        return tuple(period for period in self.periods if period.traded)

    @property
    def alpha_first(self) -> float:
        traded = self.traded_periods
        return traded[0].alpha if traded else math.nan

    @property
    def alpha_last(self) -> float:
        traded = self.traded_periods
        return traded[-1].alpha if traded else math.nan

    @property
    def alpha_overall(self) -> float:
        """Alpha over the whole session, each trade scored against its own equilibrium.

        Pooling the raw prices and comparing them to a single equilibrium would
        report the shocked scenario as wildly divergent when what actually
        happened is that the target moved and prices followed it. Each period
        contributes its own relative squared deviation, weighted by how much
        trading it saw.
        """

        traded = self.traded_periods
        if not traded:
            return math.nan
        total = sum(period.n_transactions for period in traded)
        weighted = sum(period.n_transactions * period.alpha**2 for period in traded)
        return math.sqrt(weighted / total)

    @property
    def n_silent_periods(self) -> int:
        """Periods in which the pooled market did no trade at all.

        A homogeneous ZIP market can stall, every trader waiting for someone else
        to move the book. Counting the silence turns that from a caveat in the
        notebook's prose into a number in the table.
        """

        return sum(1 for period in self.periods if not period.traded)

    @property
    def final_trades(self) -> int:
        traded = self.traded_periods
        return traded[-1].n_transactions if traded else 0

    @property
    def convergence_ratio(self) -> float:
        """alpha in the final period as a fraction of the first. Below one converges."""

        first, last = self.alpha_first, self.alpha_last
        if not first or math.isnan(first):
            return math.nan
        return last / first

    @property
    def final_mean_price(self) -> float:
        traded = self.traded_periods
        return traded[-1].mean_price if traded else math.nan

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "period": [p.index for p in self.periods],
                "start": [p.start for p in self.periods],
                "end": [p.end for p in self.periods],
                "equilibrium": [p.equilibrium for p in self.periods],
                "n_transactions": [p.n_transactions for p in self.periods],
                "mean_price": [p.mean_price for p in self.periods],
                "alpha": [p.alpha for p in self.periods],
            }
        )

    def summary_row(self) -> dict[str, object]:
        return {
            "scenario": self.name,
            "sessions": self.n_sessions,
            "trades": self.n_transactions,
            "equilibrium": self.equilibrium,
            "alpha_first": self.alpha_first,
            "alpha_last": self.alpha_last,
            "alpha_overall": self.alpha_overall,
            "ratio": self.convergence_ratio,
            "final_trades": self.final_trades,
            "final_mean_price": self.final_mean_price,
            "silent_periods": self.n_silent_periods,
        }


def smiths_alpha(prices: np.ndarray | Sequence[float], equilibrium: float) -> float:
    """Smith's coefficient of convergence: RMS deviation from equilibrium, as a percentage.

    Zero means every trade happened at the equilibrium price. Smith reports
    values falling from roughly 10 to 2 across the periods of an experiment.
    """

    if equilibrium <= 0:
        raise ValueError(f"equilibrium price must be positive, got {equilibrium}")
    values = np.asarray(prices, dtype=float)
    if values.size == 0:
        return math.nan
    rms = math.sqrt(float(np.mean((values - equilibrium) ** 2)))
    return 100.0 * rms / equilibrium


def period_bounds(start_time: float, end_time: float, n_periods: int) -> list[tuple[float, float]]:
    """Split a session into equal trading periods, Smith's "days"."""

    if n_periods < 1:
        raise ValueError(f"n_periods must be at least one, got {n_periods}")
    if end_time <= start_time:
        raise ValueError(f"session must have positive duration, got {start_time}-{end_time}")
    width = (end_time - start_time) / n_periods
    return [(start_time + i * width, start_time + (i + 1) * width) for i in range(n_periods)]


def period_stats(
    pooled: PooledTape,
    schedule: OrderSchedule,
    population: TraderPopulation,
    start_time: float,
    end_time: float,
    n_periods: int,
) -> tuple[PeriodStats, ...]:
    """Score each trading period against the equilibrium in force during it.

    Taking the equilibrium from the schedule at the period's midpoint is what
    makes the shocked scenario measurable: after the shock the target moves, and
    alpha is quoted against the new target rather than the stale one.
    """

    stats: list[PeriodStats] = []
    for index, (lower, upper) in enumerate(period_bounds(start_time, end_time, n_periods)):
        equilibrium = equilibrium_at(schedule, population, (lower + upper) / 2.0)
        prices = pooled.between(lower, upper)
        stats.append(
            PeriodStats(
                index=index,
                start=lower,
                end=upper,
                equilibrium=equilibrium,
                n_transactions=int(prices.size),
                mean_price=float(np.mean(prices)) if prices.size else math.nan,
                alpha=smiths_alpha(prices, equilibrium),
            )
        )
    return tuple(stats)


def run_scenario(
    name: str,
    schedule: OrderSchedule,
    population: TraderPopulation,
    description: str = "",
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    start_time: float = 0.0,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> ScenarioResult:
    """Run n sessions of one market configuration and measure their convergence."""

    sessions = run_sessions(
        schedule,
        population,
        seed=seed,
        n_sessions=n_sessions,
        start_time=start_time,
        end_time=end_time,
        session_id=_session_prefix(name),
    )
    pooled = pool_tapes(sessions)
    periods = period_stats(pooled, schedule, population, start_time, end_time, n_periods)
    return ScenarioResult(
        name=name,
        description=description,
        seed=seed,
        n_sessions=n_sessions,
        equilibrium=equilibrium_at(schedule, population, start_time),
        periods=periods,
        sessions=tuple(sessions),
        pooled=pooled,
    )


def _session_prefix(name: str) -> str:
    keep = [char if char.isalnum() else "_" for char in name.lower()]
    return "".join(keep).strip("_") or "session"


def chart1_baseline(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    n_per_side: int = 11,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> ScenarioResult:
    """Smith's Chart 1 as the notebook sets it up: 11 ZIPs a side, periodic orders."""

    schedule = fixed_schedule(0.0, end_time, interval=60.0, timemode="periodic")
    return run_scenario(
        "chart1 baseline",
        schedule,
        all_zip(n_per_side),
        description=f"{n_per_side} ZIP a side, periodic 60s, range "
        f"{CHART1_RANGE.low}-{CHART1_RANGE.high}",
        seed=seed,
        n_sessions=n_sessions,
        end_time=end_time,
        n_periods=n_periods,
    )


def arrival_mode_comparison(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    n_per_side: int = 11,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> list[ScenarioResult]:
    """Periodic re-supply against a Poisson drip, holding the population fixed."""

    population = all_zip(n_per_side)
    configurations = (
        ("periodic 60s", 60.0, "periodic"),
        ("drip-poisson 10s", 10.0, "drip-poisson"),
    )
    return [
        run_scenario(
            f"arrival {label}",
            fixed_schedule(0.0, end_time, interval=interval, timemode=timemode),
            population,
            description=f"{n_per_side} ZIP a side, {label}",
            seed=seed,
            n_sessions=n_sessions,
            end_time=end_time,
            n_periods=n_periods,
        )
        for label, interval, timemode in configurations
    ]


def trader_count_comparison(
    counts: Sequence[int] = (11, 40),
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> list[ScenarioResult]:
    """Does a thicker market converge harder? Same schedule, more ZIPs."""

    schedule = fixed_schedule(0.0, end_time, interval=10.0, timemode="drip-poisson")
    return [
        run_scenario(
            f"traders {count} a side",
            schedule,
            all_zip(count),
            description=f"{count} ZIP a side, drip-poisson 10s",
            seed=seed,
            n_sessions=n_sessions,
            end_time=end_time,
            n_periods=n_periods,
        )
        for count in counts
    ]


def trader_mix_comparison(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    n_each: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> list[ScenarioResult]:
    """Homogeneous ZIP against the ZIP/ZIC/SHVR/GVWY mix, at equal head count."""

    schedule = fixed_schedule(0.0, end_time, interval=10.0, timemode="drip-poisson")
    populations = (all_zip(4 * n_each), mixed_population(n_each))
    return [
        run_scenario(
            f"mix {population.label}",
            schedule,
            population,
            description=f"{population.size} traders, drip-poisson 10s",
            seed=seed,
            n_sessions=n_sessions,
            end_time=end_time,
            n_periods=n_periods,
        )
        for population in populations
    ]


def shock_scenario(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 10,
    n_each: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
) -> ScenarioResult:
    """The mixed population under a mid-session jump to a new price range."""

    schedule = shock_schedule(0.0, end_time, interval=10.0, timemode="drip-poisson")
    return run_scenario(
        "market shock",
        schedule,
        mixed_population(n_each),
        description="mixed population, equilibrium jumps half way through",
        seed=seed,
        n_sessions=n_sessions,
        end_time=end_time,
        n_periods=n_periods,
    )


def summary_table(results: Sequence[ScenarioResult]) -> pd.DataFrame:
    """One row per scenario, in the order they were run."""

    return pd.DataFrame([result.summary_row() for result in results])


def plot_transactions(result: ScenarioResult, directory: str | Path) -> Path:
    """Pooled transaction prices against time, with the equilibrium drawn on."""

    fig, ax = new_figure()
    ax.plot(
        result.pooled.times,
        result.pooled.prices,
        "x",
        color="black",
        markersize=4,
        alpha=0.6,
        label="transactions",
    )
    steps_time = [period.start for period in result.periods] + [result.periods[-1].end]
    steps_eq = [period.equilibrium for period in result.periods] + [result.periods[-1].equilibrium]
    ax.step(
        steps_time,
        steps_eq,
        where="post",
        color="crimson",
        linewidth=1.5,
        label="theoretical equilibrium",
    )
    ax.set_xlabel("time (simulated seconds)")
    ax.set_ylabel("transaction price")
    ax.set_title(f"{result.name}: {result.n_sessions} sessions, {result.n_transactions} trades")
    ax.legend(frameon=False, loc="best")
    path = save_figure(fig, f"smith1962_{result.name}", directory)
    plt.close(fig)
    return path


def plot_alpha(
    results: Sequence[ScenarioResult], directory: str | Path, title: str = "convergence"
) -> Path:
    """Smith's alpha per trading period, one line per scenario."""

    if not results:
        raise ValueError("nothing to plot")
    fig, ax = new_figure()
    for result in results:
        traded = result.traded_periods
        ax.plot(
            [period.index + 1 for period in traded],
            [period.alpha for period in traded],
            marker="o",
            label=result.name,
        )
    ax.set_xlabel("trading period")
    ax.set_ylabel("Smith's alpha (% of equilibrium)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    path = save_figure(fig, f"smith1962_alpha_{title}", directory)
    plt.close(fig)
    return path
