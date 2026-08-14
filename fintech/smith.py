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

Reporting one alpha per scenario was still not enough. A scenario's alpha is
pooled over every session it ran, which collapses the sample down to a single
number and leaves "4.50 is better than 18.12" as an eyeball comparison of two
point estimates with no spread attached. The sessions are independent runs, so
the sample was there all along: `session_alphas` scores each session on its own,
giving n numbers per scenario, and `convergence_claims` puts the claims this
repository makes about those numbers through the same assumption-aware pipeline
the week 5 strand uses on somebody else's data, with the same Holm correction
across the family.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fintech.hypothesis_tests import DEFAULT_ALPHA, AnalysisResult, analyse, holm_correction
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
    def session_alphas(self) -> np.ndarray:
        """One alpha per session: the sample the pooled figures were hiding.

        Each session is scored the way `alpha_overall` scores the pooled tape,
        period by period against the equilibrium in force, so the two are the
        same measurement at different granularity. A session that never traded
        contributes NaN rather than zero, because "no trades" is not "perfectly
        converged", and the testing pipeline drops NaN rather than believing it.
        """

        return session_alphas(self)

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
            "alpha_session_mean": float(np.nanmean(self.session_alphas))
            if self.sessions
            else math.nan,
            "alpha_session_sd": float(np.nanstd(self.session_alphas, ddof=1))
            if len(self.sessions) > 1
            else math.nan,
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


@dataclass(frozen=True, eq=False)
class AlphaSample:
    """A named sample of per-session alphas, which is what claims are made about."""

    label: str
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.values)))

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.values)) if self.n else math.nan


@dataclass(frozen=True, eq=False)
class ClaimResult:
    """One stated claim about convergence, and whether the sessions support it.

    `supported` requires two things and reports them separately, because they
    fail in different ways: the difference has to point the way the claim says it
    does, and it has to survive the test. A claim can be true in direction and
    still be unsupported, which is the honest thing to say about most of these.
    """

    claim: str
    better: str
    worse: str
    n_better: int
    n_worse: int
    mean_better: float
    mean_worse: float
    test: str
    p_value: float
    p_value_corrected: float
    significant: bool
    analysis: AnalysisResult

    @property
    def direction_holds(self) -> bool:
        """Is the scenario the claim calls better actually the one with lower alpha?"""

        return self.mean_better < self.mean_worse

    @property
    def supported(self) -> bool:
        return self.direction_holds and self.significant

    @property
    def verdict(self) -> str:
        if self.supported:
            return "supported"
        if not self.direction_holds:
            return "contradicted: the difference runs the other way"
        return "not supported: the difference is not distinguishable from noise"


@dataclass(frozen=True, eq=False)
class ClaimFamily:
    """Every convergence claim tested together, with one correction across the set."""

    claims: tuple[ClaimResult, ...]
    alpha: float = DEFAULT_ALPHA

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "claim": claim.claim,
                    "better": claim.better,
                    "worse": claim.worse,
                    "mean_alpha_better": claim.mean_better,
                    "mean_alpha_worse": claim.mean_worse,
                    "n": min(claim.n_better, claim.n_worse),
                    "test": claim.test,
                    "p_value": claim.p_value,
                    "p_holm": claim.p_value_corrected,
                    "verdict": claim.verdict,
                }
                for claim in self.claims
            ]
        )

    @property
    def supported(self) -> tuple[ClaimResult, ...]:
        return tuple(claim for claim in self.claims if claim.supported)

    @property
    def unsupported(self) -> tuple[ClaimResult, ...]:
        return tuple(claim for claim in self.claims if not claim.supported)


def session_alphas(result: ScenarioResult, periods: Sequence[int] | None = None) -> np.ndarray:
    """Alpha per session, optionally restricted to a subset of trading periods.

    Restricting the periods is what makes the shock claim testable. FINDINGS.md
    says the shocked market ends up tighter than the market ever tracked the
    original equilibrium, and that is a statement about the periods after the
    shock. Scoring the whole session would mix them with the pre-shock half,
    which is the unshocked scenario, and dilute the comparison into meaninglessness.
    """

    wanted = result.periods if periods is None else tuple(result.periods[i] for i in periods)
    values: list[float] = []
    for session in result.sessions:
        total = 0
        weighted = 0.0
        for period in wanted:
            prices = session.between(period.start, period.end)
            if prices.size == 0:
                continue
            total += int(prices.size)
            weighted += prices.size * smiths_alpha(prices, period.equilibrium) ** 2
        values.append(math.sqrt(weighted / total) if total else math.nan)
    return np.asarray(values, dtype=float)


def alpha_sample(
    result: ScenarioResult, periods: Sequence[int] | None = None, label: str = ""
) -> AlphaSample:
    """A named per-session alpha sample drawn from one scenario."""

    return AlphaSample(label=label or result.name, values=session_alphas(result, periods))


def alpha_frame(samples: Sequence[AlphaSample]) -> pd.DataFrame:
    """One column per sample, ready for the testing pipeline.

    Samples need not be the same length, so the columns are padded with NaN,
    which `fintech.hypothesis_tests` drops column by column.
    """

    if not samples:
        raise ValueError("nothing to compare")
    return pd.DataFrame({sample.label: pd.Series(sample.values) for sample in samples})


def compare_scenarios(
    results: Sequence[ScenarioResult], name: str = "convergence", alpha: float = DEFAULT_ALPHA
) -> AnalysisResult:
    """Put whole-session alpha for several scenarios through the repo's own pipeline."""

    frame = alpha_frame([alpha_sample(result) for result in results])
    return analyse(frame, name=name, alpha=alpha)


def judge_claim(
    claim: str,
    better: AlphaSample,
    worse: AlphaSample,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[str, AlphaSample, AlphaSample, AnalysisResult]:
    """Package one stated claim with the two-sample analysis that will judge it."""

    analysis = analyse(alpha_frame([better, worse]), name=claim, alpha=alpha)
    return (claim, better, worse, analysis)


def build_claim_family(
    judged: Sequence[tuple[str, AlphaSample, AlphaSample, AnalysisResult]],
    alpha: float = DEFAULT_ALPHA,
) -> ClaimFamily:
    """Holm correct across every claim, then decide each one.

    Testing four claims at 0.05 each gives roughly a one in five chance of a
    spurious "significant" somewhere. The week 5 module already refuses to make
    that mistake with its normality tests; there is no reason for this strand to
    make it with its own.
    """

    corrected = holm_correction([analysis.omnibus.p_value for _, _, _, analysis in judged])
    claims = tuple(
        ClaimResult(
            claim=text,
            better=better.label,
            worse=worse.label,
            n_better=better.n,
            n_worse=worse.n,
            mean_better=better.mean,
            mean_worse=worse.mean,
            test=analysis.omnibus.test,
            p_value=analysis.omnibus.p_value,
            p_value_corrected=float(adjusted),
            significant=bool(adjusted < alpha),
            analysis=analysis,
        )
        for (text, better, worse, analysis), adjusted in zip(judged, corrected, strict=True)
    )
    return ClaimFamily(claims=claims, alpha=alpha)


def convergence_claims(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 20,
    end_time: float = DEFAULT_SESSION_SECONDS,
    n_periods: int = DEFAULT_PERIODS,
    n_each: int = 10,
    alpha: float = DEFAULT_ALPHA,
) -> ClaimFamily:
    """Test the convergence claims this repository makes, as one corrected family.

    The claims are taken from FINDINGS.md rather than invented here, so that the
    document and the test cannot drift apart: if a claim fails, the document is
    what has to change.
    """

    common = {"seed": seed, "n_sessions": n_sessions, "end_time": end_time, "n_periods": n_periods}
    periodic, drip = arrival_mode_comparison(**common)
    homogeneous, mixed = trader_mix_comparison(n_each=n_each, **common)
    thin, thick = trader_count_comparison(**common)
    shock = shock_scenario(n_each=n_each, **common)

    # The shock lands half way through, so the periods after it are the back half.
    late = tuple(range(n_periods // 2, n_periods))
    judged = [
        judge_claim(
            "drip-poisson converges better than periodic",
            alpha_sample(drip, label="drip-poisson"),
            alpha_sample(periodic, label="periodic"),
            alpha=alpha,
        ),
        judge_claim(
            "the mixed population converges better than all-ZIP",
            alpha_sample(mixed, label="mixed"),
            alpha_sample(homogeneous, label="all-ZIP"),
            alpha=alpha,
        ),
        judge_claim(
            "a thicker all-ZIP market converges better",
            alpha_sample(thick, label="40 ZIP a side"),
            alpha_sample(thin, label="11 ZIP a side"),
            alpha=alpha,
        ),
        judge_claim(
            "after the shock the market tracks the new equilibrium more tightly "
            "than the unshocked market tracked the old one",
            alpha_sample(shock, late, label="shocked, post-shock periods"),
            alpha_sample(mixed, late, label="unshocked mixed, same periods"),
            alpha=alpha,
        ),
    ]
    return build_claim_family(judged, alpha=alpha)


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
