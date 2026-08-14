"""Which trading algorithm makes money, and whether the difference is real.

The coursework asks whether prices converge. It never asks who profits, which is
the question anybody running these agents for money would ask first. BSE records
the answer in its average-balance dump and the wrapper used to throw it away.

Two measurement decisions here are not cosmetic.

Mean profit per trader, not total profit. BSE hands every trader a limit price
off the supply or demand schedule, and a trader can never earn more than the gap
between its assignment and the price it transacts at. A type with more traders in
the population collects more total profit while being no better at trading, and
in a market where the schedule is symmetric the extreme assignments are close to
worthless whoever holds them. Dividing by head count is the only comparison that
is not just counting bodies.

The population mix is a confound and it cannot be removed, only bounded. In a
mixed market the strategies trade against each other, so a profit figure for ZIP
is a figure for ZIP-in-this-population, not for ZIP. SHVR shaves the best quote
and needs somebody to shave; GVWY accepts whatever the book offers and needs
somebody to be offering. Change who else is present and the ranking can move.
This module therefore reports two things rather than one: the mixed-population
tournament, which is the interesting market, and every two-strategy head to head,
which holds the opponent fixed at the cost of no longer being a realistic market.
A strategy that wins both is winning for a reason that survives the mix. A
strategy that wins one and loses the other is telling you about the mix.

Statistics are not written here. The repository already has an assumption-aware
pipeline in fintech.hypothesis_tests, built for the week 5 strand and never once
pointed at the repository's own data, so it is what these comparisons call. That
pipeline tests independent groups, and inside one session the four strategies are
not independent: they share a customer order stream and the total surplus
available is fixed by the schedule, so one type's gain is partly another's loss.
A repeated-measures design would use that pairing and would have more power. The
independent-groups test is used anyway, because it is the one the repository
already owns and because it errs towards not finding a difference, which is the
safe direction for a claim about which algorithm wins. It is a limitation, not a
detail, and the experiment prints it alongside the numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from fintech.hypothesis_tests import DEFAULT_ALPHA, AnalysisResult, analyse, holm_correction
from fintech.market import SessionResult, run_sessions
from fintech.schedules import (
    OrderSchedule,
    TraderPopulation,
    fixed_schedule,
    mixed_population,
)

#: The four trader types the coursework population is built from. PRSH, PRDE and
#: the ZIPSH variants also ship with BSE but they adapt their own parameters mid
#: session, which is a different experiment from comparing fixed strategies.
STRATEGIES: tuple[str, ...] = ("ZIP", "ZIC", "SHVR", "GVWY")

DEFAULT_SEED = 100
DEFAULT_SESSION_SECONDS = 600.0


@dataclass(frozen=True)
class StrategyProfit:
    """One strategy's profit per trader, summarised across sessions."""

    strategy: str
    n_sessions: int
    n_traders: int
    mean: float
    std: float
    sem: float
    minimum: float
    maximum: float


@dataclass(frozen=True, eq=False)
class ProfitComparison:
    """A population, the profit it produced, and what the repo's pipeline said.

    `samples` has one row per session and one column per strategy, which is the
    shape fintech.hypothesis_tests expects: the session is the unit of
    replication, because traders inside a session are not independent of one
    another.
    """

    name: str
    description: str
    seed: int
    n_sessions: int
    samples: pd.DataFrame
    analysis: AnalysisResult
    profits: tuple[StrategyProfit, ...]

    @property
    def ranking(self) -> tuple[str, ...]:
        return tuple(profit.strategy for profit in sorted_profits(self.profits))

    @property
    def winner(self) -> str:
        return self.ranking[0]

    @property
    def significant(self) -> bool:
        return self.analysis.omnibus.significant

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": profit.strategy,
                    "sessions": profit.n_sessions,
                    "traders": profit.n_traders,
                    "mean_profit": profit.mean,
                    "std": profit.std,
                    "sem": profit.sem,
                    "min": profit.minimum,
                    "max": profit.maximum,
                }
                for profit in sorted_profits(self.profits)
            ]
        )


@dataclass(frozen=True)
class PairwiseResult:
    """One head to head, with its p-value corrected across the whole family."""

    left: str
    right: str
    n_sessions: int
    mean_left: float
    mean_right: float
    test: str
    p_value: float
    p_value_corrected: float
    significant: bool

    @property
    def winner(self) -> str:
        return self.left if self.mean_left >= self.mean_right else self.right

    @property
    def difference(self) -> float:
        """Mean profit of the winner minus mean profit of the loser."""

        return abs(self.mean_left - self.mean_right)


@dataclass(frozen=True, eq=False)
class HeadToHeadFamily:
    """Every two-strategy contest, treated as one family of comparisons.

    Six pairs from four strategies means six chances to find a difference that is
    not there, which is the same mistake the week 5 notebooks made with their
    uncorrected normality tests. The p-values are Holm corrected together.
    """

    comparisons: tuple[ProfitComparison, ...]
    pairs: tuple[PairwiseResult, ...]
    alpha: float = DEFAULT_ALPHA

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "left": pair.left,
                    "right": pair.right,
                    "sessions": pair.n_sessions,
                    "mean_left": pair.mean_left,
                    "mean_right": pair.mean_right,
                    "winner": pair.winner,
                    "difference": pair.difference,
                    "test": pair.test,
                    "p_value": pair.p_value,
                    "p_holm": pair.p_value_corrected,
                    "significant": pair.significant,
                }
                for pair in self.pairs
            ]
        )

    def wins(self) -> dict[str, int]:
        """How many head to heads each strategy won at the corrected threshold."""

        tally = dict.fromkeys(sorted(self.strategies), 0)
        for pair in self.pairs:
            if pair.significant:
                tally[pair.winner] += 1
        return tally

    @property
    def strategies(self) -> tuple[str, ...]:
        names: list[str] = []
        for pair in self.pairs:
            for name in (pair.left, pair.right):
                if name not in names:
                    names.append(name)
        return tuple(names)


@dataclass(frozen=True, eq=False)
class ProfitStudy:
    """Both halves of the answer, kept together so neither is quoted alone."""

    mixed: ProfitComparison
    head_to_head: HeadToHeadFamily
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def agrees(self) -> bool:
        """True when the mixed-market winner also wins every head to head it fought."""

        winner = self.mixed.winner
        fought = [pair for pair in self.head_to_head.pairs if winner in (pair.left, pair.right)]
        return bool(fought) and all(pair.winner == winner for pair in fought)


def sorted_profits(profits: Sequence[StrategyProfit]) -> tuple[StrategyProfit, ...]:
    """Most profitable first."""

    return tuple(sorted(profits, key=lambda profit: profit.mean, reverse=True))


def two_strategy_population(left: str, right: str, n_each: int = 10) -> TraderPopulation:
    """`n_each` of each of two types on both sides of the book.

    Both types sit on both sides deliberately. Putting one strategy on the buy
    side and the other on the sell side would compare them under different
    assignments, and the symmetric Chart 1 schedule is the only thing making the
    two sides comparable in the first place.
    """

    if left == right:
        raise ValueError(f"a head to head needs two different strategies, got {left!r} twice")
    if n_each <= 0:
        raise ValueError(f"n_each must be positive, got {n_each}")
    side = ((left, n_each), (right, n_each))
    return TraderPopulation(sellers=side, buyers=side, label=f"{left} vs {right} x{n_each}")


def profit_samples(
    sessions: Sequence[SessionResult], strategies: Sequence[str] | None = None
) -> pd.DataFrame:
    """One row per session, one column per strategy, holding mean profit per trader.

    A session that saw no trade still produces a row of zeros rather than being
    dropped, because a strategy that cannot get the market moving has genuinely
    earned nothing and silently discarding those sessions would flatter it.
    """

    if not sessions:
        raise ValueError("cannot summarise an empty sequence of sessions")

    if strategies is None:
        seen: list[str] = []
        for session in sessions:
            for name in session.final_balances:
                if name not in seen:
                    seen.append(name)
        strategies = seen
    if not strategies:
        raise ValueError("no trader types found in the balance dumps")

    rows = [
        {name: session.mean_profit_per_trader.get(name, 0.0) for name in strategies}
        for session in sessions
    ]
    return pd.DataFrame(rows, columns=list(strategies))


def summarise_profits(
    samples: pd.DataFrame, sessions: Sequence[SessionResult] | None = None
) -> tuple[StrategyProfit, ...]:
    """Per-strategy mean, spread and standard error over the session samples."""

    head_count: dict[str, int] = {}
    if sessions:
        for name, entry in sessions[0].final_balances.items():
            head_count[name] = entry.n_traders

    profits: list[StrategyProfit] = []
    for column in samples.columns:
        values = samples[column].to_numpy(dtype=float)
        n = int(values.size)
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        profits.append(
            StrategyProfit(
                strategy=str(column),
                n_sessions=n,
                n_traders=head_count.get(str(column), 0),
                mean=float(values.mean()),
                std=std,
                sem=std / np.sqrt(n) if n > 1 else 0.0,
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
        )
    return tuple(profits)


def compare_population(
    name: str,
    population: TraderPopulation,
    schedule: OrderSchedule | None = None,
    description: str = "",
    seed: int = DEFAULT_SEED,
    n_sessions: int = 20,
    end_time: float = DEFAULT_SESSION_SECONDS,
    alpha: float = DEFAULT_ALPHA,
) -> ProfitComparison:
    """Run one population many times and test whether its strategies differ in profit."""

    if schedule is None:
        schedule = fixed_schedule(0.0, end_time, interval=10.0, timemode="drip-poisson")
    sessions = run_sessions(
        schedule,
        population,
        seed=seed,
        n_sessions=n_sessions,
        end_time=end_time,
        session_id=_session_prefix(name),
    )
    samples = profit_samples(sessions)
    return ProfitComparison(
        name=name,
        description=description or population.label,
        seed=seed,
        n_sessions=len(sessions),
        samples=samples,
        analysis=analyse(samples, name=name, alpha=alpha),
        profits=summarise_profits(samples, sessions),
    )


def mixed_tournament(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 20,
    n_each: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    alpha: float = DEFAULT_ALPHA,
) -> ProfitComparison:
    """All four strategies in one market, at equal head count, over many seeds."""

    return compare_population(
        "mixed tournament",
        mixed_population(n_each),
        description=f"ZIP/ZIC/SHVR/GVWY, {n_each} each a side, drip-poisson 10s",
        seed=seed,
        n_sessions=n_sessions,
        end_time=end_time,
        alpha=alpha,
    )


def head_to_head(
    strategies: Sequence[str] = STRATEGIES,
    seed: int = DEFAULT_SEED,
    n_sessions: int = 20,
    n_each: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    alpha: float = DEFAULT_ALPHA,
) -> HeadToHeadFamily:
    """Every pair of strategies alone in a market together, Holm corrected as a family."""

    contests = list(combinations(strategies, 2))
    comparisons: list[ProfitComparison] = []
    for index, (left, right) in enumerate(contests):
        comparisons.append(
            compare_population(
                f"{left} vs {right}",
                two_strategy_population(left, right, n_each),
                description=f"{left} and {right} only, {n_each} each a side",
                # A different seed per pair, so the pairs are not all replaying
                # the same customer order stream.
                seed=seed + 1000 * (index + 1),
                n_sessions=n_sessions,
                end_time=end_time,
                alpha=alpha,
            )
        )

    corrected = holm_correction([c.analysis.omnibus.p_value for c in comparisons])
    pairs = tuple(
        PairwiseResult(
            left=left,
            right=right,
            n_sessions=comparison.n_sessions,
            mean_left=float(comparison.samples[left].mean()),
            mean_right=float(comparison.samples[right].mean()),
            test=comparison.analysis.omnibus.test,
            p_value=comparison.analysis.omnibus.p_value,
            p_value_corrected=float(adjusted),
            significant=bool(adjusted < alpha),
        )
        for (left, right), comparison, adjusted in zip(
            contests, comparisons, corrected, strict=True
        )
    )
    return HeadToHeadFamily(comparisons=tuple(comparisons), pairs=pairs, alpha=alpha)


def profit_study(
    seed: int = DEFAULT_SEED,
    n_sessions: int = 20,
    n_each: int = 10,
    end_time: float = DEFAULT_SESSION_SECONDS,
    alpha: float = DEFAULT_ALPHA,
) -> ProfitStudy:
    """The mixed market and the head to heads, run together and reported together."""

    mixed = mixed_tournament(
        seed=seed, n_sessions=n_sessions, n_each=n_each, end_time=end_time, alpha=alpha
    )
    pairs = head_to_head(
        seed=seed, n_sessions=n_sessions, n_each=n_each, end_time=end_time, alpha=alpha
    )
    notes = [
        "profit is mean per trader; BSE bounds each trader by the limit price it was "
        "assigned, so totals reward head count rather than skill",
        "the mixed figures measure a strategy against this particular set of opponents, "
        "not the strategy in isolation",
        f"{len(pairs.pairs)} head to heads are Holm corrected together",
        "the mixed-market columns come from the same sessions, so they are not "
        "independent groups; the test used treats them as though they were, which "
        "costs power rather than manufacturing significance",
    ]
    if not mixed.significant:
        notes.append(
            "the mixed-market omnibus test is not significant, so the mixed ranking is "
            "not evidence of a difference at all"
        )
    return ProfitStudy(mixed=mixed, head_to_head=pairs, notes=tuple(notes))


def _session_prefix(name: str) -> str:
    keep = [char if char.isalnum() else "_" for char in name.lower()]
    return "".join(keep).strip("_") or "session"
