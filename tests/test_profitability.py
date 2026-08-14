"""Tests for the profit measure, kept away from the simulator wherever possible.

Most of what can go wrong here is arithmetic and bookkeeping: reading the wrong
column out of the balance dump, comparing totals when head counts differ, or
treating a trader rather than a session as the unit of replication. None of that
needs a market to be simulated, so the fast tests build session results by hand
and only the two marked slow actually run BSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fintech.hypothesis_tests import DegenerateDataError, analyse
from fintech.market import BalanceSnapshot, SessionResult, StrategyBalance
from fintech.profitability import (
    STRATEGIES,
    head_to_head,
    mixed_tournament,
    profit_samples,
    sorted_profits,
    summarise_profits,
    two_strategy_population,
)

SHORT_END = 120.0


def fake_session(profits: dict[str, float], n_traders: int = 4) -> SessionResult:
    """A session result carrying nothing but a final balance row."""

    strategies = {
        name: StrategyBalance(
            strategy=name,
            total_profit=value * n_traders,
            n_traders=n_traders,
            mean_profit=value,
        )
        for name, value in profits.items()
    }
    snapshot = BalanceSnapshot(time=600.0, best_bid=100.0, best_ask=101.0, strategies=strategies)
    return SessionResult(
        session_id="fake",
        seed=0,
        start_time=0.0,
        end_time=600.0,
        times=np.empty(0),
        prices=np.empty(0),
        balances=(snapshot,),
    )


def fake_sessions(series: dict[str, list[float]]) -> list[SessionResult]:
    length = {len(values) for values in series.values()}
    assert len(length) == 1, "every strategy needs the same number of sessions"
    return [
        fake_session({name: values[index] for name, values in series.items()})
        for index in range(length.pop())
    ]


def test_profit_samples_is_one_row_per_session_and_one_column_per_strategy():
    sessions = fake_sessions({"ZIP": [10.0, 20.0, 30.0], "ZIC": [1.0, 2.0, 3.0]})

    samples = profit_samples(sessions)

    assert list(samples.columns) == ["ZIP", "ZIC"]
    assert len(samples) == 3
    assert samples["ZIP"].tolist() == [10.0, 20.0, 30.0]


def test_profit_samples_reports_mean_per_trader_not_the_total():
    # Twice the head count on the same per-trader profit must not read as twice
    # as good a strategy.
    small = fake_session({"ZIP": 50.0}, n_traders=4)
    large = fake_session({"ZIP": 50.0}, n_traders=40)

    assert small.final_balances["ZIP"].total_profit == 200.0
    assert large.final_balances["ZIP"].total_profit == 2000.0
    assert profit_samples([small])["ZIP"].iloc[0] == profit_samples([large])["ZIP"].iloc[0]


def test_a_strategy_missing_from_a_session_counts_as_earning_nothing():
    sessions = [fake_session({"ZIP": 10.0, "ZIC": 5.0}), fake_session({"ZIP": 10.0})]

    samples = profit_samples(sessions, strategies=["ZIP", "ZIC"])

    assert samples["ZIC"].tolist() == [5.0, 0.0]


def test_profit_samples_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="empty sequence"):
        profit_samples([])


def test_summarise_reports_spread_and_the_standard_error_of_the_mean():
    sessions = fake_sessions({"ZIP": [10.0, 20.0, 30.0]})
    samples = profit_samples(sessions)

    profit = summarise_profits(samples, sessions)[0]

    assert profit.strategy == "ZIP"
    assert profit.n_sessions == 3
    assert profit.n_traders == 4
    assert profit.mean == pytest.approx(20.0)
    assert profit.std == pytest.approx(10.0)
    assert profit.sem == pytest.approx(10.0 / np.sqrt(3))
    assert (profit.minimum, profit.maximum) == (10.0, 30.0)


def test_ranking_is_by_mean_profit_descending():
    sessions = fake_sessions({"ZIP": [1.0, 2.0, 3.0], "GVWY": [10.0, 20.0, 30.0]})
    profits = summarise_profits(profit_samples(sessions), sessions)

    assert [profit.strategy for profit in sorted_profits(profits)] == ["GVWY", "ZIP"]


def test_a_two_strategy_population_puts_both_types_on_both_sides():
    population = two_strategy_population("ZIP", "GVWY", n_each=5)

    assert population.sellers == (("ZIP", 5), ("GVWY", 5))
    assert population.buyers == population.sellers
    assert population.n_sellers == population.n_buyers == 10


def test_a_head_to_head_needs_two_different_strategies():
    with pytest.raises(ValueError, match="two different strategies"):
        two_strategy_population("ZIP", "ZIP")


def test_a_head_to_head_needs_a_positive_head_count():
    with pytest.raises(ValueError, match="must be positive"):
        two_strategy_population("ZIP", "GVWY", n_each=0)


def test_a_strategy_that_never_earns_anything_is_refused_rather_than_tested():
    """A constant column makes every test statistic undefined.

    The interesting case is a strategy that earns exactly zero in every session,
    which is a real outcome in a market that never trades. The pipeline must say
    so rather than return a NaN p-value that compares false against 0.05 and
    reads as "no significant difference".
    """

    sessions = fake_sessions({"ZIP": [10.0, 20.0, 30.0], "DEAD": [0.0, 0.0, 0.0]})

    with pytest.raises(DegenerateDataError, match="constant"):
        analyse(profit_samples(sessions))


@pytest.mark.slow
def test_the_mixed_tournament_measures_every_strategy_and_tests_them_together():
    comparison = mixed_tournament(n_sessions=4, n_each=3, end_time=SHORT_END, seed=11)

    assert set(comparison.samples.columns) == set(STRATEGIES)
    assert len(comparison.samples) == 4
    assert comparison.winner in STRATEGIES
    assert set(comparison.ranking) == set(STRATEGIES)
    # Profit is bounded by the assignment, so nobody can earn a negative amount
    # and nobody can earn an unbounded one.
    assert comparison.samples.to_numpy().min() >= 0.0
    assert comparison.analysis.omnibus.test in {
        "one-way ANOVA",
        "Welch ANOVA",
        "Kruskal-Wallis H",
    }
    assert isinstance(comparison.summary_frame(), pd.DataFrame)


@pytest.mark.slow
def test_head_to_head_covers_every_pair_and_corrects_across_the_family():
    family = head_to_head(
        strategies=("ZIP", "ZIC", "GVWY"), n_sessions=4, n_each=3, end_time=SHORT_END, seed=21
    )

    assert len(family.pairs) == 3
    assert {(pair.left, pair.right) for pair in family.pairs} == {
        ("ZIP", "ZIC"),
        ("ZIP", "GVWY"),
        ("ZIC", "GVWY"),
    }
    # Holm can only ever raise a p-value, never lower it.
    assert all(pair.p_value_corrected >= pair.p_value for pair in family.pairs)
    assert all(pair.winner in (pair.left, pair.right) for pair in family.pairs)
    assert set(family.wins()) == {"ZIP", "ZIC", "GVWY"}
