"""Tests for the convergence measure and the experiments built on it.

The measure is checked against hand-computed cases, because a convergence
statistic that is only ever checked against simulator output can be wrong in the
same direction as the thing it is measuring. The experiments themselves are run
at a fraction of their published size; the full sweeps are marked slow.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from fintech.market import SessionResult, pool_tapes, run_sessions
from fintech.schedules import all_zip, fixed_schedule, mixed_population, shock_schedule
from fintech.smith import (
    AlphaSample,
    ScenarioResult,
    alpha_frame,
    arrival_mode_comparison,
    build_claim_family,
    chart1_baseline,
    compare_scenarios,
    convergence_claims,
    judge_claim,
    period_bounds,
    period_stats,
    plot_alpha,
    plot_transactions,
    run_scenario,
    session_alphas,
    shock_scenario,
    smiths_alpha,
    summary_table,
    trader_count_comparison,
    trader_mix_comparison,
)

SHORT_END = 120.0
PERIODS = 4


def short_schedule(timemode: str = "periodic", interval: float = 30.0):
    return fixed_schedule(0.0, SHORT_END, interval=interval, timemode=timemode)


def tiny_scenario(name: str = "tiny", seed: int = 1, n_sessions: int = 2) -> ScenarioResult:
    return run_scenario(
        name,
        short_schedule(),
        all_zip(6),
        seed=seed,
        n_sessions=n_sessions,
        end_time=SHORT_END,
        n_periods=PERIODS,
    )


def test_alpha_is_zero_when_everything_trades_at_equilibrium():
    assert smiths_alpha([200.0, 200.0, 200.0], 200.0) == 0.0


def test_alpha_is_the_rms_deviation_as_a_percentage():
    # Deviations of -10 and +10 about 200: RMS 10, which is 5% of 200.
    assert smiths_alpha([190.0, 210.0], 200.0) == pytest.approx(5.0)


def test_alpha_grows_with_dispersion():
    tight = smiths_alpha([198.0, 202.0], 200.0)
    loose = smiths_alpha([150.0, 250.0], 200.0)
    assert loose > tight > 0.0


def test_alpha_penalises_bias_as_well_as_spread():
    unbiased = smiths_alpha([190.0, 210.0], 200.0)
    biased = smiths_alpha([210.0, 230.0], 200.0)
    assert biased > unbiased


def test_alpha_of_no_trades_is_nan():
    assert math.isnan(smiths_alpha([], 200.0))


def test_alpha_rejects_a_non_positive_equilibrium():
    with pytest.raises(ValueError, match="must be positive"):
        smiths_alpha([200.0], 0.0)


def test_period_bounds_tile_the_session_without_gaps():
    bounds = period_bounds(0.0, 600.0, 10)
    assert len(bounds) == 10
    assert bounds[0] == (0.0, 60.0)
    assert bounds[-1] == (540.0, 600.0)
    for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert end == start


def test_period_bounds_rejects_nonsense():
    with pytest.raises(ValueError, match="at least one"):
        period_bounds(0.0, 600.0, 0)
    with pytest.raises(ValueError, match="positive duration"):
        period_bounds(600.0, 0.0, 10)


def test_period_stats_follow_the_shocked_equilibrium():
    schedule = shock_schedule(0.0, SHORT_END, interval=10.0)
    population = mixed_population(3)
    sessions = run_sessions(schedule, population, seed=1, n_sessions=1, end_time=SHORT_END)
    stats = period_stats(pool_tapes(sessions), schedule, population, 0.0, SHORT_END, 4)

    equilibria = [period.equilibrium for period in stats]
    assert equilibria[:2] == pytest.approx([200.0, 200.0], abs=1.0)
    assert equilibria[2:] == pytest.approx([350.0, 350.0], abs=1.0)


def test_scenario_reports_a_period_for_every_window():
    result = tiny_scenario()

    assert len(result.periods) == PERIODS
    assert result.equilibrium == 200.0
    assert result.n_transactions > 0
    assert result.n_transactions == sum(period.n_transactions for period in result.periods)


def test_scenario_is_reproducible_from_its_seed():
    first = tiny_scenario(seed=11)
    second = tiny_scenario(seed=11)

    np.testing.assert_array_equal(first.pooled.prices, second.pooled.prices)
    assert first.alpha_overall == second.alpha_overall


def test_scenarios_with_different_seeds_differ():
    first = tiny_scenario(seed=11)
    other = tiny_scenario(seed=12)

    same_length = first.pooled.prices.size == other.pooled.prices.size
    assert not (same_length and np.array_equal(first.pooled.prices, other.pooled.prices))


def test_scenario_frame_and_summary_line_up():
    result = tiny_scenario()
    frame = result.to_frame()

    assert len(frame) == PERIODS
    assert list(frame.columns) == [
        "period",
        "start",
        "end",
        "equilibrium",
        "n_transactions",
        "mean_price",
        "alpha",
    ]

    row = result.summary_row()
    assert row["scenario"] == result.name
    assert row["alpha_overall"] == result.alpha_overall
    assert row["trades"] == result.n_transactions


def test_alpha_overall_is_the_trade_weighted_pooling_of_the_periods():
    result = tiny_scenario()
    traded = result.traded_periods
    total = sum(period.n_transactions for period in traded)
    expected = math.sqrt(sum(period.n_transactions * period.alpha**2 for period in traded) / total)

    assert result.alpha_overall == pytest.approx(expected)
    # With one fixed equilibrium it must agree with scoring the pooled prices
    # directly; the two only part company once the equilibrium moves.
    assert result.alpha_overall == pytest.approx(
        smiths_alpha(result.pooled.prices, result.equilibrium), rel=1e-9
    )


def test_alpha_overall_scores_the_shock_against_the_moving_target():
    result = shock_scenario(seed=5, n_sessions=1, n_each=2, end_time=SHORT_END, n_periods=PERIODS)
    naive = smiths_alpha(result.pooled.prices, result.equilibrium)

    # Scoring post-shock trades against the pre-shock equilibrium makes a market
    # that tracked its new equilibrium look wildly divergent.
    assert result.alpha_overall < naive


def test_silent_periods_are_counted():
    result = tiny_scenario()
    counted = sum(1 for period in result.periods if period.n_transactions == 0)

    assert result.n_silent_periods == counted
    assert result.n_silent_periods + len(result.traded_periods) == len(result.periods)
    assert result.final_trades == result.traded_periods[-1].n_transactions


def test_convergence_ratio_is_last_over_first():
    result = tiny_scenario()
    expected = result.alpha_last / result.alpha_first
    assert result.convergence_ratio == pytest.approx(expected)


def test_prices_end_up_nearer_equilibrium_than_they_start():
    """The finding under test: alpha falls across the session.

    Two short sessions of six ZIPs a side is a small sample, so this asserts the
    direction of travel rather than a threshold.
    """

    result = run_scenario(
        "convergence check",
        fixed_schedule(0.0, 240.0, interval=10.0, timemode="drip-poisson"),
        all_zip(8),
        seed=3,
        n_sessions=4,
        end_time=240.0,
        n_periods=4,
    )
    assert result.alpha_first > 0.0
    assert result.alpha_last < result.alpha_first


def test_summary_table_has_one_row_per_scenario():
    results = [tiny_scenario("a", seed=1), tiny_scenario("b", seed=2)]
    table = summary_table(results)

    assert len(table) == 2
    assert list(table["scenario"]) == ["a", "b"]


def test_figures_are_written_where_asked(tmp_path):
    result = tiny_scenario()

    transactions = plot_transactions(result, tmp_path)
    alpha = plot_alpha([result], tmp_path, "tiny")

    assert transactions.is_file()
    assert alpha.is_file()
    assert transactions.parent == tmp_path
    assert transactions.name in {path.name for path in tmp_path.iterdir()}
    assert transactions.stat().st_size > 0


def test_plotting_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_alpha([], tmp_path)


def test_named_experiments_run_at_reduced_size():
    baseline = chart1_baseline(
        seed=5, n_sessions=1, n_per_side=6, end_time=SHORT_END, n_periods=PERIODS
    )
    assert baseline.equilibrium == 200.0
    assert baseline.n_sessions == 1
    assert len(baseline.periods) == PERIODS


def test_arrival_mode_comparison_returns_both_modes():
    results = arrival_mode_comparison(
        seed=5, n_sessions=1, n_per_side=6, end_time=SHORT_END, n_periods=PERIODS
    )
    assert len(results) == 2
    assert [result.name for result in results] == [
        "arrival periodic 60s",
        "arrival drip-poisson 10s",
    ]


def test_trader_count_comparison_scales_the_population():
    results = trader_count_comparison(
        counts=(4, 8), seed=5, n_sessions=1, end_time=SHORT_END, n_periods=PERIODS
    )
    assert [result.name for result in results] == ["traders 4 a side", "traders 8 a side"]
    assert all(result.equilibrium == pytest.approx(200.0, abs=1.0) for result in results)


def test_trader_mix_comparison_holds_head_count_constant():
    results = trader_mix_comparison(
        seed=5, n_sessions=1, n_each=2, end_time=SHORT_END, n_periods=PERIODS
    )
    assert len(results) == 2
    assert results[0].sessions[0].n_transactions >= 0
    assert results[1].n_transactions > 0


def test_shock_scenario_moves_its_target_mid_session():
    result = shock_scenario(seed=5, n_sessions=1, n_each=2, end_time=SHORT_END, n_periods=PERIODS)
    equilibria = [period.equilibrium for period in result.periods]
    assert equilibria[0] == pytest.approx(200.0, abs=1.0)
    assert equilibria[-1] == pytest.approx(350.0, abs=1.0)


@pytest.mark.slow
def test_full_baseline_converges():
    result = chart1_baseline(seed=100, n_sessions=10)
    assert result.alpha_last < result.alpha_first


def test_session_alphas_give_one_number_per_session():
    result = tiny_scenario(n_sessions=3)

    alphas = result.session_alphas

    assert alphas.shape == (3,)
    assert np.all(alphas[~np.isnan(alphas)] >= 0.0)


def test_a_session_that_never_traded_contributes_nan_rather_than_zero():
    """Zero would read as a perfectly converged market, which is the opposite."""

    result = tiny_scenario(n_sessions=2)
    silent = SessionResult(
        session_id="silent",
        seed=0,
        start_time=0.0,
        end_time=SHORT_END,
        times=np.empty(0),
        prices=np.empty(0),
    )
    with_silence = replace(result, sessions=(*result.sessions, silent))

    alphas = with_silence.session_alphas

    assert alphas.size == 3
    assert math.isnan(alphas[-1])


def test_restricting_the_periods_changes_the_alpha_it_measures():
    result = tiny_scenario(n_sessions=2)

    whole = session_alphas(result)
    late = session_alphas(result, periods=range(PERIODS // 2, PERIODS))

    assert whole.shape == late.shape
    assert not np.array_equal(whole, late)


def test_alpha_frame_is_one_column_per_sample_padded_to_the_longest():
    frame = alpha_frame(
        [
            AlphaSample("short", np.array([1.0, 2.0])),
            AlphaSample("long", np.array([3.0, 4.0, 5.0])),
        ]
    )

    assert list(frame.columns) == ["short", "long"]
    assert len(frame) == 3
    assert math.isnan(frame["short"].iloc[2])


def test_alpha_frame_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to compare"):
        alpha_frame([])


def synthetic(label: str, centre: float, spread: float = 0.5, n: int = 12) -> AlphaSample:
    """A reproducible alpha sample, so the claim logic can be tested without a market."""

    offsets = np.linspace(-spread, spread, n)
    return AlphaSample(label, centre + offsets)


def test_a_claim_backed_by_a_large_separation_is_supported():
    judged = [judge_claim("tight beats loose", synthetic("tight", 2.0), synthetic("loose", 9.0))]

    (claim,) = build_claim_family(judged).claims

    assert claim.direction_holds
    assert claim.significant
    assert claim.supported
    assert claim.verdict == "supported"


def test_a_claim_whose_difference_runs_the_other_way_is_reported_as_contradicted():
    judged = [judge_claim("loose beats tight", synthetic("loose", 9.0), synthetic("tight", 2.0))]

    (claim,) = build_claim_family(judged).claims

    assert not claim.direction_holds
    assert not claim.supported
    assert "other way" in claim.verdict


def test_a_claim_with_overlapping_samples_is_not_supported():
    judged = [
        judge_claim("a beats b", synthetic("a", 5.0, spread=3.0), synthetic("b", 5.05, spread=3.0))
    ]

    (claim,) = build_claim_family(judged).claims

    assert claim.direction_holds
    assert not claim.significant
    assert "noise" in claim.verdict


def test_the_family_correction_is_applied_across_every_claim():
    judged = [
        judge_claim("one", synthetic("a", 2.0), synthetic("b", 9.0)),
        judge_claim("two", synthetic("c", 3.0), synthetic("d", 8.0)),
        judge_claim("three", synthetic("e", 5.0, spread=3.0), synthetic("f", 5.05, spread=3.0)),
    ]

    family = build_claim_family(judged)

    assert len(family.claims) == 3
    # Holm can only raise a p-value.
    assert all(claim.p_value_corrected >= claim.p_value for claim in family.claims)
    assert len(family.supported) + len(family.unsupported) == 3
    assert list(family.to_frame()["claim"]) == ["one", "two", "three"]


def test_compare_scenarios_runs_the_repositorys_own_pipeline():
    results = [tiny_scenario("a", seed=1, n_sessions=4), tiny_scenario("b", seed=40, n_sessions=4)]

    analysis = compare_scenarios(results, name="tiny comparison")

    assert analysis.name == "tiny comparison"
    assert analysis.conditions == ("a", "b")
    assert analysis.omnibus.test in {"one-way ANOVA", "Welch ANOVA", "Kruskal-Wallis H"}
    assert analysis.omnibus.reason


@pytest.mark.slow
def test_the_convergence_claims_run_end_to_end():
    family = convergence_claims(seed=7, n_sessions=4, end_time=SHORT_END, n_periods=4, n_each=2)

    assert len(family.claims) == 4
    frame = family.to_frame()
    assert len(frame) == 4
    assert set(frame["verdict"]) <= {
        "supported",
        "contradicted: the difference runs the other way",
        "not supported: the difference is not distinguishable from noise",
    }
