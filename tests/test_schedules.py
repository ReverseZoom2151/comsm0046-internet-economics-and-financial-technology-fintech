"""Schedules are the part most easily got wrong in silence, so they are checked hard."""

from __future__ import annotations

import pytest

from fintech.schedules import (
    CHART1_RANGE,
    SHOCK_RANGE,
    OrderSchedule,
    PriceRange,
    ScheduleStep,
    TraderPopulation,
    all_zip,
    equilibrium_at,
    equilibrium_price,
    fixed_schedule,
    limit_prices,
    mixed_population,
    shock_schedule,
)


def test_price_range_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="low must not exceed high"):
        PriceRange(320, 80)


def test_price_range_rejects_non_positive_prices():
    with pytest.raises(ValueError, match="positive"):
        PriceRange(0, 100)


def test_schedule_step_rejects_zero_duration():
    with pytest.raises(ValueError, match="positive duration"):
        ScheduleStep(10.0, 10.0, CHART1_RANGE)


def test_schedule_step_rejects_unknown_stepmode():
    with pytest.raises(ValueError, match="unknown stepmode"):
        ScheduleStep(0.0, 60.0, CHART1_RANGE, stepmode="wobbly")


def test_order_schedule_rejects_unknown_timemode():
    with pytest.raises(ValueError, match="unknown timemode"):
        fixed_schedule(0.0, 60.0, timemode="whenever")


def test_order_schedule_rejects_overlapping_steps():
    steps = (ScheduleStep(0.0, 100.0, CHART1_RANGE), ScheduleStep(50.0, 200.0, SHOCK_RANGE))
    with pytest.raises(ValueError, match="overlap"):
        OrderSchedule(supply=steps, demand=steps, interval=10.0)


def test_fixed_schedule_matches_the_shape_bse_expects():
    schedule = fixed_schedule(0.0, 600.0, interval=60.0)
    spec = schedule.as_bse()

    assert set(spec) == {"sup", "dem", "interval", "timemode"}
    assert spec["interval"] == 60.0
    assert spec["timemode"] == "periodic"
    for side in ("sup", "dem"):
        assert len(spec[side]) == 1
        step = spec[side][0]
        assert set(step) == {"from", "to", "ranges", "stepmode"}
        assert step["ranges"] == [(80, 320)]
        assert step["stepmode"] == "fixed"


def test_shock_schedule_switches_range_half_way():
    schedule = shock_schedule(0.0, 600.0)
    supply, demand = schedule.step_at(100.0)
    assert supply.price_range == CHART1_RANGE
    assert demand.price_range == CHART1_RANGE

    supply, demand = schedule.step_at(500.0)
    assert supply.price_range == SHOCK_RANGE
    assert demand.price_range == SHOCK_RANGE

    spec = schedule.as_bse()
    assert [step["from"] for step in spec["sup"]] == [0.0, 300.0]
    assert [step["to"] for step in spec["sup"]] == [300.0, 600.0]


def test_shock_time_must_fall_inside_the_session():
    with pytest.raises(ValueError, match="must fall inside"):
        shock_schedule(0.0, 600.0, shock_time=900.0)


def test_step_at_clamps_beyond_the_end_of_the_schedule():
    schedule = shock_schedule(0.0, 600.0)
    supply, _ = schedule.step_at(10_000.0)
    assert supply.price_range == SHOCK_RANGE


def test_populations_report_their_head_count():
    assert all_zip(11).n_buyers == 11
    assert all_zip(11).n_sellers == 11
    assert all_zip(11).size == 22

    mixed = mixed_population(10)
    assert mixed.n_buyers == 40
    assert mixed.size == 80
    assert mixed.as_bse()["sellers"] == [("ZIP", 10), ("ZIC", 10), ("SHVR", 10), ("GVWY", 10)]


def test_population_rejects_empty_or_zero_sides():
    with pytest.raises(ValueError, match="at least one trader type"):
        TraderPopulation(sellers=(), buyers=(("ZIP", 4),))
    with pytest.raises(ValueError, match="positive count"):
        TraderPopulation(sellers=(("ZIP", 0),), buyers=(("ZIP", 4),))


def test_limit_prices_reproduce_the_chart1_ladder():
    prices = limit_prices(CHART1_RANGE, 11)
    assert prices == [80, 104, 128, 152, 176, 200, 224, 248, 272, 296, 320]
    assert prices[0] == CHART1_RANGE.low
    assert prices[-1] == CHART1_RANGE.high


def test_limit_prices_needs_two_traders():
    with pytest.raises(ValueError, match="at least two"):
        limit_prices(CHART1_RANGE, 1)


def test_symmetric_schedule_clears_at_the_midpoint():
    assert equilibrium_price(CHART1_RANGE, CHART1_RANGE, 11, 11) == 200.0
    assert equilibrium_price(SHOCK_RANGE, SHOCK_RANGE, 11, 11) == 350.0


def test_truncated_ladders_shift_the_clearing_price_by_half_a_unit():
    """BSE truncates each step with int(), so the ladder is not always symmetric.

    With eleven traders the step of 24 is exact and the market clears at 200.
    With forty it is 6.15, truncation nudges both ladders down, and the clearing
    interval sits half a unit below the range midpoint. Quoting alpha against a
    hard-coded 200 in that case would report a bias that is an artefact of the
    assignment code, so the equilibrium is derived from the same arithmetic.
    """

    assert CHART1_RANGE.midpoint == 200.0
    assert equilibrium_price(CHART1_RANGE, CHART1_RANGE, 11, 11) == CHART1_RANGE.midpoint
    assert equilibrium_price(CHART1_RANGE, CHART1_RANGE, 40, 40) == 199.5


def test_schedule_reports_its_own_span():
    schedule = shock_schedule(0.0, 600.0)
    assert schedule.start == 0.0
    assert schedule.end == 600.0


def test_equilibrium_moves_when_demand_shifts_up():
    dearer = PriceRange(180, 420)
    equilibrium = equilibrium_price(CHART1_RANGE, dearer, 11, 11)
    assert equilibrium > 200.0


def test_non_overlapping_schedules_have_no_equilibrium():
    with pytest.raises(ValueError, match="do not overlap"):
        equilibrium_price(PriceRange(400, 500), PriceRange(80, 320), 11, 11)


def test_equilibrium_at_tracks_the_shock():
    schedule = shock_schedule(0.0, 600.0)
    population = mixed_population(10)
    assert equilibrium_at(schedule, population, 100.0) == pytest.approx(200.0, abs=1.0)
    assert equilibrium_at(schedule, population, 500.0) == pytest.approx(350.0, abs=1.0)
