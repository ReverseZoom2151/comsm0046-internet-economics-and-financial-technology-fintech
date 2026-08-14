"""Invariants that must hold for every valid input, not just the ones I thought of.

The rest of the suite is example based: it pins particular seeds, particular
datasets and particular schedules, and it is very good at catching a regression
in a number that has already been computed once. What it cannot do is tell me
whether the number was right for a reason. A test that asserts
`equilibrium_price(CHART1_RANGE, CHART1_RANGE, 11, 11) == 200.0` passes just as
happily against an implementation that returns the midpoint of the range and
ignores the ladder entirely.

Property based testing attacks the same code from the other end. Instead of
naming an input and its expected output, each test here states a relationship
that has to survive any input at all, and lets Hypothesis hunt for the
counterexample: alpha is a ratio, so rescaling every price and the equilibrium
together must not move it; the tape parser filters on the event type, so no
amount of cancellation rows may leak a price; the omnibus test is chosen from
the assumptions, so a rejected normality test must never be followed by a
parametric test. Those are the claims the modules' docstrings actually make, and
they are the claims worth testing, because a counterexample to any of them is a
bug rather than a changed number.

Runtimes are kept honest by leaning on the cheap pure functions for volume and
keeping the simulated sessions short, small and few.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from fintech.hypothesis_tests import (
    DegenerateDataError,
    analyse,
    holm_correction,
)
from fintech.market import parse_tape, run_session
from fintech.schedules import (
    OrderSchedule,
    PriceRange,
    ScheduleStep,
    TraderPopulation,
    all_zip,
    equilibrium_price,
    fixed_schedule,
)
from fintech.smith import smiths_alpha

TRADER_TYPES = ("ZIP", "ZIC", "SHVR", "GVWY")

#: Sessions in this file are deliberately tiny. The economics is measured in
#: test_smith; here a session is only ever asked whether it obeyed its contract.
CHEAP = settings(max_examples=75, deadline=None)
MODERATE = settings(max_examples=20, deadline=None)
SIMULATED = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --- strategies ----------------------------------------------------------------


@st.composite
def price_ranges(draw, min_low: int = 1, max_high: int = 5_000) -> PriceRange:
    """A valid price range: positive, and low never above high."""

    low = draw(st.integers(min_value=min_low, max_value=max_high))
    high = draw(st.integers(min_value=low, max_value=max_high))
    return PriceRange(low, high)


@st.composite
def schedule_steps(draw, count: int = 1) -> tuple[ScheduleStep, ...]:
    """`count` contiguous, strictly increasing time slices."""

    start = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    durations = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=200.0, allow_nan=False),
            min_size=count,
            max_size=count,
        )
    )
    stepmode = draw(st.sampled_from(["fixed", "jittered", "random"]))
    steps: list[ScheduleStep] = []
    cursor = start
    for duration in durations:
        steps.append(ScheduleStep(cursor, cursor + duration, draw(price_ranges()), stepmode))
        cursor += duration
    return tuple(steps)


@st.composite
def order_schedules(draw) -> OrderSchedule:
    """A whole schedule, both sides, with an arrival mode BSE recognises."""

    count = draw(st.integers(min_value=1, max_value=3))
    return OrderSchedule(
        supply=draw(schedule_steps(count)),
        demand=draw(schedule_steps(count)),
        interval=draw(st.floats(min_value=1.0, max_value=120.0, allow_nan=False)),
        timemode=draw(st.sampled_from(["periodic", "drip-fixed", "drip-jitter", "drip-poisson"])),
    )


def trader_sides():
    return st.lists(
        st.tuples(st.sampled_from(TRADER_TYPES), st.integers(min_value=1, max_value=25)),
        min_size=1,
        max_size=4,
    )


@st.composite
def populations(draw) -> TraderPopulation:
    return TraderPopulation(
        sellers=tuple(draw(trader_sides())),
        buyers=tuple(draw(trader_sides())),
    )


def measurements(min_size: int = 4, max_size: int = 25):
    """Distinct finite values, so a generated column is never constant."""

    return st.lists(
        st.floats(
            min_value=-1_000.0,
            max_value=1_000.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


@st.composite
def frames(draw, min_columns: int = 2, max_columns: int = 3) -> pd.DataFrame:
    """A tidy frame of one column per condition, all columns the same length."""

    n_columns = draw(st.integers(min_value=min_columns, max_value=max_columns))
    n_rows = draw(st.integers(min_value=4, max_value=25))
    columns = {}
    for index in range(n_columns):
        values = draw(measurements(min_size=n_rows, max_size=n_rows))
        columns[f"c{index}"] = values
    return pd.DataFrame(columns)


# --- schedules -----------------------------------------------------------------


@given(
    low=st.integers(min_value=1, max_value=10_000), high=st.integers(min_value=1, max_value=10_000)
)
@CHEAP
def test_a_price_range_with_low_above_high_is_always_refused(low, high):
    assume(low > high)
    with pytest.raises(ValueError, match="low must not exceed high"):
        PriceRange(low, high)


@given(price_range=price_ranges())
@CHEAP
def test_a_valid_price_range_brackets_its_own_midpoint(price_range):
    assert price_range.low <= price_range.midpoint <= price_range.high
    assert price_range.as_bse() == (price_range.low, price_range.high)


@given(schedule=order_schedules())
@CHEAP
def test_the_emitted_dict_has_the_keys_bse_requires(schedule):
    """BSE indexes this dict by name, so a missing key is a mid-run KeyError."""

    spec = schedule.as_bse()
    assert set(spec) == {"sup", "dem", "interval", "timemode"}
    assert spec["interval"] > 0.0
    assert spec["timemode"] == schedule.timemode

    for side in ("sup", "dem"):
        assert spec[side], "each side needs at least one time slice"
        for step in spec[side]:
            assert set(step) == {"from", "to", "ranges", "stepmode"}
            assert step["to"] > step["from"]
            assert len(step["ranges"]) == 1
            low, high = step["ranges"][0]
            assert 0 < low <= high


@given(schedule=order_schedules())
@CHEAP
def test_time_slices_are_ordered_and_never_overlap(schedule):
    spec = schedule.as_bse()
    for side in ("sup", "dem"):
        boundaries = [(step["from"], step["to"]) for step in spec[side]]
        for (_, earlier_end), (later_start, _) in zip(boundaries, boundaries[1:], strict=False):
            assert later_start >= earlier_end
    assert schedule.start <= schedule.end


@given(schedule=order_schedules())
@CHEAP
def test_as_bse_round_trips_the_values_the_schedule_was_built_from(schedule):
    spec = schedule.as_bse()
    assert spec["interval"] == schedule.interval
    assert spec["timemode"] == schedule.timemode
    for side, steps in (("sup", schedule.supply), ("dem", schedule.demand)):
        assert len(spec[side]) == len(steps)
        for emitted, step in zip(spec[side], steps, strict=True):
            assert emitted["from"] == step.start
            assert emitted["to"] == step.end
            assert emitted["stepmode"] == step.stepmode
            assert emitted["ranges"] == [(step.price_range.low, step.price_range.high)]


@given(population=populations())
@CHEAP
def test_a_population_sums_to_the_head_count_it_was_asked_for(population):
    assert population.n_sellers == sum(count for _, count in population.sellers)
    assert population.n_buyers == sum(count for _, count in population.buyers)
    assert population.size == population.n_sellers + population.n_buyers
    assert population.size > 0

    spec = population.as_bse()
    assert sum(count for _, count in spec["sellers"]) == population.n_sellers
    assert sum(count for _, count in spec["buyers"]) == population.n_buyers


@given(
    price_range=price_ranges(),
    n_sellers=st.integers(min_value=2, max_value=60),
    n_buyers=st.integers(min_value=2, max_value=60),
)
@CHEAP
def test_the_equilibrium_lies_inside_the_range_the_schedule_spans(price_range, n_sellers, n_buyers):
    """A clearing price outside the ladder would be a price nobody could have quoted."""

    equilibrium = equilibrium_price(price_range, price_range, n_sellers, n_buyers)
    assert math.isfinite(equilibrium)
    assert price_range.low <= equilibrium <= price_range.high


@given(count=st.integers(min_value=2, max_value=40))
@CHEAP
def test_a_symmetric_ladder_never_clears_outside_its_own_limit_prices(count):
    schedule = fixed_schedule(0.0, 600.0)
    population = all_zip(count)
    step, _ = schedule.step_at(0.0)
    equilibrium = equilibrium_price(
        step.price_range, step.price_range, population.n_sellers, population.n_buyers
    )
    assert step.price_range.low <= equilibrium <= step.price_range.high


# --- market and tape -----------------------------------------------------------

SESSION_SECONDS = st.floats(min_value=60.0, max_value=120.0, allow_nan=False)
SEEDS = st.integers(min_value=0, max_value=10_000)
SIDE_SIZES = st.integers(min_value=4, max_value=8)
ARRIVALS = st.sampled_from([("periodic", 30.0), ("drip-poisson", 10.0)])


@given(seed=SEEDS, n_per_side=SIDE_SIZES, arrival=ARRIVALS)
@SIMULATED
def test_the_same_seed_always_gives_an_identical_tape(seed, n_per_side, arrival):
    timemode, interval = arrival
    schedule = fixed_schedule(0.0, 120.0, interval=interval, timemode=timemode)
    first = run_session(schedule, all_zip(n_per_side), seed=seed, end_time=120.0)
    second = run_session(schedule, all_zip(n_per_side), seed=seed, end_time=120.0)

    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.prices, second.prices)


@given(seed=st.integers(min_value=0, max_value=5_000), n_per_side=SIDE_SIZES)
@SIMULATED
def test_different_seeds_give_different_tapes(seed, n_per_side):
    """Stated so it cannot flake.

    Two empty tapes are legitimately identical, so the claim is only made once
    the market has actually traded. A full two minutes and at least four traders
    a side is enough time for that, but the assume() keeps the property honest
    rather than relying on it.
    """

    schedule = fixed_schedule(0.0, 120.0, interval=10.0, timemode="drip-poisson")
    first = run_session(schedule, all_zip(n_per_side), seed=seed, end_time=120.0)
    other = run_session(schedule, all_zip(n_per_side), seed=seed + 1, end_time=120.0)

    assume(first.n_transactions > 2 and other.n_transactions > 2)
    same_length = first.prices.size == other.prices.size
    assert not (same_length and np.array_equal(first.prices, other.prices))


@given(seed=SEEDS, n_per_side=SIDE_SIZES, arrival=ARRIVALS)
@SIMULATED
def test_every_transaction_price_is_finite_positive_and_sane(seed, n_per_side, arrival):
    timemode, interval = arrival
    schedule = fixed_schedule(0.0, 120.0, interval=interval, timemode=timemode)
    step, _ = schedule.step_at(0.0)
    result = run_session(schedule, all_zip(n_per_side), seed=seed, end_time=120.0)

    prices = result.prices
    assert np.isfinite(prices).all()
    assert (prices > 0.0).all()
    # A trade cannot be licensed by a limit price nobody holds, so anything
    # outside a factor of two either side of the ladder is a parsing error
    # rather than an aggressive quote.
    assert (prices >= 0.5 * step.price_range.low).all()
    assert (prices <= 2.0 * step.price_range.high).all()


@given(seed=SEEDS, end_time=SESSION_SECONDS, n_per_side=SIDE_SIZES)
@SIMULATED
def test_transaction_times_are_ordered_and_inside_the_session_window(seed, end_time, n_per_side):
    schedule = fixed_schedule(0.0, end_time, interval=10.0, timemode="drip-poisson")
    result = run_session(schedule, all_zip(n_per_side), seed=seed, end_time=end_time)

    times = result.times
    assert times.size == result.prices.size
    assert np.all(np.diff(times) >= 0.0), "the tape is written in the order trades happened"
    if times.size:
        assert times.min() >= 0.0
        assert times.max() <= end_time


#: Order ids are drawn from a band that no generated trade price can reach, so
#: a leaked cancellation is visible in the parsed prices rather than plausible.
ORDER_IDS = st.integers(min_value=100_001, max_value=110_000)
TRADE_PRICES = st.integers(min_value=1, max_value=10_000)
TAPE_TIMES = st.floats(min_value=0.0, max_value=600.0, allow_nan=False)


@given(
    cancellations=st.lists(
        st.tuples(TAPE_TIMES, ORDER_IDS, st.sampled_from(["Bid", "Ask"]), TRADE_PRICES),
        max_size=12,
    ),
    trades=st.lists(st.tuples(TAPE_TIMES, TRADE_PRICES), max_size=12),
)
@MODERATE
def test_a_parsed_tape_never_contains_a_cancellation(cancellations, trades):
    """The original notebook read column two of every row, cancellations included.

    A CAN row's column two is the cancelled order's id, so that bug turned
    arbitrary order ids into transaction prices. No mixture of rows, in any
    order, may recover that behaviour here.
    """

    cancel_rows = [
        f"CAN, {time:.6f}, {qid}, {side}, {price}" for time, qid, side, price in cancellations
    ]
    trade_rows = [f"TRD, {time:.6f}, {price}" for time, price in trades]

    rows: list[str] = []
    for index in range(max(len(cancel_rows), len(trade_rows))):
        if index < len(cancel_rows):
            rows.append(cancel_rows[index])
        if index < len(trade_rows):
            rows.append(trade_rows[index])

    with tempfile.TemporaryDirectory(prefix="tape-") as scratch:
        path = Path(scratch) / "mixed_tape.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        times, prices = parse_tape(path)

    assert times.size == prices.size == len(trades)
    np.testing.assert_allclose(prices, [float(price) for _, price in trades])
    assert not set(prices.tolist()) & {float(qid) for _, qid, _, _ in cancellations}


def test_the_cleanliness_guard_from_conftest_applies_to_this_module(request):
    """No need to rescan for stray CSVs: conftest already does it after every test.

    `_no_stray_files` is autouse, so it wraps the simulated sessions above as
    well. This asserts that it is in fact active here rather than duplicating
    the check, so that removing the fixture fails a test in this file too.
    """

    assert "_no_stray_files" in request.fixturenames
    run_session(fixed_schedule(0.0, 60.0, interval=30.0), all_zip(4), seed=1, end_time=60.0)


# --- convergence ---------------------------------------------------------------


def price_lists(min_size: int = 1, max_size: int = 30):
    return st.lists(
        st.floats(
            min_value=1.0,
            max_value=10_000.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
        min_size=min_size,
        max_size=max_size,
    )


EQUILIBRIA = st.floats(
    min_value=1.0,
    max_value=10_000.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


@given(prices=price_lists(), equilibrium=EQUILIBRIA)
@CHEAP
def test_alpha_is_non_negative_and_finite_for_any_non_empty_trade_set(prices, equilibrium):
    alpha = smiths_alpha(prices, equilibrium)
    assert math.isfinite(alpha)
    assert alpha >= 0.0


@given(
    equilibrium=EQUILIBRIA,
    count=st.integers(min_value=1, max_value=30),
    prices=price_lists(),
)
@CHEAP
def test_alpha_is_zero_exactly_when_every_trade_was_at_equilibrium(equilibrium, count, prices):
    assert smiths_alpha([equilibrium] * count, equilibrium) == 0.0

    off_equilibrium = any(price != equilibrium for price in prices)
    assert (smiths_alpha(prices, equilibrium) > 0.0) == off_equilibrium


@given(prices=price_lists(), equilibrium=EQUILIBRIA, seed=st.integers(min_value=0, max_value=1_000))
@CHEAP
def test_alpha_does_not_depend_on_the_order_of_the_trades(prices, equilibrium, seed):
    shuffled = list(prices)
    np.random.default_rng(seed).shuffle(shuffled)
    assert smiths_alpha(shuffled, equilibrium) == pytest.approx(
        smiths_alpha(prices, equilibrium), rel=1e-12
    )


@given(
    prices=price_lists(min_size=1),
    equilibrium=EQUILIBRIA,
    factor=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_subnormal=False),
)
@CHEAP
def test_alpha_is_invariant_under_a_common_rescaling(prices, equilibrium, factor):
    """Alpha is an RMS deviation divided by the equilibrium, so it is a pure ratio.

    Scaling every price and the equilibrium by the same positive factor is a
    change of currency unit, and a convergence measure that moved under one
    would not be comparable across the price ranges these experiments use.
    """

    scaled = smiths_alpha([price * factor for price in prices], equilibrium * factor)
    assert scaled == pytest.approx(smiths_alpha(prices, equilibrium), rel=1e-9, abs=1e-9)


# --- the hypothesis testing pipeline -------------------------------------------


@given(frame=frames())
@MODERATE
def test_a_parametric_omnibus_test_is_never_run_after_normality_is_rejected(frame):
    """The invariant the whole module exists for.

    The notebooks ran Shapiro-Wilk and then ran ANOVA whatever it said. Here, if
    any condition fails the corrected normality test, the omnibus test must be
    the rank-based one.
    """

    result = analyse(frame, name="generated")
    if not result.all_normal:
        assert result.omnibus.test == "Kruskal-Wallis H"
        assert "ANOVA" not in result.omnibus.test
    else:
        assert "ANOVA" in result.omnibus.test


@given(frame=frames())
@MODERATE
def test_every_reported_p_value_is_a_probability_or_explicitly_nan(frame):
    result = analyse(frame, name="generated")

    reported = [result.omnibus.p_value, result.variance.p_value]
    for normality in result.normality:
        reported += [normality.p_value, normality.p_value_corrected]
    for comparison in result.post_hoc.comparisons:
        reported.append(float(comparison["p_value"]))

    for p_value in reported:
        assert math.isnan(p_value) or 0.0 <= p_value <= 1.0


@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_subnormal=False),
        min_size=1,
        max_size=12,
    )
)
@CHEAP
def test_holm_never_decreases_a_p_value_and_never_exceeds_one(p_values):
    adjusted = holm_correction(p_values)

    assert adjusted.shape == (len(p_values),)
    assert (adjusted <= 1.0).all()
    assert (adjusted >= 0.0).all()
    for raw, corrected in zip(p_values, adjusted, strict=True):
        assert corrected >= raw or corrected == pytest.approx(raw)

    # A correction that reordered the evidence would be worse than none: the
    # smallest raw p-value must still be the smallest corrected one.
    order = np.argsort(np.asarray(p_values))
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


@given(values=measurements(min_size=4, max_size=20))
@MODERATE
def test_a_single_condition_is_always_refused(values):
    with pytest.raises(DegenerateDataError, match="at least 2 conditions"):
        analyse(pd.DataFrame({"only": values}))


@given(
    values=measurements(min_size=4, max_size=20),
    constant=st.floats(
        min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_subnormal=False
    ),
)
@MODERATE
def test_a_constant_condition_is_always_refused(values, constant):
    frame = pd.DataFrame({"varying": values, "flat": [constant] * len(values)})
    with pytest.raises(DegenerateDataError, match="constant"):
        analyse(frame)


@given(
    a=measurements(min_size=1, max_size=2),
    b=measurements(min_size=1, max_size=2),
)
@MODERATE
def test_too_few_observations_is_always_refused(a, b):
    size = min(len(a), len(b))
    frame = pd.DataFrame({"a": a[:size], "b": b[:size]})
    with pytest.raises(DegenerateDataError, match="fewer than the 3"):
        analyse(frame)
