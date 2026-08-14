"""The wrapper's job is reproducibility and leaving no mess, so both are asserted.

Sessions here are short and thinly populated on purpose: the point is the
plumbing, not the economics. The economics is measured in test_smith.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from fintech.market import (
    PooledTape,
    parse_avg_balance,
    parse_tape,
    pool_tapes,
    run_session,
    run_sessions,
)
from fintech.schedules import all_zip, fixed_schedule, mixed_population

SHORT_END = 120.0

CSV_PATTERNS = (
    "*_tape.csv",
    "*_avg_balance.csv",
    "*_blotters.csv",
    "*_LOB_frames.csv",
    "*_strats.csv",
    "avg_balance.csv",
)


def short_schedule(timemode: str = "periodic", interval: float = 30.0):
    return fixed_schedule(0.0, SHORT_END, interval=interval, timemode=timemode)


def csv_files(directory: Path) -> set[str]:
    found: set[str] = set()
    for pattern in CSV_PATTERNS:
        found.update(path.name for path in directory.glob(pattern))
    return found


def test_session_returns_a_tape_with_plausible_prices():
    result = run_session(short_schedule(), all_zip(6), seed=1, end_time=SHORT_END)

    assert result.n_transactions > 0
    assert result.times.dtype == float
    assert result.prices.dtype == float
    assert result.times.size == result.prices.size
    assert result.times.min() >= 0.0
    assert result.times.max() <= SHORT_END
    # Every trade must sit inside the schedule's price range.
    assert result.prices.min() >= 80
    assert result.prices.max() <= 320


def test_session_is_reproducible_from_its_seed():
    first = run_session(short_schedule(), all_zip(6), seed=7, end_time=SHORT_END)
    second = run_session(short_schedule(), all_zip(6), seed=7, end_time=SHORT_END)

    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.prices, second.prices)


def test_different_seeds_give_different_sessions():
    first = run_session(short_schedule(), all_zip(6), seed=7, end_time=SHORT_END)
    other = run_session(short_schedule(), all_zip(6), seed=8, end_time=SHORT_END)

    same_length = first.prices.size == other.prices.size
    assert not (same_length and np.array_equal(first.prices, other.prices))


def test_session_does_not_disturb_the_callers_random_state():
    random.seed(4242)
    expected = [random.random() for _ in range(3)]

    random.seed(4242)
    run_session(short_schedule(), all_zip(6), seed=1, end_time=SHORT_END)
    after = [random.random() for _ in range(3)]

    assert after == expected


def test_no_csv_files_are_left_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))

    run_sessions(short_schedule(), all_zip(6), seed=3, n_sessions=2, end_time=SHORT_END)

    assert csv_files(tmp_path) == set()
    assert set(os.listdir(tmp_path)) == before


def test_working_directory_is_restored_even_when_the_session_fails():
    before = Path.cwd()
    with pytest.raises(ValueError):
        run_session(short_schedule(), all_zip(6), seed=1, start_time=100.0, end_time=10.0)
    assert Path.cwd() == before


def test_repository_root_stays_clean_after_a_run():
    root = Path(__file__).resolve().parent.parent
    before = csv_files(root)
    run_session(short_schedule(), all_zip(6), seed=11, end_time=SHORT_END)
    assert csv_files(root) == before


def test_run_sessions_uses_consecutive_seeds():
    results = run_sessions(short_schedule(), all_zip(6), seed=50, n_sessions=3, end_time=SHORT_END)

    assert [result.seed for result in results] == [50, 51, 52]
    assert len({result.session_id for result in results}) == 3


def test_run_sessions_rejects_a_non_positive_count():
    with pytest.raises(ValueError, match="at least one"):
        run_sessions(short_schedule(), all_zip(6), seed=1, n_sessions=0, end_time=SHORT_END)


def test_pooling_concatenates_tapes_and_labels_their_session():
    results = run_sessions(short_schedule(), all_zip(6), seed=60, n_sessions=3, end_time=SHORT_END)
    pooled = pool_tapes(results)

    assert isinstance(pooled, PooledTape)
    assert pooled.n_sessions == 3
    assert pooled.n_transactions == sum(result.n_transactions for result in results)
    assert set(np.unique(pooled.session)) <= {0, 1, 2}

    frame = pooled.to_frame()
    assert list(frame.columns) == ["session", "time", "price"]
    assert len(frame) == pooled.n_transactions


def test_pooling_nothing_is_an_error():
    with pytest.raises(ValueError, match="empty sequence"):
        pool_tapes([])


def test_between_selects_a_half_open_window():
    result = run_session(short_schedule(), all_zip(6), seed=5, end_time=SHORT_END)
    early = result.between(0.0, 60.0)
    late = result.between(60.0, SHORT_END)

    assert early.size + late.size == result.n_transactions


def test_session_frame_has_the_expected_columns():
    result = run_session(short_schedule(), all_zip(6), seed=2, end_time=SHORT_END)
    frame = result.to_frame()

    assert list(frame.columns) == ["time", "price"]
    assert len(frame) == result.n_transactions


def test_parse_tape_ignores_cancellations_and_malformed_rows(tmp_path):
    tape = tmp_path / "example_tape.csv"
    tape.write_text(
        "TRD, 10.000000, 200\nCAN, 11.000000, 42, Bid, 137\n\nTRD, 12.500000, 205\n",
        encoding="utf-8",
    )

    times, prices = parse_tape(tape)

    # Reading column two from every row would have taken the cancelled order's
    # id, 42, as a price.
    np.testing.assert_array_equal(times, np.array([10.0, 12.5]))
    np.testing.assert_array_equal(prices, np.array([200.0, 205.0]))


def test_parse_tape_of_an_empty_file_gives_empty_arrays(tmp_path):
    tape = tmp_path / "empty_tape.csv"
    tape.write_text("", encoding="utf-8")

    times, prices = parse_tape(tape)

    assert times.size == 0
    assert prices.size == 0
    assert times.dtype == float


def test_session_rejects_a_backwards_clock():
    with pytest.raises(ValueError, match="positive duration"):
        run_session(short_schedule(), all_zip(6), seed=1, start_time=60.0, end_time=60.0)


def test_parse_avg_balance_reads_the_repeating_four_column_groups(tmp_path):
    dump = tmp_path / "example_avg_balance.csv"
    dump.write_text(
        "prof, 000180, 193, 259, GVWY, 332, 10, 33.200000, SHVR, 567, 10, 56.700000, "
        "ZIC, 855, 10, 85.500000, ZIP, 582, 10, 58.200000, \n",
        encoding="utf-8",
    )

    (snapshot,) = parse_avg_balance(dump)

    assert snapshot.time == 180.0
    assert (snapshot.best_bid, snapshot.best_ask) == (193.0, 259.0)
    assert sorted(snapshot.strategies) == ["GVWY", "SHVR", "ZIC", "ZIP"]
    assert snapshot.strategies["ZIC"].total_profit == 855.0
    assert snapshot.strategies["ZIC"].n_traders == 10
    assert snapshot.mean_profit("ZIC") == pytest.approx(85.5)


def test_parse_avg_balance_handles_an_empty_side_of_the_book(tmp_path):
    # BSE writes the literal string None when there is no best bid or ask, which
    # float() would raise on.
    dump = tmp_path / "empty_avg_balance.csv"
    dump.write_text("prof, 000001, None, None, ZIP, 0, 6, 0.000000, \n", encoding="utf-8")

    (snapshot,) = parse_avg_balance(dump)

    assert snapshot.best_bid is None
    assert snapshot.best_ask is None
    assert snapshot.strategies["ZIP"].mean_profit == 0.0


def test_parse_avg_balance_skips_rows_it_cannot_align(tmp_path):
    dump = tmp_path / "ragged_avg_balance.csv"
    dump.write_text(
        "prof, 000001, 100, 101, ZIP, 10, 2, 5.000000, \n"
        "prof, 000002\n"
        "prof, 000003, 100, 101, ZIP, 10\n",
        encoding="utf-8",
    )

    snapshots = parse_avg_balance(dump)

    # A row with a partial group would otherwise be read with its columns
    # shifted, which is a wrong number rather than a missing one.
    assert [snapshot.time for snapshot in snapshots] == [1.0]


def test_a_session_reports_profit_per_trader_for_every_strategy_present():
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=13, end_time=SHORT_END
    )

    assert result.balances, "the average-balance dump must be read back"
    profits = result.mean_profit_per_trader
    assert sorted(profits) == ["GVWY", "SHVR", "ZIC", "ZIP"]
    assert all(value >= 0.0 for value in profits.values())
    # Six traders of each type, three a side.
    assert {entry.n_traders for entry in result.final_balances.values()} == {6}


def test_total_profit_is_the_head_count_times_the_mean():
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=14, end_time=SHORT_END
    )

    for entry in result.final_balances.values():
        assert entry.total_profit == pytest.approx(entry.mean_profit * entry.n_traders)


def test_the_balance_frame_is_a_time_series_ending_at_the_final_row():
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=15, end_time=SHORT_END
    )
    frame = result.balance_frame()

    assert list(frame.columns) == [
        "time",
        "strategy",
        "total_profit",
        "n_traders",
        "mean_profit",
    ]
    assert frame["time"].is_monotonic_increasing
    final = frame[frame["time"] == frame["time"].max()]
    assert dict(zip(final["strategy"], final["mean_profit"], strict=True)) == (
        result.mean_profit_per_trader
    )


def test_profit_never_decreases_over_a_session():
    # Balances accumulate booked profit and BSE never debits them, so a falling
    # series would mean the dump was being read wrongly.
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=16, end_time=SHORT_END
    )
    frame = result.balance_frame()

    for _, group in frame.groupby("strategy"):
        assert group["total_profit"].is_monotonic_increasing


@pytest.mark.slow
def test_a_mixed_population_also_trades():
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=9, end_time=SHORT_END
    )
    assert result.n_transactions > 0
