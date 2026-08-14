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


@pytest.mark.slow
def test_a_mixed_population_also_trades():
    result = run_session(
        short_schedule("drip-poisson", 10.0), mixed_population(3), seed=9, end_time=SHORT_END
    )
    assert result.n_transactions > 0
