"""A typed wrapper around the vendored BSE market simulator.

Three things about BSE make it awkward to call directly, and all three are fixed
here rather than in the vendored file, which is upstream code held at a checksum.

First, `market_session` writes its results as CSV into the process working
directory and returns nothing. Called from a notebook that means every run
scatters files next to the source; called from a test suite it means runs
interfere with each other. Every session here runs inside a temporary directory
that is deleted once the tape has been read back, so a run leaves the repository
exactly as it found it.

Second, the only randomness BSE has is the `random` module's global state.
Seeding is therefore an action at a distance: the demo notebook seeds once and
the other five runs inherit whatever state was left over, which is why none of
its figures reproduce. Every session here takes an explicit seed, and the
caller's global random state is restored afterwards so that seeding a session
does not silently reseed the rest of the program.

Third, the tape is a headerless CSV whose rows are not all the same shape:
transactions are `TRD, time, price` but cancellations are `CAN, time, qid, side,
price`. Reading column 2 from every row, as the notebook does, silently reads
order IDs as prices the moment cancellations are enabled. Rows are filtered on
the event type instead.
"""

from __future__ import annotations

import csv
import os
import random
import tempfile
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fintech.schedules import OrderSchedule, TraderPopulation
from fintech.vendor.BSE import market_session

#: BSE names its output files after the session id. Only the tape is ever asked
#: for; the other four files cost time to write and are never read back.
TAPE_ONLY = {
    "dump_blotters": False,
    "dump_lobs": False,
    "dump_strats": False,
    "dump_avgbals": False,
    "dump_tape": True,
}

#: The tape's event-type column for a completed transaction.
TRADE_EVENT = "TRD"

# BSE reads and writes the process-wide working directory and the process-wide
# random state, so sessions cannot safely overlap within one process.
_SESSION_LOCK = threading.Lock()


@dataclass(frozen=True, eq=False)
class SessionResult:
    """The transaction price time series from one market session."""

    session_id: str
    seed: int
    start_time: float
    end_time: float
    times: np.ndarray
    prices: np.ndarray

    @property
    def n_transactions(self) -> int:
        return int(self.times.size)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.times, "price": self.prices})

    def between(self, start: float, end: float) -> np.ndarray:
        """Transaction prices with timestamps in [start, end)."""

        mask = (self.times >= start) & (self.times < end)
        return self.prices[mask]


@dataclass(frozen=True, eq=False)
class PooledTape:
    """Transactions from several sessions, kept together with their session index."""

    times: np.ndarray
    prices: np.ndarray
    session: np.ndarray
    n_sessions: int = field(default=0)

    @property
    def n_transactions(self) -> int:
        return int(self.times.size)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"session": self.session, "time": self.times, "price": self.prices})

    def between(self, start: float, end: float) -> np.ndarray:
        mask = (self.times >= start) & (self.times < end)
        return self.prices[mask]


@contextmanager
def _working_directory(path: Path):
    """Run a block with the process working directory moved to `path`."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


@contextmanager
def _seeded(seed: int):
    """Seed the global random module, then hand the caller's state back."""

    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def parse_tape(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a BSE tape file into parallel arrays of times and prices.

    Rows are collected into lists and converted once. The notebook grows the
    arrays with np.append inside the row loop, which reallocates and copies the
    whole array on every row and makes reading a tape quadratic in its length.
    """

    times: list[float] = []
    prices: list[float] = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[0].strip() != TRADE_EVENT:
                continue
            times.append(float(row[1]))
            prices.append(float(row[2]))
    return np.asarray(times, dtype=float), np.asarray(prices, dtype=float)


def run_session(
    schedule: OrderSchedule,
    population: TraderPopulation,
    seed: int,
    start_time: float = 0.0,
    end_time: float = 600.0,
    session_id: str = "session",
    verbose: bool = False,
) -> SessionResult:
    """Run one market session and return its tape, writing nothing to the repo."""

    if end_time <= start_time:
        raise ValueError(f"session must have positive duration, got {start_time}-{end_time}")

    trader_spec = population.as_bse()
    order_schedule = schedule.as_bse()

    with _SESSION_LOCK, tempfile.TemporaryDirectory(prefix="bse-") as scratch:
        scratch_path = Path(scratch)
        with _working_directory(scratch_path), _seeded(seed):
            market_session(
                session_id,
                start_time,
                end_time,
                trader_spec,
                order_schedule,
                dict(TAPE_ONLY),
                verbose,
            )
        tape_path = scratch_path / f"{session_id}_tape.csv"
        if not tape_path.is_file():
            raise RuntimeError(f"BSE wrote no tape for session {session_id!r}")
        times, prices = parse_tape(tape_path)

    return SessionResult(
        session_id=session_id,
        seed=seed,
        start_time=start_time,
        end_time=end_time,
        times=times,
        prices=prices,
    )


def run_sessions(
    schedule: OrderSchedule,
    population: TraderPopulation,
    seed: int,
    n_sessions: int = 10,
    start_time: float = 0.0,
    end_time: float = 600.0,
    session_id: str = "session",
    verbose: bool = False,
) -> list[SessionResult]:
    """Run n independent sessions with seeds derived from one base seed."""

    if n_sessions < 1:
        raise ValueError(f"n_sessions must be at least one, got {n_sessions}")
    return [
        run_session(
            schedule,
            population,
            seed=seed + index,
            start_time=start_time,
            end_time=end_time,
            session_id=f"{session_id}_{index:02d}",
            verbose=verbose,
        )
        for index in range(n_sessions)
    ]


def pool_tapes(results: Sequence[SessionResult] | Iterable[SessionResult]) -> PooledTape:
    """Concatenate several session tapes, keeping track of which run each came from."""

    results = list(results)
    if not results:
        raise ValueError("cannot pool an empty sequence of sessions")
    times = np.concatenate([r.times for r in results]) if results else np.empty(0)
    prices = np.concatenate([r.prices for r in results]) if results else np.empty(0)
    session = np.concatenate(
        [np.full(r.times.size, index, dtype=int) for index, r in enumerate(results)]
    )
    return PooledTape(times=times, prices=prices, session=session, n_sessions=len(results))
