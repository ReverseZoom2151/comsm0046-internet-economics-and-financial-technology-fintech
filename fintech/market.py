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

The wrapper also asks BSE for its average-balance dump, which the earlier
version of this module switched off. The tape says what prices happened; it says
nothing about who made money, and "which trading algorithm profits" is a question
the tape cannot answer at all. That dump is the only place BSE records it.
"""

from __future__ import annotations

import csv
import math
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

#: BSE names its output files after the session id. The minimum that produces a
#: tape, kept as the reference for what a price-only run costs.
TAPE_ONLY = {
    "dump_blotters": False,
    "dump_lobs": False,
    "dump_strats": False,
    "dump_avgbals": False,
    "dump_tape": True,
}

#: What a session actually asks for. Two of BSE's five dumps are read back here;
#: the other three cost time to write and are never looked at.
TAPE_AND_BALANCES = {**TAPE_ONLY, "dump_avgbals": True}

#: The tape's event-type column for a completed transaction.
TRADE_EVENT = "TRD"

#: `trade_stats` writes four fixed columns and then one group of four columns per
#: trader type present: name, total profit, head count, mean profit per trader.
BALANCE_PREFIX_COLUMNS = 4
BALANCE_GROUP_COLUMNS = 4

# BSE reads and writes the process-wide working directory and the process-wide
# random state, so sessions cannot safely overlap within one process.
_SESSION_LOCK = threading.Lock()


@dataclass(frozen=True)
class StrategyBalance:
    """One trader type's takings at one instant, as BSE records them.

    `total_profit` is the sum of the traders' balances, which start at zero and
    accumulate the profit booked on each completed customer order, so it is
    profit rather than wealth. `n_traders` counts that type across both sides of
    the book, which is why `mean_profit` is the only comparable number when the
    population is not balanced: a type with twice the head count earns roughly
    twice the total while being no better at trading.
    """

    strategy: str
    total_profit: float
    n_traders: int
    mean_profit: float


@dataclass(frozen=True)
class BalanceSnapshot:
    """One row of the average-balance dump: the whole population at one time."""

    time: float
    best_bid: float | None
    best_ask: float | None
    strategies: dict[str, StrategyBalance]

    def mean_profit(self, strategy: str) -> float:
        entry = self.strategies.get(strategy)
        return entry.mean_profit if entry is not None else math.nan


@dataclass(frozen=True, eq=False)
class SessionResult:
    """One market session: what it traded at, and what each strategy earned."""

    session_id: str
    seed: int
    start_time: float
    end_time: float
    times: np.ndarray
    prices: np.ndarray
    balances: tuple[BalanceSnapshot, ...] = ()

    @property
    def n_transactions(self) -> int:
        return int(self.times.size)

    @property
    def final_balances(self) -> dict[str, StrategyBalance]:
        """The last row of the balance dump: profit at the end of the session.

        BSE writes a row after every transaction and one more when the session
        closes, so even a market that never traded gets a final row, of zeros.
        The dict is empty only for a `SessionResult` built without a dump, which
        is what the tests do when they need a result and not a simulation.
        """

        return dict(self.balances[-1].strategies) if self.balances else {}

    @property
    def mean_profit_per_trader(self) -> dict[str, float]:
        """Final profit per trader, by strategy. The fair cross-strategy number."""

        return {name: entry.mean_profit for name, entry in self.final_balances.items()}

    def balance_frame(self) -> pd.DataFrame:
        """The balance dump as a long frame: one row per (time, strategy)."""

        records = [
            {
                "time": snapshot.time,
                "strategy": entry.strategy,
                "total_profit": entry.total_profit,
                "n_traders": entry.n_traders,
                "mean_profit": entry.mean_profit,
            }
            for snapshot in self.balances
            for entry in snapshot.strategies.values()
        ]
        columns = ["time", "strategy", "total_profit", "n_traders", "mean_profit"]
        return pd.DataFrame(records, columns=columns)

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


def _optional_price(field: str) -> float | None:
    """BSE writes the literal `None` when a side of the book is empty."""

    text = field.strip()
    if not text or text == "None":
        return None
    return float(text)


def parse_avg_balance(path: str | os.PathLike[str]) -> tuple[BalanceSnapshot, ...]:
    """Read a BSE `_avg_balance.csv` into snapshots of profit by trader type.

    The row shape is `sess_id, time, best_bid, best_ask` followed by one group of
    four columns per trader type. The number of groups is not fixed: `trade_stats`
    re-derives the set of types on every call, and a type only appears once one of
    its traders exists, so a row cannot be parsed by column index against a header
    that BSE never writes. Groups are read off the end of the prefix instead, and
    a row whose tail is not a whole number of groups is skipped rather than
    silently misaligned.
    """

    snapshots: list[BalanceSnapshot] = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            fields = [field.strip() for field in row]
            while fields and fields[-1] == "":
                fields.pop()
            if len(fields) < BALANCE_PREFIX_COLUMNS:
                continue
            tail = fields[BALANCE_PREFIX_COLUMNS:]
            if len(tail) % BALANCE_GROUP_COLUMNS:
                continue
            strategies: dict[str, StrategyBalance] = {}
            for start in range(0, len(tail), BALANCE_GROUP_COLUMNS):
                name, total, count, mean = tail[start : start + BALANCE_GROUP_COLUMNS]
                strategies[name] = StrategyBalance(
                    strategy=name,
                    total_profit=float(total),
                    n_traders=int(count),
                    mean_profit=float(mean),
                )
            snapshots.append(
                BalanceSnapshot(
                    time=float(fields[1]),
                    best_bid=_optional_price(fields[2]),
                    best_ask=_optional_price(fields[3]),
                    strategies=strategies,
                )
            )
    return tuple(snapshots)


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
                dict(TAPE_AND_BALANCES),
                verbose,
            )
        tape_path = scratch_path / f"{session_id}_tape.csv"
        if not tape_path.is_file():
            raise RuntimeError(f"BSE wrote no tape for session {session_id!r}")
        times, prices = parse_tape(tape_path)
        balance_path = scratch_path / f"{session_id}_avg_balance.csv"
        if not balance_path.is_file():
            raise RuntimeError(f"BSE wrote no balance dump for session {session_id!r}")
        balances = parse_avg_balance(balance_path)

    return SessionResult(
        session_id=session_id,
        seed=seed,
        start_time=start_time,
        end_time=end_time,
        times=times,
        prices=prices,
        balances=balances,
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
