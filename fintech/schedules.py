"""Supply and demand schedules as validated data rather than dicts built inline.

The demo notebook builds BSE's nested dict-of-lists-of-dicts by hand in six
separate cells, so a typo in a key name is only discovered when the simulator
raises a KeyError several minutes into a run, and nothing checks that the two
sides of the market are even comparable. Building the dicts from frozen
dataclasses moves those mistakes to construction time.

The equilibrium helpers live here because the equilibrium price is a property of
the schedule and the trader counts, not of any particular session. BSE assigns
limit prices deterministically under `stepmode='fixed'`, so the theoretical
equilibrium can be derived rather than assumed, and every convergence measure in
fintech.smith is quoted against a number computed the same way the simulator
computes the assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STEP_MODES = frozenset({"fixed", "jittered", "random"})
TIME_MODES = frozenset({"periodic", "drip-fixed", "drip-jitter", "drip-poisson"})

#: The symmetric range from Chart 1 of Smith (1962): eleven units a side, prices
#: from 80 to 320 in steps of 20.
CHART1_RANGE_LOW = 80
CHART1_RANGE_HIGH = 320

#: The jump the notebook's final cell applies half way through the session.
SHOCK_RANGE_LOW = 300
SHOCK_RANGE_HIGH = 400


@dataclass(frozen=True)
class PriceRange:
    """The lowest and highest limit price a schedule assigns."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError(f"limit prices must be positive, got {self.low}-{self.high}")
        if self.low > self.high:
            raise ValueError(f"low must not exceed high, got {self.low}-{self.high}")

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def as_bse(self) -> tuple[int, int]:
        return (self.low, self.high)


CHART1_RANGE = PriceRange(CHART1_RANGE_LOW, CHART1_RANGE_HIGH)
SHOCK_RANGE = PriceRange(SHOCK_RANGE_LOW, SHOCK_RANGE_HIGH)


@dataclass(frozen=True)
class ScheduleStep:
    """One time-slice of a supply or demand schedule."""

    start: float
    end: float
    price_range: PriceRange
    stepmode: str = "fixed"

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"step must have positive duration, got {self.start}-{self.end}")
        if self.stepmode not in STEP_MODES:
            raise ValueError(
                f"unknown stepmode {self.stepmode!r}, expected one of {sorted(STEP_MODES)}"
            )

    def covers(self, time: float) -> bool:
        return self.start <= time < self.end

    def as_bse(self) -> dict[str, Any]:
        return {
            "from": self.start,
            "to": self.end,
            "ranges": [self.price_range.as_bse()],
            "stepmode": self.stepmode,
        }


@dataclass(frozen=True)
class OrderSchedule:
    """The customer orders fed to the market: both sides, plus their arrival mode."""

    supply: tuple[ScheduleStep, ...]
    demand: tuple[ScheduleStep, ...]
    interval: float
    timemode: str = "periodic"

    def __post_init__(self) -> None:
        if not self.supply or not self.demand:
            raise ValueError("both a supply and a demand schedule are required")
        if self.interval <= 0:
            raise ValueError(f"order interval must be positive, got {self.interval}")
        if self.timemode not in TIME_MODES:
            raise ValueError(
                f"unknown timemode {self.timemode!r}, expected one of {sorted(TIME_MODES)}"
            )
        for side in (self.supply, self.demand):
            _check_contiguous(side)

    @property
    def start(self) -> float:
        return min(self.supply[0].start, self.demand[0].start)

    @property
    def end(self) -> float:
        return max(self.supply[-1].end, self.demand[-1].end)

    def step_at(self, time: float) -> tuple[ScheduleStep, ScheduleStep]:
        """The supply and demand steps in force at `time`.

        BSE clamps to the last step once the session runs past the schedule, so
        this does the same rather than raising.
        """

        return (_step_at(self.supply, time), _step_at(self.demand, time))

    def as_bse(self) -> dict[str, Any]:
        return {
            "sup": [step.as_bse() for step in self.supply],
            "dem": [step.as_bse() for step in self.demand],
            "interval": self.interval,
            "timemode": self.timemode,
        }


@dataclass(frozen=True)
class TraderPopulation:
    """How many traders of each type sit on each side of the book."""

    sellers: tuple[tuple[str, int], ...]
    buyers: tuple[tuple[str, int], ...]
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for side, name in ((self.sellers, "sellers"), (self.buyers, "buyers")):
            if not side:
                raise ValueError(f"{name} must contain at least one trader type")
            for ttype, count in side:
                if count <= 0:
                    raise ValueError(f"{name} entry {ttype!r} must have a positive count")

    @property
    def n_sellers(self) -> int:
        return sum(count for _, count in self.sellers)

    @property
    def n_buyers(self) -> int:
        return sum(count for _, count in self.buyers)

    @property
    def size(self) -> int:
        return self.n_sellers + self.n_buyers

    def as_bse(self) -> dict[str, list[tuple[str, int]]]:
        return {
            "sellers": [tuple(entry) for entry in self.sellers],
            "buyers": [tuple(entry) for entry in self.buyers],
        }


def _check_contiguous(steps: tuple[ScheduleStep, ...]) -> None:
    for earlier, later in zip(steps, steps[1:], strict=False):
        if later.start < earlier.end:
            raise ValueError(
                f"schedule steps overlap: {earlier.start}-{earlier.end} then "
                f"{later.start}-{later.end}"
            )


def _step_at(steps: tuple[ScheduleStep, ...], time: float) -> ScheduleStep:
    for step in steps:
        if step.covers(time):
            return step
    if time < steps[0].start:
        return steps[0]
    return steps[-1]


def fixed_schedule(
    start: float = 0.0,
    end: float = 600.0,
    interval: float = 60.0,
    timemode: str = "periodic",
    price_range: PriceRange = CHART1_RANGE,
) -> OrderSchedule:
    """The single unchanging schedule of Smith's Chart 1."""

    supply = (ScheduleStep(start, end, price_range),)
    demand = (ScheduleStep(start, end, price_range),)
    return OrderSchedule(supply=supply, demand=demand, interval=interval, timemode=timemode)


def shock_schedule(
    start: float = 0.0,
    end: float = 600.0,
    interval: float = 10.0,
    timemode: str = "drip-poisson",
    before: PriceRange = CHART1_RANGE,
    after: PriceRange = SHOCK_RANGE,
    shock_time: float | None = None,
) -> OrderSchedule:
    """A schedule that jumps to a new price range part way through the session."""

    if shock_time is None:
        shock_time = start + (end - start) / 2.0
    if not start < shock_time < end:
        raise ValueError(f"shock_time {shock_time} must fall inside {start}-{end}")
    steps = (
        ScheduleStep(start, shock_time, before),
        ScheduleStep(shock_time, end, after),
    )
    return OrderSchedule(supply=steps, demand=steps, interval=interval, timemode=timemode)


def all_zip(n_per_side: int = 11) -> TraderPopulation:
    """The homogeneous ZIP population the notebook starts from."""

    side = (("ZIP", n_per_side),)
    return TraderPopulation(sellers=side, buyers=side, label=f"all-ZIP x{n_per_side}")


def mixed_population(n_each: int = 10) -> TraderPopulation:
    """ZIP, ZIC, SHVR and GVWY in equal numbers on both sides.

    The mix matters: a market of nothing but ZIPs can stall, because every ZIP
    waits for someone else to move the book. Noise traders keep it alive.
    """

    side = (("ZIP", n_each), ("ZIC", n_each), ("SHVR", n_each), ("GVWY", n_each))
    return TraderPopulation(sellers=side, buyers=side, label=f"mixed x{n_each} each")


def limit_prices(price_range: PriceRange, n_traders: int) -> list[int]:
    """The limit prices BSE hands out under `stepmode='fixed'`.

    Mirrors getorderprice() in the vendored simulator: price i is
    pmin + int(i * (pmax - pmin) / (n - 1)).
    """

    if n_traders < 2:
        raise ValueError(f"need at least two traders a side, got {n_traders}")
    span = price_range.high - price_range.low
    step = span / (n_traders - 1)
    return [price_range.low + int(i * step) for i in range(n_traders)]


def equilibrium_price(
    supply_range: PriceRange,
    demand_range: PriceRange,
    n_sellers: int,
    n_buyers: int,
) -> float:
    """Where the step supply and demand curves cross.

    Sellers are sorted cheapest first and buyers dearest first; the market
    clears at the last index where a buyer still outbids the matching seller.
    The clearing interval is returned as its midpoint, which for the symmetric
    Chart 1 schedule collapses to a single price.
    """

    supply = sorted(limit_prices(supply_range, n_sellers))
    demand = sorted(limit_prices(demand_range, n_buyers), reverse=True)

    k = 0
    for i in range(min(len(supply), len(demand))):
        if demand[i] >= supply[i]:
            k = i + 1
        else:
            break
    if k == 0:
        raise ValueError("supply and demand schedules do not overlap: no trade is possible")

    lower = max(supply[k - 1], demand[k] if k < len(demand) else supply[k - 1])
    upper = min(demand[k - 1], supply[k] if k < len(supply) else demand[k - 1])
    if upper < lower:
        lower = upper = (supply[k - 1] + demand[k - 1]) / 2.0
    return (lower + upper) / 2.0


def equilibrium_at(
    schedule: OrderSchedule, population: TraderPopulation, time: float = 0.0
) -> float:
    """The theoretical equilibrium price in force at `time`."""

    supply_step, demand_step = schedule.step_at(time)
    return equilibrium_price(
        supply_step.price_range,
        demand_step.price_range,
        population.n_sellers,
        population.n_buyers,
    )
