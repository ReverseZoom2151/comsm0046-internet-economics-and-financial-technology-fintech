"""Loading the week 5 activity datasets by name, with their shapes written down.

The original notebooks read the data with a bare relative path,
`pd.read_csv('w5.data/data1.csv')`, which only resolves if the interpreter
happens to be started in the directory above the data. Worse, the choice of
dataset was made by commenting one line out and uncommenting another, which is
why two notebooks exist that are byte-identical in 24 of their 27 cells.

Selecting data is a parameter, not an edit. This module resolves the path
relative to the package so it works from anywhere, and records the shape of
each dataset so that downstream code does not quietly assume three columns.
The two datasets have different widths (data1 has three conditions, data2 has
two) and that difference is the whole point of the activity: with two groups a
one-way ANOVA is just an independent t-test, and the correct omnibus test
depends on the data rather than on which line was uncommented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: The repository's data directory, resolved from this file rather than from
#: the process working directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """What we expect a dataset to look like once loaded."""

    name: str
    filename: str
    conditions: tuple[str, ...]
    n_per_condition: int
    description: str

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)


#: The datasets shipped with the week 5 activity. Each column is one condition
#: and each row is one independent observation under every condition.
DATASETS: dict[str, DatasetSpec] = {
    "data1": DatasetSpec(
        name="data1",
        filename="data1.csv",
        conditions=("a", "b", "c"),
        n_per_condition=15,
        description="Three conditions, 15 observations each. Roughly normal.",
    ),
    "data2": DatasetSpec(
        name="data2",
        filename="data2.csv",
        conditions=("x", "y"),
        n_per_condition=20,
        description="Two conditions, 20 observations each. Neither is normal.",
    ),
}


def available_datasets() -> tuple[str, ...]:
    """Names accepted by :func:`load_dataset`, in a stable order."""

    return tuple(DATASETS)


def dataset_spec(name: str) -> DatasetSpec:
    """The recorded shape of a dataset, without reading it from disk."""

    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(available_datasets())
        raise KeyError(f"unknown dataset {name!r}; available datasets are {known}") from None


def load_dataset(
    name: str, data_dir: Path | str | None = None, validate: bool = True
) -> pd.DataFrame:
    """Read one activity dataset into a dataframe of one column per condition.

    Passing ``validate=False`` skips the shape check, which is useful if the
    CSV files are ever regenerated with a different number of observations.
    """

    spec = dataset_spec(name)
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    path = directory / spec.filename
    if not path.is_file():
        raise FileNotFoundError(f"dataset {name!r} is not at {path}")

    frame = pd.read_csv(path)

    if validate:
        columns = tuple(frame.columns)
        if columns != spec.conditions:
            raise ValueError(
                f"dataset {name!r} has conditions {columns}, expected {spec.conditions}"
            )
        if len(frame) != spec.n_per_condition:
            raise ValueError(
                f"dataset {name!r} has {len(frame)} rows, expected {spec.n_per_condition}"
            )

    return frame


def load_all(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Every activity dataset, keyed by name."""

    return {name: load_dataset(name, data_dir=data_dir) for name in available_datasets()}
