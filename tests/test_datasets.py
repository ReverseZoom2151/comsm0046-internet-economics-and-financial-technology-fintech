"""The datasets load by name, from anywhere, with the shape we claim they have."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from fintech.datasets import (
    DATA_DIR,
    available_datasets,
    dataset_spec,
    load_all,
    load_dataset,
)


def test_both_datasets_are_available():
    assert available_datasets() == ("data1", "data2")


@pytest.mark.parametrize(
    ("name", "conditions", "rows"),
    [("data1", ("a", "b", "c"), 15), ("data2", ("x", "y"), 20)],
)
def test_shapes_are_as_documented(name, conditions, rows):
    frame = load_dataset(name)
    assert tuple(frame.columns) == conditions
    assert len(frame) == rows
    assert dataset_spec(name).n_conditions == len(conditions)


def test_the_datasets_have_different_widths():
    """The reason the pipeline must not hardcode three columns."""

    frames = load_all()
    assert frames["data1"].shape[1] != frames["data2"].shape[1]


def test_values_are_numeric_and_complete():
    for frame in load_all().values():
        assert not frame.isna().to_numpy().any()
        assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)


def test_loading_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """The notebooks used a bare relative path and broke outside one directory."""

    monkeypatch.chdir(tmp_path)
    frame = load_dataset("data1")
    assert len(frame) == 15
    assert DATA_DIR.is_absolute()
    assert os.getcwd() == str(tmp_path)


def test_unknown_dataset_is_rejected_by_name():
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("data3")


def test_missing_file_reports_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="data1"):
        load_dataset("data1", data_dir=tmp_path)


def test_validation_catches_a_reshaped_csv(tmp_path):
    (tmp_path / "data1.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conditions"):
        load_dataset("data1", data_dir=tmp_path)

    frame = load_dataset("data1", data_dir=tmp_path, validate=False)
    assert tuple(frame.columns) == ("a", "b")
