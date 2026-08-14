"""Guards that make this suite safe and fast to run.

Three things would otherwise make it unpleasant.

Matplotlib would try to open a window, so the backend is forced to Agg before
anything imports pyplot.

The market simulator writes its session data as CSV files into the current
working directory, and it does so unconditionally. A test that forgets to
isolate itself would leave those files in the repository, and the next test
would read a stale one. `_no_stray_files` fails any test that leaves a CSV
behind, which turns a silent mess into a named failure.

TextBlob needs an NLTK tokenizer corpus that is downloaded at first use. A test
suite must not depend on the network, so `nltk_available` reports whether the
corpus is present and lets a test skip cleanly rather than fail with a download
error.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

#: CSV names the market simulator writes into the working directory.
_SESSION_GLOBS = (
    "*_tape.csv",
    "*_avg_balance.csv",
    "*_blotters.csv",
    "*_LOB_frames.csv",
    "*_strats.csv",
    "avg_balance.csv",
)


@pytest.fixture(autouse=True)
def _no_stray_files(request):
    """Fail a test that leaves simulator output in the working directory.

    The wrapper in fintech.market is supposed to run every session inside a
    temporary directory. This is what proves it does, on every test rather than
    on the one test that thought to check.
    """

    if request.node.get_closest_marker("allow_stray_files"):
        yield
        return

    root = Path(os.getcwd())
    before = {p for pattern in _SESSION_GLOBS for p in root.glob(pattern)}

    yield

    after = {p for pattern in _SESSION_GLOBS for p in root.glob(pattern)}
    stray = sorted(p.name for p in after - before)
    if stray:
        for p in after - before:
            p.unlink(missing_ok=True)
        raise AssertionError(
            f"{request.node.nodeid} left simulator output in the working directory: "
            f"{stray}. Sessions must run in a temporary directory."
        )


@pytest.fixture(scope="session")
def nltk_available() -> bool:
    """True when the tokenizer TextBlob needs is already downloaded."""

    try:
        import nltk

        for corpus in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(corpus)
                return True
            except LookupError:
                continue
        return False
    except ImportError:
        return False


@pytest.fixture
def data_dir() -> Path:
    """The course datasets, which live beside the package rather than inside it."""

    return Path(__file__).resolve().parent.parent / "data"
