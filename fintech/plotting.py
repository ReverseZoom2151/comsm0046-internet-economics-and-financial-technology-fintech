"""Saving figures, safely.

Filenames are sanitised in one place. A scenario title contains a colon, and on
NTFS `name:stream` denotes an alternate data stream: the write succeeds, no
error is raised, and the file left on disk is zero bytes with the real content
in a hidden stream. Path.is_file() returns true for such a path and stat()
reports the stream's size, so the obvious check passes and the figure is gone.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

if not matplotlib.get_backend():  # pragma: no cover
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

#: Characters that are illegal in a Windows filename, plus the ones that are
#: legal but change what the path means.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(title: str, extension: str = "png") -> str:
    """Turn a figure title into a filename that survives every platform."""

    cleaned = _UNSAFE.sub(" ", title)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = cleaned.strip("._") or "figure"
    extension = extension.lstrip(".")
    return f"{cleaned}.{extension}"


def save_figure(fig, title: str, directory, extension: str = "png") -> Path:
    """Save a figure under a sanitised name and confirm it reached the disk."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / safe_filename(title, extension)

    fig.savefig(path, dpi=140, bbox_inches="tight")

    # Listing the directory is the only check that catches an alternate data
    # stream. Path.is_file() and stat() both report success for one.
    if path.name not in {p.name for p in directory.iterdir()}:
        raise OSError(f"figure was not written to {path}")

    return path


def new_figure(width: float = 8.0, height: float = 4.5):
    """A figure and axes with the chrome this project does not want."""

    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
