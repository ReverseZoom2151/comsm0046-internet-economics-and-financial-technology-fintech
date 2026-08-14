"""Guard the vendored simulator against edits and against upstream drift.

BSE.py is Dave Cliff's code, copied in verbatim under the MIT licence. Nothing in
this repository may modify it, so the checksum is asserted rather than trusted.
The signature check exists because the demo notebook was written against an
older BSE that took eight arguments; the wrapper targets the seven-argument form,
and an upstream bump that changes it should fail here rather than surface as a
confusing TypeError in the middle of an experiment.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from fintech.vendor import BSE

VENDOR_DIR = Path(BSE.__file__).resolve().parent
SOURCE = VENDOR_DIR / "BSE.py"
CHECKSUM = VENDOR_DIR / "BSE.sha256"

EXPECTED_PARAMETERS = (
    "sess_id",
    "starttime",
    "endtime",
    "trader_spec",
    "order_schedule",
    "dumpfile_flags",
    "sess_vrbs",
)

EXPECTED_DUMP_FLAGS = {
    "dump_blotters",
    "dump_lobs",
    "dump_strats",
    "dump_avgbals",
    "dump_tape",
}


def _recorded_checksum() -> str:
    return CHECKSUM.read_text(encoding="utf-8").split()[0].strip()


def test_vendored_source_matches_recorded_checksum():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == _recorded_checksum(), (
        "fintech/vendor/BSE.py has been modified. It is upstream code and must stay verbatim; "
        "restore it rather than updating the checksum."
    )


def test_market_session_takes_seven_arguments():
    parameters = tuple(inspect.signature(BSE.market_session).parameters)
    assert parameters == EXPECTED_PARAMETERS


def test_market_session_reads_the_dump_flags_this_project_supplies():
    source = inspect.getsource(BSE.market_session)
    for flag in EXPECTED_DUMP_FLAGS:
        assert f"dumpfile_flags['{flag}']" in source, f"BSE no longer reads {flag}"


def test_wrapper_supplies_exactly_the_expected_flags():
    from fintech.market import TAPE_ONLY

    assert set(TAPE_ONLY) == EXPECTED_DUMP_FLAGS
    assert TAPE_ONLY["dump_tape"] is True
    assert not any(value for key, value in TAPE_ONLY.items() if key != "dump_tape")
