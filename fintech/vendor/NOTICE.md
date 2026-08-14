# Third party code

## BSE: The Bristol Stock Exchange

`BSE.py` in this directory is **not** part of this project. It is vendored
verbatim from the upstream release and is not modified here.

- Upstream: <https://github.com/davecliff/BristolStockExchange>
- Version: 1.91 (November 2024)
- Copyright (c) 2012-2024, Dave Cliff
- Licence: MIT, reproduced in full in the header of `BSE.py`

It is vendored rather than installed because BSE is distributed as a single
file rather than as a package on PyPI, and the coursework notebook it supports
assumes it sits beside the code that imports it. Pinning a copy here means a
session run today produces what it produced when the results in this repository
were recorded.

**Do not edit `BSE.py`.** Everything this project adds lives in `fintech/`, and
wraps the upstream API rather than changing it. A test asserts the vendored file
still matches the upstream release byte for byte, so an accidental edit fails
the suite.

### The API this project targets

The coursework notebook was written against an earlier BSE, and calls:

```python
market_session(trial_id, start_time, end_time, traders_spec, order_sched,
               tdump, dump_all, verbose)          # 8 arguments
```

Version 1.91 takes seven, and replaces the open file handle and the `dump_all`
boolean with a dictionary of flags:

```python
market_session(sess_id, starttime, endtime, trader_spec, order_schedule,
               dumpfile_flags, sess_vrbs)          # 7 arguments
```

So the notebook cannot run against current BSE at all. `fintech/market.py`
targets the current signature and hides it behind an interface that does not
change when upstream does.
