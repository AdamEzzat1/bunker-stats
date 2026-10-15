"""
The bindings release the GIL while their Rust kernels run.

Each case starts a long call in a worker thread, then measures how long the
main thread takes to get through a short ``time.sleep``. Waking from a sleep
needs the GIL back: if the worker held it for the whole call, the main thread
would stay blocked until the call returned, so its "short sleep" would take
about as long as the call itself.

Also checks that results are unchanged when the same call runs from several
threads at once.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

try:
    from bunker_stats import _rs as bsr
except ImportError:  # pragma: no cover
    pytest.skip("bunker_stats extension not built", allow_module_level=True)


RNG = np.random.default_rng(20240607)
X_LONG = RNG.normal(size=4_000_000)
Y_LONG = RNG.normal(size=4_000_000)
X_MAT = RNG.normal(size=(60_000, 60))
X_BOOT = RNG.normal(size=20_000)
X_SORT = RNG.normal(size=3_000_000)

# (label, callable) - each takes long enough to observe (tens of ms or more).
LONG_CALLS = [
    ("bootstrap_mean", lambda: bsr.bootstrap_mean(X_BOOT, 20_000, 1)),
    ("rolling_std_np", lambda: bsr.rolling_std_np(X_LONG, 50)),
    ("rolling_corr_np", lambda: bsr.rolling_corr_np(X_LONG, Y_LONG, 50)),
    ("median_np", lambda: bsr.median_np(X_SORT)),
    ("mad_np", lambda: bsr.mad_np(X_SORT)),
    ("zscore_np", lambda: bsr.zscore_np(X_LONG)),
    ("acf", lambda: bsr.acf(X_LONG, 40)),
    ("corr_matrix_np", lambda: bsr.corr_matrix_np(X_MAT)),
    ("norm_cdf", lambda: bsr.norm_cdf(X_LONG, 0.0, 1.0)),
]


def _sleep_resolution(tick=0.002, n=25):
    """Longest observed gap between wake-ups of an idle ticking loop.

    On Windows the default timer granularity makes ``time.sleep(0.002)`` wake
    only every ~15 ms, so a stall shorter than this is indistinguishable from
    ordinary scheduling.
    """
    gaps = []
    last = time.perf_counter()
    for _ in range(n):
        time.sleep(tick)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    return max(gaps)


_SLEEP_RES = _sleep_resolution()


def _main_thread_stall(call, tick=0.002, repeats=5):
    """Return (longest main-thread stall, call duration) while `call` runs in a worker.

    The main thread timestamps every wake-up from a short sleep. The stall is
    the longest stretch of the call's own [start, end] window with no main
    thread wake-up in it; a call that never lets go of the GIL leaves the whole
    window empty, so its stall equals its duration.
    """
    best = None
    for _ in range(repeats):
        window = {}

        def work():
            window["start"] = time.perf_counter()
            call()
            window["end"] = time.perf_counter()

        ticks = [time.perf_counter()]
        worker = threading.Thread(target=work)
        worker.start()
        while worker.is_alive():
            time.sleep(tick)
            ticks.append(time.perf_counter())
        worker.join()
        ticks.append(time.perf_counter())

        start, end = window["start"], window["end"]
        inside = [start] + [t for t in ticks if start < t < end] + [end]
        stall = max(b - a for a, b in zip(inside, inside[1:]))
        if best is None or stall / (end - start) < best[0] / best[1]:
            best = (stall, end - start)
    return best


@pytest.mark.parametrize("label,call", LONG_CALLS, ids=[c[0] for c in LONG_CALLS])
def test_main_thread_keeps_running_during_a_long_call(label, call):
    if not hasattr(bsr, label):
        pytest.skip(f"{label} not exported")
    stall, duration = _main_thread_stall(call)
    # The call must be several timer ticks long, or a single granularity-sized
    # gap already exceeds any fraction of the duration we could assert on.
    floor = max(0.02, 4 * _SLEEP_RES)
    if duration < floor:
        pytest.skip(
            f"{label} finished in {duration * 1e3:.1f} ms; too fast to observe "
            f"(sleep resolution {_SLEEP_RES * 1e3:.1f} ms)"
        )
    # With the GIL held the stall would be ~= duration. Allow generous
    # scheduling slack: the kernels may be using every core, and one stall of
    # timer-resolution size is expected even when the GIL is free.
    assert stall < max(0.5 * duration, 2 * _SLEEP_RES), (
        f"{label}: main thread stalled {stall * 1e3:.0f} ms of a {duration * 1e3:.0f} ms call"
    )


@pytest.mark.parametrize("label,call", LONG_CALLS, ids=[c[0] for c in LONG_CALLS])
def test_concurrent_calls_match_the_serial_result(label, call):
    if not hasattr(bsr, label):
        pytest.skip(f"{label} not exported")

    def as_bytes(res):
        parts = res if isinstance(res, tuple) else (res,)
        return b"".join(np.asarray(p).tobytes() for p in parts)

    want = as_bytes(call())
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda _: as_bytes(call()), range(6)))
    assert all(g == want for g in got)
