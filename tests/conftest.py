from __future__ import annotations

import os
import pytest

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--bs-debug", action="store_true", help="Enable bunker-stats debug printing")
    parser.addoption("--bs-seed", type=int, default=0, help="Seed for bunker-stats tests")
    parser.addoption(
        "--strict-skips",
        action="store_true",
        help="Fail the run if any test is skipped (for CI, where every optional "
        "dependency is installed and a skip means a test silently stopped running)",
    )


# A skipped test is indistinguishable from a passing one in a green build.
# Four whole files once skipped on every run, locally and in CI, because they
# imported a module name that no longer existed. With --strict-skips any skip
# fails the session and is listed, so that cannot happen quietly again.
_SKIPPED: list = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and not hasattr(report, "wasxfail"):
        _SKIPPED.append((report.nodeid, _skip_reason(report)))


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:  # module-level importorskip / pytest.skip
        _SKIPPED.append((report.nodeid, _skip_reason(report)))


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if session.config.getoption("--strict-skips") and _SKIPPED and exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if config.getoption("--strict-skips") and _SKIPPED:
        terminalreporter.section("skipped tests are failures under --strict-skips", red=True)
        for nodeid, reason in _SKIPPED:
            terminalreporter.line(f"SKIPPED {nodeid}: {reason}")

@pytest.fixture(scope="session")
def bs_debug(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--bs-debug"))

@pytest.fixture(scope="session")
def bs_seed(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bs-seed"))

def pytest_configure(config: pytest.Config) -> None:
    # optional: rich tracebacks if installed
    try:
        from rich.traceback import install
        install(show_locals=True)
    except Exception:
        pass
