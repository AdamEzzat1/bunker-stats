from __future__ import annotations

import os
import pytest

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--bs-debug", action="store_true", help="Enable bunker-stats debug printing")
    parser.addoption("--bs-seed", type=int, default=0, help="Seed for bunker-stats tests")

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
