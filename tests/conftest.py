"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _no_api_rate_limit(monkeypatch):
    # The API rate limit (D104) counts every request from the one test client; tests that
    # exercise it set CARDIOLENS_RATE_LIMIT_PER_MIN themselves.
    monkeypatch.setenv("CARDIOLENS_RATE_LIMIT_PER_MIN", "0")
