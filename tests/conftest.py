"""Test configuration and fixtures for vastimage tests."""

import pytest


@pytest.fixture(autouse=True)
def _set_env_vars(monkeypatch):
    """Set environment variables for tests."""
    monkeypatch.setenv("VAST_API_KEY", "test-api-key-for-testing")
