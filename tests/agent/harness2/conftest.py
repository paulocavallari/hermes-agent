"""Hermetic test fixtures for the Harness² package."""

from contextlib import contextmanager

import pytest

from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tests._fixtures.harness2_guard import block_network


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    block_network(monkeypatch)


@pytest.fixture
def home(tmp_path):
    """Return a context manager that temporarily selects a test Hermes home."""

    @contextmanager
    def override():
        token = set_hermes_home_override(tmp_path)
        try:
            yield tmp_path
        finally:
            reset_hermes_home_override(token)

    return override
