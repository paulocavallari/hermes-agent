"""Tests for side-effect and subprocess guards."""

import os
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home_override
from tests._fixtures.harness2_guard import block_llm

REPO = Path(__file__).resolve().parents[3]
BOOTSTRAP = REPO / "tests/_fixtures/harness2_bootstrap.py"


def test_import_harness2_is_side_effect_free():
    code = (
        "import sys; import agent.harness2; "
        "assert 'agent.auxiliary_client' not in sys.modules"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=False)
    assert result.returncode == 0


def test_socket_connect_is_blocked(monkeypatch):
    from tests._fixtures.harness2_guard import block_network

    block_network(monkeypatch)
    import socket

    try:
        socket.create_connection(("127.0.0.1", 9))
    except RuntimeError as exc:
        assert str(exc) == "harness2: network disabled"
    else:
        raise AssertionError("socket connection was not blocked")


def test_call_llm_is_blocked(monkeypatch):
    block_llm(monkeypatch)
    from agent.auxiliary_client import call_llm

    try:
        call_llm(task="harness2")
    except RuntimeError as exc:
        assert str(exc) == "harness2: network disabled"
    else:
        raise AssertionError("LLM call was not blocked")


def test_child_guard_blocks_socket_and_aux_import():
    for kind in ("socket", "aux"):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HARNESS2_FORBID_AUX"] = "1"
        env["HARNESS2_GUARD_SELFTEST"] = kind
        result = subprocess.run(
            [sys.executable, "-c", BOOTSTRAP.read_text(encoding="utf-8")],
            cwd=REPO,
            env=env,
            check=False,
        )
        assert result.returncode == 0, f"{kind} guard returned {result.returncode}"


def test_home_fixture_overrides_and_restores(home):
    original = get_hermes_home_override()
    with home() as selected:
        assert get_hermes_home_override() == str(selected)
    assert get_hermes_home_override() == original
