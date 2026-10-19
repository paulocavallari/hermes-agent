"""Hermetic guard helpers for Harness² tests."""

import socket


def _deny(*_args, **_kwargs):
    raise RuntimeError("harness2: network disabled")


def block_network(monkeypatch):
    """Make outbound socket operations fail explicitly in this process."""
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)
    monkeypatch.setattr(socket.socket, "connect", _deny)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny)


def block_llm(monkeypatch):
    """Replace the real auxiliary LLM call with an explicit failure."""
    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "call_llm", _deny)
