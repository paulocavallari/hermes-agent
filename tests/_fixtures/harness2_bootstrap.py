"""Child-process network and optional auxiliary-client guard for Harness²."""

import json
import os
import runpy
import socket
import sys


def _deny(*_args, **_kwargs):
    raise RuntimeError("harness2: network disabled")


socket.create_connection = socket.getaddrinfo = _deny
socket.socket.connect = socket.socket.connect_ex = _deny


class _BlockAux:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[:2] == ["agent", "auxiliary_client"]:
            raise RuntimeError("harness2: auxiliary_client imported")


if os.environ.get("HARNESS2_FORBID_AUX") == "1":
    sys.meta_path.insert(0, _BlockAux())

selftest = os.environ.get("HARNESS2_GUARD_SELFTEST")
if selftest == "socket":
    try:
        socket.create_connection(("127.0.0.1", 9))
    except RuntimeError:
        sys.exit(0)
    sys.exit(3)
if selftest == "aux":
    try:
        __import__("agent.auxiliary_client")
    except RuntimeError:
        sys.exit(0)
    sys.exit(3)

sys.argv = ["hermes", *json.loads(os.environ["HARNESS2_ARGV"])]
runpy.run_module("hermes_cli.main", run_name="__main__")
