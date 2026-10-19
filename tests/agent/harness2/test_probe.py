"""Probe runner tests; never contact a live provider."""

import ast
from pathlib import Path

from agent.harness2 import probe
from agent.harness2.models import Harness, LlmResult, Status, Usage, behavior_signature


def test_null_runner_returns_skipped_and_zero_usage():
    result = probe.run_probe(Harness.empty(), "task", probe.NullRunner(), task_id="task", label="base")
    assert result.status is Status.skipped
    assert result.error == "probe_disabled"
    assert result.usage == Usage()
    assert result.probe_mode == "none"


def test_fake_runner_ok_and_signature():
    result = probe.run_probe(Harness.empty(), "task", probe.FakeRunner({"base": "answer"}),
                             task_id="task", label="base")
    assert result.status is Status.ok
    assert result.text == "answer"
    assert result.behavior_signature == behavior_signature(Status.ok, "answer")
    assert result.model == result.provider == "fake"


def test_whitespace_only_change_keeps_signature():
    runner = probe.FakeRunner({"base": "answer  \n\n\nnext", "cand-01": "answer\n\nnext"})
    base = probe.run_probe(Harness.empty(), "task", runner, task_id="task", label="base")
    candidate = probe.run_probe(Harness.make("changed"), "task", runner, task_id="task", label="cand-01")
    assert base.behavior_signature == candidate.behavior_signature


def test_text_change_changes_signature():
    runner = probe.FakeRunner({"base": "answer A", "cand-01": "answer B"})
    base = probe.run_probe(Harness.empty(), "task", runner, task_id="task", label="base")
    candidate = probe.run_probe(Harness.make("changed"), "task", runner, task_id="task", label="cand-01")
    assert base.behavior_signature != candidate.behavior_signature


def test_prompt_overflow_skips_without_calling_runner():
    class CountingRunner:
        probe_mode = "fake"

        def __init__(self):
            self.calls = 0

        def run(self, **_kwargs):
            self.calls += 1
            raise AssertionError("must not run")

    runner = CountingRunner()
    result = probe.run_probe(Harness.make("x" * 24001), "task", runner, task_id="task", label="base")
    assert runner.calls == 0
    assert result.status is Status.skipped
    assert result.error == "prompt_overflow"


def test_live_runner_makes_one_call_and_copies_route():
    class Client:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return LlmResult("live answer", "actual-model", "actual-provider", 3, 4, 5)

    client = Client()
    runner = probe.LiveProbeRunner(client, max_tokens=222)
    result = probe.run_probe(Harness.empty(), "same task", runner, task_id="task", label="base")
    assert len(client.requests) == 1
    assert client.requests[0].purpose == "probe"
    assert client.requests[0].user == "same task"
    assert client.requests[0].max_tokens == 222
    assert client.requests[0].temperature == 0.0
    assert result.model == "actual-model"
    assert result.provider == "actual-provider"


def test_live_output_is_sanitized_before_signature():
    class Client:
        def complete(self, _request):
            return LlmResult("SERVICE_API_KEY=unit-test-secret-value", "m", "p", 0, 0, 0)

    result = probe.run_probe(Harness.empty(), "task", probe.LiveProbeRunner(Client()), task_id="task", label="base")
    assert "unit-test-secret-value" not in result.text
    assert result.behavior_signature == behavior_signature(Status.ok, result.text)


def test_live_error_is_recorded_not_raised():
    class Client:
        def complete(self, _request):
            raise RuntimeError("SERVICE_API_KEY=unit-test-secret-value")

    result = probe.run_probe(Harness.empty(), "task", probe.LiveProbeRunner(Client()), task_id="task", label="base")
    assert result.status is Status.error
    assert result.error
    assert "unit-test-secret-value" not in result.error


def test_probe_imports_redact_strict_directly_from_store():
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "agent.harness2.store"]
    assert any(any(alias.name == "redact_strict" for alias in node.names) for node in imports)
