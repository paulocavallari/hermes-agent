"""Single-call probe runners with explicit offline and live modes."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from agent.harness2.store import redact_strict

from .harness import render_system_prompt
from .models import (
    Harness,
    LlmClient,
    LlmRequest,
    ProbeResult,
    Status,
    Usage,
    behavior_signature,
)


class NullRunner:
    probe_mode = "none"

    def run(self, *, task_id: str, system: str, task: str, label: str, harness_id: str) -> ProbeResult:
        del system, task
        return _result(task_id, label, Status.skipped, "", "probe_disabled", Usage(), "", "", self.probe_mode, harness_id)


class FakeRunner:
    probe_mode = "fake"

    def __init__(self, outputs: Mapping[str, str] | Callable[..., str], errors: Mapping[str, str] | None = None):
        self.outputs = outputs
        self.errors = dict(errors or {})
        self.calls: list[dict[str, str]] = []

    def run(self, *, task_id: str, system: str, task: str, label: str, harness_id: str) -> ProbeResult:
        self.calls.append({"task_id": task_id, "system": system, "task": task, "label": label, "harness_id": harness_id})
        if label in self.errors:
            error = redact_strict(str(self.errors[label]))[:200]
            return _result(task_id, label, Status.error, "", error, Usage(), "fake", "fake", self.probe_mode, harness_id)
        if callable(self.outputs):
            text = self.outputs(task_id=task_id, system=system, task=task, label=label, harness_id=harness_id)
        else:
            text = self.outputs.get(label, "")
        if not isinstance(text, str):
            return _result(task_id, label, Status.error, "", "invalid_runner_output", Usage(), "fake", "fake", self.probe_mode, harness_id)
        text = redact_strict(text)
        return _result(task_id, label, Status.ok, text, None, Usage(), "fake", "fake", self.probe_mode, harness_id)


class LiveProbeRunner:
    probe_mode = "live"

    def __init__(self, client: LlmClient, max_tokens: int = 512):
        self.client = client
        self.max_tokens = max_tokens

    def run(self, *, task_id: str, system: str, task: str, label: str, harness_id: str) -> ProbeResult:
        try:
            response = self.client.complete(LlmRequest(
                purpose="probe", system=system, user=task, max_tokens=self.max_tokens, temperature=0.0
            ))
            text = redact_strict(response.text)
            usage = Usage(response.input_tokens, response.output_tokens, response.input_tokens + response.output_tokens)
            return _result(task_id, label, Status.ok, text, None, usage, response.model, response.provider,
                           self.probe_mode, harness_id)
        except Exception as exc:
            error = redact_strict(str(exc))[:200]
            return _result(task_id, label, Status.error, "", error, Usage(), "", "", self.probe_mode, harness_id)


def _result(task_id: str, label: str, status: Status, text: str, error: str | None, usage: Usage,
            model: str, provider: str, probe_mode: str, harness_id: str, wall_ms: int = 0) -> ProbeResult:
    return ProbeResult(
        task_id=task_id,
        label=label,
        status=status,
        text=text,
        error=error,
        usage=usage,
        model=model,
        provider=provider,
        behavior_signature=behavior_signature(status, text),
        probe_mode=probe_mode,
        harness_id=harness_id,
        wall_ms=wall_ms,
    )


def run_probe(
    harness: Harness,
    task: str,
    runner,
    *,
    task_id: str,
    label: str,
    monotonic: Callable[[], float] = time.monotonic,
) -> ProbeResult:
    system = render_system_prompt(harness)
    if system is None:
        return _result(task_id, label, Status.skipped, "", "prompt_overflow", Usage(), "", "",
                       runner.probe_mode, harness.harness_id)
    if runner.probe_mode == "none":
        return _result(task_id, label, Status.skipped, "", "probe_disabled", Usage(), "", "",
                       runner.probe_mode, harness.harness_id)
    started = monotonic()
    try:
        observed = runner.run(task_id=task_id, system=system, task=task, label=label, harness_id=harness.harness_id)
        status = observed.status if isinstance(observed.status, Status) else Status(observed.status)
        text = redact_strict(observed.text)
        error = redact_strict(observed.error)[:200] if observed.error else None
        usage = observed.usage if isinstance(observed.usage, Usage) else Usage.from_dict(observed.usage)
        model = str(observed.model or "")
        provider = str(observed.provider or "")
    except Exception as exc:
        status = Status.error
        text = ""
        error = redact_strict(str(exc))[:200]
        usage = Usage()
        model = provider = ""
    elapsed = max(0, int((monotonic() - started) * 1000))
    return _result(task_id, label, status, text, error, usage, model, provider, runner.probe_mode,
                   harness.harness_id, elapsed)
