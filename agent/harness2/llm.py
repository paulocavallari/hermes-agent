"""Opt-in LLM transport, strict JSON decoding, and deterministic fakes."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from .models import LlmClient, LlmRequest, LlmResult
from agent.harness2.store import redact_strict


class LlmError(RuntimeError):
    def __init__(self, detail: str):
        self.detail = redact_strict(str(detail))[:200]
        super().__init__(self.detail)


class InvalidPayload(ValueError):
    def __init__(self, detail: str = "invalid JSON object"):
        self.detail = str(detail)[:200]
        super().__init__(self.detail)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, *, max_calls: int = 32, max_tokens: int | None = None):
        if type(max_calls) is not int or max_calls < 0 or (max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0)):
            raise ValueError("invalid budget")
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens = 0

    def reserve(self, tokens: int = 0) -> None:
        if type(tokens) is not int or tokens < 0:
            raise ValueError("invalid token reservation")
        if self.calls + 1 > self.max_calls:
            raise BudgetExceeded("call budget exceeded")
        if self.max_tokens is not None and self.tokens + tokens > self.max_tokens:
            raise BudgetExceeded("token budget exceeded")
        self.calls += 1
        self.tokens += tokens


class AuxLlmClient:
    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        timeout: float = 120.0,
        call: Callable[..., Any] | None = None,
    ):
        self._model = model
        self._provider = provider
        self._timeout = timeout
        if call is None:
            from agent.auxiliary_client import call_llm

            call = call_llm
        self._call = call

    def complete(self, req: LlmRequest) -> LlmResult:
        route: dict[str, str] = {}
        started = time.monotonic()
        try:
            response = self._call(
                task="harness2",
                messages=[
                    {"role": "system", "content": req.system},
                    {"role": "user", "content": req.user},
                ],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                timeout=self._timeout,
                model=self._model,
                provider=self._provider,
                route_info=route,
            )
            text = _response_text(response)
            usage = _response_usage(response)
            latency = int((time.monotonic() - started) * 1000)
            return LlmResult(
                text=text,
                model=route.get("model") or self._model or "",
                provider=route.get("provider") or self._provider or "",
                input_tokens=usage[0],
                output_tokens=usage[1],
                latency_ms=latency,
            )
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(redact_strict(str(exc))[:200]) from None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = _get(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        message = _get(choices[0], "message")
        content = _get(message, "content")
        if isinstance(content, str):
            return content
    content = _get(response, "content")
    if isinstance(content, str):
        return content
    raise LlmError("provider returned no text content")


def _response_usage(response: Any) -> tuple[int, int, int]:
    usage = _get(response, "usage")
    if usage is None:
        return 0, 0, 0
    input_tokens = _get(usage, "input_tokens", _get(usage, "prompt_tokens", 0))
    output_tokens = _get(usage, "output_tokens", _get(usage, "completion_tokens", 0))
    total_tokens = _get(usage, "total_tokens", None)
    values = (input_tokens or 0, output_tokens or 0, total_tokens)
    if any(type(value) is not int or value < 0 for value in values[:2]):
        raise LlmError("provider returned invalid usage")
    total = values[2] if type(values[2]) is int and values[2] >= 0 else values[0] + values[1]
    return values[0], values[1], total


def parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise InvalidPayload("invalid JSON response type")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise InvalidPayload()
    try:
        result = json.loads(stripped[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        raise InvalidPayload() from None
    if not isinstance(result, dict):
        raise InvalidPayload("JSON response must be an object")
    return result


class FakeLlm:
    def __init__(self, responses: Sequence[str] | Callable[[LlmRequest], str]):
        self._responses = responses if callable(responses) else list(responses)
        self.requests: list[LlmRequest] = []

    def complete(self, req: LlmRequest) -> LlmResult:
        self.requests.append(req)
        if callable(self._responses):
            text = self._responses(req)
        else:
            if not self._responses:
                raise LlmError("fake LLM has no remaining responses")
            text = self._responses.pop(0)
        if not isinstance(text, str):
            raise LlmError("fake LLM response must be text")
        return LlmResult(text, "fake", "fake", 0, 0, 0)
