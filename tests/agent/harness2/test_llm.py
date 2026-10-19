"""Hermetic LLM transport and budget tests."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness2 import llm
from agent.harness2.models import LlmRequest


def _request():
    return LlmRequest("propose", "system prompt", "user prompt", 128, 0.2)


def test_complete_builds_messages_without_tools_or_stream():
    captured = {}

    def call(**kwargs):
        captured.update(kwargs)
        return "reply"

    result = llm.AuxLlmClient(call=call).complete(_request())
    assert result.text == "reply"
    assert captured["task"] == "harness2"
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert "tools" not in captured
    assert "stream" not in captured
    assert "extra_body" not in captured


@pytest.mark.parametrize("response, expected", [
    ("plain", "plain"),
    ({"choices": [{"message": {"content": "dict text"}}]}, "dict text"),
    (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="object text"))]), "object text"),
])
def test_complete_extracts_text_from_str_dict_and_object(response, expected):
    client = llm.AuxLlmClient(call=lambda **_kwargs: response)
    assert client.complete(_request()).text == expected


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}, (3, 4, 7)),
    (SimpleNamespace(input_tokens=5, output_tokens=6, total_tokens=11), (5, 6, 11)),
])
def test_usage_extraction_object_and_dict(usage, expected):
    response = {"choices": [{"message": {"content": "ok"}}], "usage": usage}
    result = llm.AuxLlmClient(call=lambda **_kwargs: response).complete(_request())
    assert (result.input_tokens, result.output_tokens) == expected[:2]
    assert llm._response_usage(response) == expected


def test_aux_client_forwards_model_and_provider_kwargs():
    captured = {}

    def call(**kwargs):
        captured.update(kwargs)
        kwargs["route_info"].update({"model": "actual-model", "provider": "actual-provider"})
        return {"choices": [{"message": {"content": "ok"}}]}

    result = llm.AuxLlmClient(model="selected-model", provider="selected-provider", call=call).complete(_request())
    assert captured["model"] == "selected-model"
    assert captured["provider"] == "selected-provider"
    assert result.model == "actual-model"
    assert result.provider == "actual-provider"


def test_parse_json_object_accepts_fenced_and_bare():
    assert llm.parse_json_object('{"x": 1}') == {"x": 1}
    assert llm.parse_json_object('```json\n{"x": 2}\n```') == {"x": 2}


def test_parse_rejects_non_object():
    with pytest.raises(llm.InvalidPayload):
        llm.parse_json_object('["not", "object"]')


def test_transport_error_is_wrapped_without_secrets():
    def fail(**_kwargs):
        raise RuntimeError("SERVICE_API_KEY=unit-test-secret-value")

    with pytest.raises(llm.LlmError) as exc:
        llm.AuxLlmClient(call=fail).complete(_request())
    assert "unit-test-secret-value" not in str(exc.value)
    assert len(exc.value.detail) <= 200


def test_budget_exceeded_after_max_calls():
    budget = llm.Budget(max_calls=2, max_tokens=100)
    budget.reserve(10)
    budget.reserve(20)
    with pytest.raises(llm.BudgetExceeded):
        budget.reserve(1)


def test_llm_imports_redact_strict_directly_from_store():
    path = Path(llm.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module in {"agent.harness2.store", ".store"}
    ]
    assert any(any(alias.name == "redact_strict" for alias in node.names) for node in imports)
    assert not any(isinstance(node, ast.Import) and any(alias.name.endswith("redact_strict") for alias in node.names)
                   for node in tree.body)
