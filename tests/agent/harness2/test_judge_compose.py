"""Anonymous composition, retry, and majority-vote tests."""

import json
import re

from agent.harness2 import judge
from agent.harness2.llm import FakeLlm
from agent.harness2.models import (
    Decision,
    ProbeResult,
    Status,
    Usage,
    behavior_signature,
)


def _probe(label, text, *, status=Status.ok, hid=None):
    hid = hid or ("h-0123456789ab" if label == "base" else "h-aaaaaaaaaaaa")
    return ProbeResult("task", label, status, text, None, Usage(), "fake", "fake",
                       behavior_signature(status, text), "fake", hid, 0)


def _views(*entries):
    return [(candidate_id, result) for candidate_id, result in entries]


def _answer_for(payload, target):
    option = next(label for label, destination in payload.mapping.items() if destination == target)
    return json.dumps({"decision": "CANDIDATE", "chosen": option, "rationale": "based on output"})


def _payload(base, views, voter=1, seed=0, task="same task"):
    return judge.build_compose_payload(task, base, views, (), seed=seed, round=1, voter=voter)


def test_payload_has_no_harness_or_candidate_ids():
    base = _probe("base", "base output")
    candidates = _views(("cand-01", _probe("cand-01", "candidate output", hid="h-bbbbbbbbbbbb")))
    payload = _payload(base, candidates)
    assert "h-0123456789ab" not in payload.text
    assert "h-bbbbbbbbbbbb" not in payload.text
    assert "cand-01" not in payload.text
    assert "opt-" in payload.text


def test_shuffle_is_seeded_and_per_voter():
    base = _probe("base", "baseline")
    views = _views(
        ("cand-01", _probe("cand-01", "one", hid="h-111111111111")),
        ("cand-02", _probe("cand-02", "two", hid="h-222222222222")),
        ("cand-03", _probe("cand-03", "three", hid="h-333333333333")),
    )
    first = _payload(base, views, voter=1, seed=7)
    repeat = _payload(base, views, voter=1, seed=7)
    second = _payload(base, views, voter=2, seed=7)
    assert first == repeat
    assert set(first.mapping) == set(second.mapping)
    assert first.mapping != second.mapping


def test_base_unusable_forces_keep_base_without_llm_call():
    client = FakeLlm([])
    result = judge.compose("task", _probe("base", "", status=Status.error),
                           _views(("cand-01", _probe("cand-01", "candidate"))),
                           composer="llm", llm_client=client, k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.reason_code == "base_unusable"
    assert client.requests == []


def test_no_usable_candidate_forces_keep_base_without_llm_call():
    client = FakeLlm([])
    result = judge.compose("task", _probe("base", "base"),
                           _views(("cand-01", _probe("cand-01", "", status=Status.skipped))),
                           composer="llm", llm_client=client, k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.reason_code == "no_usable_candidate"
    assert client.requests == []


def test_choosing_base_label_maps_to_keep_base():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    payload = _payload(base, views)
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm([_answer_for(payload, "base")]), k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.reason_code == "chose_base"
    assert result.votes[0].chosen == "base"


def test_chosen_unusable_label_maps_to_keep_base():
    base = _probe("base", "base")
    views = _views(
        ("cand-01", _probe("cand-01", "", status=Status.skipped)),
        ("cand-02", _probe("cand-02", "usable", hid="h-bbbbbbbbbbbb")),
    )
    payload = _payload(base, views)
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm([_answer_for(payload, "cand-01")]), k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.votes[0].chosen == "cand-01"
    assert result.votes[0].error == "chosen_unusable"


def test_vote_chosen_is_resolved_label_not_opt():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    payload = _payload(base, views)
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm([_answer_for(payload, "cand-01")]), k=1)
    assert result.decision is Decision.CANDIDATE
    assert result.votes[0].chosen == "cand-01"
    assert not result.votes[0].chosen.startswith("opt-")


def test_majority_two_of_three_selects_candidate():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    responses = [_answer_for(_payload(base, views, voter=i), "cand-01") for i in (1, 2)]
    responses.append(_answer_for(_payload(base, views, voter=3), "base"))
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm(responses), k=3)
    assert result.decision is Decision.CANDIDATE
    assert result.chosen_label == "cand-01"
    assert result.reason_code == "llm"


def test_tie_forces_keep_base():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    responses = [
        _answer_for(_payload(base, views, voter=1), "cand-01"),
        _answer_for(_payload(base, views, voter=2), "base"),
    ]
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm(responses), k=2)
    assert result.decision is Decision.KEEP_BASE
    assert result.reason_code == "no_majority"


def test_swapping_only_outputs_changes_choice():
    base = _probe("base", "baseline")
    views_a = _views(("cand-01", _probe("cand-01", "PREFERRED A")),
                     ("cand-02", _probe("cand-02", "other B", hid="h-bbbbbbbbbbbb")))
    views_b = _views(("cand-01", _probe("cand-01", "other A")),
                     ("cand-02", _probe("cand-02", "PREFERRED B", hid="h-bbbbbbbbbbbb")))

    def choose_preferred(request):
        options = re.findall(r"(opt-\d+) \[status=ok\]\n([^\n]+)", request.user)
        label = next(label for label, text in options if "PREFERRED" in text)
        return json.dumps({"decision": "CANDIDATE", "chosen": label, "rationale": "preferred output"})

    first = judge.compose("task", base, views_a, composer="llm", llm_client=FakeLlm(choose_preferred), k=1)
    second = judge.compose("task", base, views_b, composer="llm", llm_client=FakeLlm(choose_preferred), k=1)
    assert first.chosen_label == "cand-01"
    assert second.chosen_label == "cand-02"


def test_caps_are_deterministic_and_never_cut_json():
    base = _probe("base", "B" * 9000)
    views = _views(("cand-01", _probe("cand-01", "C" * 9000)))
    first = judge.build_compose_payload("T" * 5000, base, views, (), seed=3, round=1, voter=1)
    second = judge.build_compose_payload("T" * 5000, base, views, (), seed=3, round=1, voter=1)
    assert first.text == second.text
    assert "truncated" in first.text
    assert 'Reply JSON only: {"decision"' in judge.COMPOSE_SYSTEM
    assert "opt-" in first.text


def test_invalid_then_valid_retries_once():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm(["not json", '{"decision":"KEEP_BASE","chosen":null,"rationale":"hold"}']), k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.votes[0].error is None
    assert result.votes[0].rationale == "hold"


def test_invalid_payload_twice_yields_invalid_vote():
    base = _probe("base", "base")
    views = _views(("cand-01", _probe("cand-01", "candidate")))
    result = judge.compose("task", base, views, composer="llm", llm_client=FakeLlm(["bad", "still bad"]), k=1)
    assert result.decision is Decision.KEEP_BASE
    assert result.reason_code == "no_majority"
    assert result.votes[0].error == "invalid_payload"
