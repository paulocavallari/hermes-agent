"""Contract tests for Harness² models and wire parsers."""

import json
from dataclasses import fields

import pytest

from agent.harness2 import models


def test_ids_accept_and_reject():
    assert models.valid_id("run", "run-0123456789ab")
    assert models.valid_id("harness", "h-0123456789ab")
    assert models.valid_id("domain", "my.domain-2")
    assert models.valid_id("proposal", "p-05")
    assert models.valid_id("candidate", "cand-05")
    assert models.valid_id("label", "opt-9")
    assert models.valid_id("lesson", "l-0123abcd")
    for kind, value in (("run", "run-ABC"), ("domain", "Bad"), ("proposal", "p-5"),
                        ("candidate", "cand-100"), ("label", "opt-100"),
                        ("lesson", "l-xyz"), ("domain", "a:b")):
        assert not models.valid_id(kind, value)


def test_harness_id_is_content_addressed_and_stable():
    harness = models.Harness.make("hello")
    assert harness.content_sha256 == __import__("hashlib").sha256(
        models.canonical_bytes(harness)
    ).hexdigest()
    assert harness.harness_id == "h-" + harness.content_sha256[:12]
    assert models.Harness.empty().harness_id.startswith("h-")


def test_harness_id_ignores_dict_order_and_crlf():
    left = models.Harness.make("hello\r\n", {"skills/z/SKILL.md": "z\r\n", "skills/a/SKILL.md": "a"})
    right = models.Harness.make("hello\n", {"skills/a/SKILL.md": "a", "skills/z/SKILL.md": "z\n"})
    assert left.harness_id == right.harness_id


def test_harness_id_changes_on_one_char():
    assert models.Harness.make("hello").harness_id != models.Harness.make("hello!").harness_id


def test_behavior_signature_excludes_harness_id():
    one = models.behavior_signature(models.Status.ok, "same", "h-0123456789ab")
    two = models.behavior_signature(models.Status.ok, "same", "h-ffffffffffff")
    assert one == two


def test_signature_ignores_trailing_spaces_and_blank_runs():
    one = models.behavior_signature(models.Status.ok, "a  \n\n\nb\t")
    two = models.behavior_signature(models.Status.ok, "a\n\nb")
    assert one == two
    assert models.behavior_signature(models.Status.ok, "A") != models.behavior_signature(models.Status.ok, "a")


def test_proposal_payload_parses_prompt_literal():
    literal = '{"dimension":"INV","edits":[{"op":"replace","path":"overlay.md","new_text":"Do work."}],"rationale":"clear","keep_base":false}'
    payload = models.parse_proposal_payload(json.loads(literal), expected_dimension=models.Dimension.INV)
    assert payload.dimension is models.Dimension.INV
    assert payload.edits[0] == models.WireEdit(models.Op.replace, "overlay.md", "Do work.")
    assert payload.keep_base is False


def test_selection_payload_parses_prompt_literal():
    literal = '{"decision":"CANDIDATE","chosen":"opt-2","rationale":"clear"}'
    payload = models.parse_selection_payload(json.loads(literal))
    assert payload.decision is models.Decision.CANDIDATE
    assert payload.chosen == "opt-2"


def test_wire_rejects_missing_extra_and_misplaced_new_text():
    valid = {"dimension": "INV", "edits": [], "rationale": "", "keep_base": True}
    for bad in (
        {"dimension": "INV", "edits": [], "keep_base": True},
        {**valid, "unexpected": 1},
        {**valid, "edits": [{"op": "remove", "path": "overlay.md", "new_text": "x"}], "keep_base": False},
        {**valid, "edits": [{"op": "add", "path": "overlay.md"}], "keep_base": False},
    ):
        with pytest.raises(models.InvalidPayload) as exc:
            models.parse_proposal_payload(bad)
        assert len(exc.value.detail) <= 200
        assert "offensive-secret" not in exc.value.detail


def test_wire_requires_dimension_match():
    with pytest.raises(models.InvalidPayload):
        models.parse_proposal_payload(
            {"dimension": "DOM", "edits": [], "rationale": "", "keep_base": True},
            expected_dimension=models.Dimension.INV,
        )


def test_persisted_from_dict_requires_internal_fields():
    with pytest.raises((KeyError, ValueError, TypeError)):
        models.Proposal.from_dict({"proposal_id": "p-01", "round": 1, "dimension": "INV"})


def test_from_dict_rejects_unknown_and_missing_keys():
    proposal = {
        "proposal_id": "p-01", "round": 1, "dimension": "INV", "edits": [],
        "rationale": "", "keep_base": True, "fell_back": False, "error": None,
    }
    with pytest.raises((KeyError, ValueError, TypeError)):
        models.Proposal.from_dict({**proposal, "extra": 1})
    with pytest.raises((KeyError, ValueError, TypeError)):
        models.Proposal.from_dict({key: val for key, val in proposal.items() if key != "error"})


def test_render_history_is_bounded_and_contains_classifications():
    classification = models.Classification("cand-01", "overlay.md", models.Verdict.hurt,
                                           models.Confidence.low, "not_isolated")
    summary = models.RoundSummary(1, "h-0123456789ab", ("cand-01",), models.Decision.KEEP_BASE,
                                  None, (classification,), None, "not_isolated")
    rendered = models.render_history([summary] * 20, cap_chars=300)
    assert len(rendered) <= 300
    assert "PRIOR ROUNDS" in rendered
    assert "overlay.md=hurt" in rendered
    assert len(rendered.splitlines()) <= 10


def test_schemas_match_required_fields():
    assert models.SCHEMAS["proposal_wire"] == ("dimension", "edits", "rationale", "keep_base")
    assert models.SCHEMAS["selection_wire"] == ("decision", "chosen", "rationale")
    for type_name, required in models.SCHEMAS.items():
        if type_name.endswith("_wire"):
            continue
        cls = models.PERSISTED_TYPES[type_name]
        names = {field.name for field in fields(cls) if field.init}
        assert set(required) == names
        assert len(required) == len(set(required))
