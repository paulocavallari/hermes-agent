"""Deterministic reflection and evidence-backed lesson tests."""

from dataclasses import replace

from agent.harness2 import harness, judge
from agent.harness2.models import (
    Confidence,
    Decision,
    Edit,
    Op,
    ProbeResult,
    Selection,
    Status,
    Usage,
    Verdict,
    Vote,
    behavior_signature,
)


def _probe(label, bundle, text, *, status=Status.ok, task_id="task", mode="fake", model="fake", provider="fake"):
    return ProbeResult(task_id, label, status, text, None if status is Status.ok else "failed", Usage(),
                       model, provider, behavior_signature(status, text), mode, bundle.harness_id, 0)


def _candidate(base, candidate_id="cand-01", text="candidate answer", paths=("overlay.md",), *, status=Status.ok,
               task_id="task", mode="fake", model="fake", provider="fake", diff_override=None):
    edits = []
    current = base
    for index, path in enumerate(paths):
        if path == "overlay.md":
            edit = Edit(Op.replace, path, f"candidate {index}")
        else:
            edit = Edit(Op.add, path, "support body")
        current, applied = harness.apply_edits(current, (edit,))
        edits.extend(applied)
    diff = harness.diff_harness(base, current) if diff_override is None else diff_override
    probe = _probe(candidate_id, current, text, status=status, task_id=task_id, mode=mode, model=model, provider=provider)
    return judge.CandidateObservation(candidate_id, current, tuple(edits), probe,
                                      f"diffs/{candidate_id}.diff", diff)


def _selection(decision=Decision.CANDIDATE, chosen: str | None = "cand-01", reason="llm"):
    vote = Vote(1, decision, chosen if decision is Decision.CANDIDATE else "base", "vote", None)
    return Selection(1, decision, chosen if decision is Decision.CANDIDATE else None,
                     "selected", reason, (vote,), {"opt-1": chosen or "base"})


def _reflect(base, base_probe, candidates, selection, **kwargs):
    return judge.reflect(base, base_probe, candidates, selection, task_id="task", domain="demo",
                         run_id="run-0123456789ab", round_number=1,
                         created_at="2025-01-01T00:00:00Z", **kwargs)


def test_identical_behavior_is_no_observed_effect_and_no_entry():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "same answer")
    observation = _candidate(base, text="same answer")
    result = _reflect(base, base_probe, (observation,), _selection(Decision.KEEP_BASE, None, "no_majority"))
    item = result.candidates[0].classifications[0]
    assert item.classification is Verdict.no_observed_effect
    assert item.confidence is Confidence.medium
    assert result.no_entry is True
    assert result.lesson is None


def test_reflect_classifies_all_candidates():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "base output")
    candidates = tuple(_candidate(base, candidate_id=f"cand-{n:02d}", text=f"output {n}") for n in range(1, 4))
    result = _reflect(base, base_probe, candidates, _selection(chosen="cand-02"))
    assert [candidate.candidate_id for candidate in result.candidates] == ["cand-01", "cand-02", "cand-03"]
    assert all(candidate.classifications for candidate in result.candidates)
    assert result.candidates[1].classifications[0].classification is Verdict.helped


def test_chosen_single_edit_is_helped_with_lesson():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "base answer with evidence")
    observation = _candidate(base, text="candidate answer with evidence")
    result = _reflect(base, base_probe, (observation,), _selection())
    assert result.candidates[0].classifications[0].classification is Verdict.helped
    assert result.candidates[0].classifications[0].confidence is Confidence.high
    assert result.lesson is not None
    assert result.lesson.candidate_id == "cand-01"
    assert result.lesson.cue == "editing overlay.md for demo tasks"
    assert result.lesson.evidence.base_excerpt in base_probe.text
    assert result.lesson.evidence.candidate_excerpt in observation.probe.text


def test_explicit_base_majority_is_hurt_with_lesson():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "base answer")
    observation = _candidate(base, text="candidate answer")
    result = _reflect(base, base_probe, (observation,), _selection(Decision.KEEP_BASE, None, "chose_base"))
    classification = result.candidates[0].classifications[0]
    assert classification.classification is Verdict.hurt
    assert classification.confidence is Confidence.medium
    assert result.lesson is not None


def test_composer_disabled_with_distinct_outputs_is_confounded():
    base = harness.Harness.make("base")
    result = _reflect(base, _probe("base", base, "base"), (_candidate(base, text="candidate"),),
                      _selection(Decision.KEEP_BASE, None, "composer_disabled"))
    assert result.candidates[0].classifications[0].classification is Verdict.confounded
    assert result.candidates[0].classifications[0].reason == "no_preference"
    assert result.lesson is None


def test_tie_is_no_preference_not_hurt():
    base = harness.Harness.make("base")
    result = _reflect(base, _probe("base", base, "base"), (_candidate(base, text="candidate"),),
                      _selection(Decision.KEEP_BASE, None, "no_majority"))
    classification = result.candidates[0].classifications[0]
    assert classification.classification is Verdict.confounded
    assert classification.reason == "no_preference"
    assert result.lesson is None


def test_parse_failure_is_no_preference():
    base = harness.Harness.make("base")
    result = _reflect(base, _probe("base", base, "base"), (_candidate(base, text="candidate"),),
                      _selection(Decision.KEEP_BASE, None, "invalid_payload"))
    assert result.candidates[0].classifications[0].classification is Verdict.confounded
    assert result.candidates[0].classifications[0].reason == "no_preference"


def test_model_or_provider_mismatch_is_confounded():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "base")
    observation = _candidate(base, text="candidate", provider="other")
    result = _reflect(base, base_probe, (observation,), _selection())
    assert result.candidates[0].classifications[0].classification is Verdict.confounded
    assert result.candidates[0].classifications[0].reason == "model_mismatch"


def test_two_paths_confounded():
    base = harness.Harness.make("base")
    observation = _candidate(base, text="candidate", paths=("overlay.md", "skills/demo/SKILL.md"))
    result = _reflect(base, _probe("base", base, "base"), (observation,), _selection())
    assert len(result.candidates[0].classifications) == 2
    assert all(item.classification is Verdict.confounded for item in result.candidates[0].classifications)
    assert result.candidates[0].reason == "not_isolated"
    assert result.lesson is None


def test_errored_probe_is_confounded():
    base = harness.Harness.make("base")
    observation = _candidate(base, text="", status=Status.error)
    result = _reflect(base, _probe("base", base, "base"), (observation,), _selection())
    assert result.candidates[0].classifications[0].classification is Verdict.confounded
    assert result.candidates[0].classifications[0].reason == "missing_observation"


def test_lesson_priority_chosen_then_stable_order():
    base = harness.Harness.make("base")
    base_probe = _probe("base", base, "base")
    observations = (_candidate(base, "cand-02", "two"), _candidate(base, "cand-01", "one"))
    selected = _reflect(base, base_probe, observations, _selection(chosen="cand-02"))
    assert selected.lesson.candidate_id == "cand-02"
    unselected = _reflect(base, base_probe, observations, _selection(Decision.KEEP_BASE, None, "chose_base"))
    assert unselected.lesson.candidate_id == "cand-01"


def test_at_most_one_lesson_per_round():
    base = harness.Harness.make("base")
    observations = tuple(_candidate(base, f"cand-{n:02d}", f"result {n}") for n in range(1, 4))
    result = _reflect(base, _probe("base", base, "base"), observations, _selection(chosen="cand-02"))
    assert sum(candidate.candidate_id == result.lesson.candidate_id for candidate in result.candidates) == 1


def test_no_change_round_has_no_lesson():
    base = harness.Harness.make("base")
    observation = judge.CandidateObservation("cand-01", base, (), _probe("cand-01", base, "base"), "diffs/cand-01.diff", "")
    result = _reflect(base, _probe("base", base, "base"), (observation,),
                      _selection(Decision.KEEP_BASE, None, "no_majority"))
    assert result.candidates[0].reason == "no_change"
    assert result.candidates[0].classifications == ()
    assert result.no_entry is True
    assert result.lesson is None


def test_reflection_fails_closed_on_task_mode_and_harness_mismatch():
    base = harness.Harness.make("base")
    cases = (
        ("task_mismatch", {"task_id": "other"}, "task_mismatch"),
        ("probe_mode_mismatch", {"mode": "live"}, "probe_mode_mismatch"),
        ("harness_mismatch", {"harness": harness.Harness.make("another")}, "harness_mismatch"),
    )
    for _, values, expected in cases:
        observation = _candidate(base, text="candidate")
        if "task_id" in values:
            observation = replace(observation, probe=replace(observation.probe, task_id=values["task_id"]))
        if "mode" in values:
            observation = replace(observation, probe=replace(observation.probe, probe_mode=values["mode"]))
        if "harness" in values:
            observation = replace(observation, probe=replace(observation.probe, harness_id=values["harness"].harness_id))
        result = _reflect(base, _probe("base", base, "base"), (observation,), _selection())
        assert result.candidates[0].classifications[0].classification is Verdict.confounded
        assert result.candidates[0].classifications[0].reason == expected


def test_evidence_mismatch_blocks_library_lesson():
    base = harness.Harness.make("base")
    observation = _candidate(base, text="candidate", diff_override="--- a/elsewhere.md\n+++ b/elsewhere.md\n+different\n")
    result = _reflect(base, _probe("base", base, "base"), (observation,), _selection())
    assert result.lesson is None
    assert result.no_entry is True
    assert result.reason == "evidence_mismatch"
