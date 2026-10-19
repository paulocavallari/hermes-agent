"""Anonymous selection and deterministic reflection for Harness²."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .harness import AppliedEdit
from .llm import BudgetExceeded, InvalidPayload as JsonInvalidPayload, parse_json_object, redact_strict
from .models import (
    CandidateReflection,
    Classification,
    Confidence,
    Decision,
    Evidence,
    Harness,
    Lesson,
    LlmClient,
    LlmRequest,
    Op,
    ProbeResult,
    Reflection,
    RoundSummary,
    Selection,
    SelectionPayload,
    Status,
    Verdict,
    Vote,
    parse_selection_payload,
    render_history,
)
from .validate import EvidenceError, verify_evidence

COMPOSE_SYSTEM = (
    'You compare system-prompt variants for the SAME task. You cannot see which is the baseline.\n'
    'Reply JSON only: {"decision":"CANDIDATE"|"KEEP_BASE","chosen":"opt-N"|null,"rationale":"<=300 chars"}.'
)
DEFAULT_COMPOSE_CAPS = {"per_option_cap": 6000, "total_cap": 30000, "task_cap": 4000, "history_cap": 1200}


@dataclass(frozen=True, slots=True)
class ComposePayload:
    text: str
    mapping: dict[str, str]


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    candidate_id: str
    harness: Harness
    edits: tuple[AppliedEdit, ...]
    probe: ProbeResult
    diff_path: str
    diff_text: str


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    removed = len(text) - cap
    head = max(1, int(cap * 0.7))
    tail = max(0, cap - head)
    marker = f"\n...[truncated {removed} chars]...\n"
    return text[:head] + marker + (text[-tail:] if tail else "")


def build_compose_payload(
    task: str,
    base_view: ProbeResult,
    candidate_views: Sequence[tuple[str, ProbeResult]],
    prior_rounds: Sequence[RoundSummary],
    *,
    seed: int,
    round: int,
    voter: int,
    caps: Mapping[str, int] | None = None,
) -> ComposePayload:
    limits = {**DEFAULT_COMPOSE_CAPS, **dict(caps or {})}
    task_view = _truncate(task, limits["task_cap"])
    history = render_history(prior_rounds, cap_chars=limits["history_cap"]) or "(none)"
    pool = [("base", base_view), *candidate_views]
    seed_text = f"{seed}:{round}:{voter}:{base_view.harness_id}"
    random_seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
    random.Random(random_seed).shuffle(pool)
    per_option_cap = min(limits["per_option_cap"], max(1200, (limits["total_cap"] - len(task_view) - len(history) - 1000) // max(1, len(pool))))
    mapping: dict[str, str] = {}
    option_lines: list[str] = []
    for index, (destination, result) in enumerate(pool, start=1):
        label = f"opt-{index}"
        mapping[label] = destination
        if result.status is Status.ok:
            option_lines.append(f"{label} [status=ok]\n{_truncate(result.text, per_option_cap)}")
        else:
            option_lines.append(f"{label} [status={result.status.value}] error={result.error or ''}")
    user = "\n".join((
        f"TASK / {task_view}",
        f"PRIOR ROUNDS / {history}",
        "OPTIONS",
        *option_lines,
    ))
    return ComposePayload(user, mapping)


def _selection_without_call(round_number: int, reason: str) -> Selection:
    return Selection(round_number, Decision.KEEP_BASE, None, reason, reason, (), {})


def _normalize_vote(voter: int, payload: SelectionPayload, mapping: dict[str, str], usable: set[str]) -> Vote:
    if payload.decision is Decision.KEEP_BASE:
        return Vote(voter, Decision.KEEP_BASE, "base", payload.rationale, None)
    destination = mapping.get(payload.chosen or "")
    if destination == "base":
        return Vote(voter, Decision.KEEP_BASE, "base", payload.rationale, None)
    if destination is None:
        return Vote(voter, Decision.KEEP_BASE, None, payload.rationale, "invalid_payload")
    if destination not in usable:
        return Vote(voter, Decision.KEEP_BASE, destination, payload.rationale, "chosen_unusable")
    return Vote(voter, Decision.CANDIDATE, destination, payload.rationale, None)


def _call_voter(
    *,
    task: str,
    base_probe: ProbeResult,
    candidate_views: Sequence[tuple[str, ProbeResult]],
    prior_rounds: Sequence[RoundSummary],
    client: LlmClient,
    seed: int,
    round_number: int,
    voter: int,
    budget: Any = None,
) -> tuple[Vote, dict[str, str]]:
    built = build_compose_payload(task, base_probe, candidate_views, prior_rounds,
                                  seed=seed, round=round_number, voter=voter)
    request = LlmRequest("compose", COMPOSE_SYSTEM, built.text, 256, 0.0)
    error = None
    for attempt in range(2):
        try:
            if budget is not None:
                budget.reserve(request.max_tokens)
            response = client.complete(request)
            payload = parse_selection_payload(parse_json_object(response.text))
            payload = SelectionPayload(payload.decision, payload.chosen, redact_strict(payload.rationale))
            usable = {candidate_id for candidate_id, probe in candidate_views if probe.status is Status.ok}
            return _normalize_vote(voter, payload, built.mapping, usable), built.mapping
        except (JsonInvalidPayload, ValueError) as exc:
            error = "invalid_payload"
            if attempt == 0:
                detail = getattr(exc, "detail", "invalid payload")
                request = LlmRequest(
                    request.purpose,
                    request.system,
                    request.user + f"\n\nYour previous reply was invalid: {detail}. Reply with JSON only.",
                    request.max_tokens,
                    request.temperature,
                    request.json_mode,
                )
                continue
            break
        except BudgetExceeded:
            error = "budget"
            break
        except Exception:  # health: allow BLE001 -- contain composer client failures as a safe fallback vote
            error = "llm_error"
            break
    return Vote(voter, Decision.KEEP_BASE, None, "", error or "invalid_payload"), built.mapping


def compose(
    task: str,
    base_probe: ProbeResult,
    candidate_views: Sequence[tuple[str, ProbeResult]],
    *,
    composer: str,
    llm_client: LlmClient | None,
    k: int,
    round_number: int = 1,
    prior_rounds: Sequence[RoundSummary] = (),
    seed: int = 0,
    budget: Any = None,
) -> Selection:
    if composer == "none":
        return _selection_without_call(round_number, "composer_disabled")
    if base_probe.status is not Status.ok:
        return _selection_without_call(round_number, "base_unusable")
    if not any(result.status is Status.ok for _, result in candidate_views):
        return _selection_without_call(round_number, "no_usable_candidate")
    if composer != "llm" or llm_client is None or type(k) is not int or not 1 <= k <= 5:
        return _selection_without_call(round_number, "invalid_payload")
    votes: list[Vote] = []
    mappings: list[dict[str, str]] = []
    for voter in range(1, k + 1):
        vote, mapping = _call_voter(
            task=task,
            base_probe=base_probe,
            candidate_views=candidate_views,
            prior_rounds=prior_rounds,
            client=llm_client,
            seed=seed,
            round_number=round_number,
            voter=voter,
            budget=budget,
        )
        votes.append(vote)
        mappings.append(mapping)
    candidate_counts: dict[str, int] = {}
    base_count = 0
    for vote in votes:
        if vote.decision is Decision.CANDIDATE and vote.chosen:
            candidate_counts[vote.chosen] = candidate_counts.get(vote.chosen, 0) + 1
        elif vote.decision is Decision.KEEP_BASE and vote.chosen == "base" and vote.error is None:
            base_count += 1
    majority = k / 2
    chosen = next((candidate for candidate, count in sorted(candidate_counts.items()) if count > majority), None)
    if chosen:
        decision, reason = Decision.CANDIDATE, "llm"
        rationale = next((vote.rationale for vote in votes if vote.chosen == chosen), "candidate selected")
    elif base_count > majority:
        decision, chosen, reason = Decision.KEEP_BASE, None, "chose_base"
        rationale = next((vote.rationale for vote in votes if vote.chosen == "base" and vote.error is None), "base retained")
    else:
        decision, chosen, reason = Decision.KEEP_BASE, None, "no_majority"
        rationale = "no strict majority"
    return Selection(round_number, decision, chosen, rationale, reason, tuple(votes), mappings[0] if mappings else {})


def _content_diff_lines(diff_text: str) -> tuple[str, ...]:
    ignored = ("--- ", "+++ ", "@@ ", "diff --git ", "index ")
    return tuple(
        line for line in diff_text.splitlines()
        if not line.startswith(ignored) and line != r"\\ No newline at end of file"
    )


def _candidate_gate_reason(
    base_probe: ProbeResult,
    observation: CandidateObservation,
    *,
    task_id: str,
) -> str | None:
    probe = observation.probe
    if base_probe.status is not Status.ok or probe.status is not Status.ok:
        return "missing_observation"
    if probe.task_id != task_id:
        return "task_mismatch"
    if probe.probe_mode != base_probe.probe_mode:
        return "probe_mode_mismatch"
    if probe.harness_id != observation.harness.harness_id:
        return "harness_mismatch"
    if not base_probe.model or not base_probe.provider or probe.model != base_probe.model or probe.provider != base_probe.provider:
        return "model_mismatch"
    return None


def _probe_evidence_paths(diff_path: str, candidate_id: str) -> tuple[str, str]:
    round_dir = PurePosixPath(diff_path).parent.parent
    return (
        (round_dir / "probes" / "base.json").as_posix(),
        (round_dir / "probes" / f"{candidate_id}.json").as_posix(),
    )


def _excerpt(text: str, limit: int = 200) -> str:
    return text[:limit]


def _build_lesson(
    observation: CandidateObservation,
    base_probe: ProbeResult,
    classification: Classification,
    *,
    domain: str,
    run_id: str,
    round_number: int,
    created_at: str,
) -> Lesson:
    path = classification.path
    applied = next(edit for edit in reversed(observation.edits) if edit.path == path)
    diff_lines = tuple(observation.diff_text.splitlines())
    first_diff_line = next(iter(_content_diff_lines(observation.diff_text)), "")
    base_probe_path, candidate_probe_path = _probe_evidence_paths(observation.diff_path, observation.candidate_id)
    evidence = Evidence(
        observation.diff_path,
        diff_lines,
        base_probe_path,
        _excerpt(base_probe.text),
        candidate_probe_path,
        _excerpt(observation.probe.text),
    )
    verify_evidence(
        evidence,
        base_probe=base_probe,
        cand_probe=observation.probe,
        diff_text=observation.diff_text,
        edited_paths=(path,),
        op=applied.op,
    )
    cue = f"editing {path} for {domain} tasks"
    lesson_text = (
        f"{applied.op.value} on {path} changed probe behavior "
        f"({base_probe.behavior_signature[:12]} -> {observation.probe.behavior_signature[:12]}); "
        f"{first_diff_line[:120]}"
    )
    lesson_id = "l-" + hashlib.sha256(f"{cue}|{lesson_text}|{observation.candidate_id}".encode()).hexdigest()[:8]
    confidence = Confidence.high if classification.classification is Verdict.helped else Confidence.medium
    return Lesson(
        lesson_id,
        domain,
        run_id,
        round_number,
        observation.candidate_id,
        path,
        cue,
        lesson_text,
        evidence,
        confidence,
        created_at,
    )


def reflect(
    base: Harness,
    base_probe: ProbeResult,
    candidates: Sequence[CandidateObservation],
    selection: Selection,
    *,
    task_id: str,
    domain: str = "default",
    run_id: str = "run-000000000000",
    round_number: int | None = None,
    created_at: str = "1970-01-01T00:00:00Z",
) -> Reflection:
    """Classify every edited candidate and produce at most one evidence-backed lesson."""
    selected_round = selection.round if round_number is None else round_number
    reflections: list[CandidateReflection] = []
    observations_by_id: dict[str, CandidateObservation] = {}
    possible_lessons: list[tuple[CandidateObservation, Classification]] = []
    evidence_failed = False

    for observation in sorted(candidates, key=lambda item: item.candidate_id):
        observations_by_id[observation.candidate_id] = observation
        paths = tuple(dict.fromkeys(edit.path for edit in observation.edits))
        if not paths:
            reflections.append(CandidateReflection(observation.candidate_id, observation.harness.harness_id, (), "no_change"))
            continue

        gate_reason = _candidate_gate_reason(base_probe, observation, task_id=task_id)
        classifications: list[Classification] = []
        if gate_reason is not None:
            classifications = [
                Classification(observation.candidate_id, path, Verdict.confounded, Confidence.low, gate_reason)
                for path in paths
            ]
        elif base_probe.behavior_signature == observation.probe.behavior_signature:
            classifications = [
                Classification(observation.candidate_id, path, Verdict.no_observed_effect, Confidence.medium, "identical_behavior")
                for path in paths
            ]
        elif len(paths) > 1:
            classifications = [
                Classification(observation.candidate_id, path, Verdict.confounded, Confidence.low, "not_isolated")
                for path in paths
            ]
        elif (
            selection.decision is Decision.CANDIDATE
            and selection.chosen_label == observation.candidate_id
            and selection.reason_code == "llm"
        ):
            classifications = [
                Classification(observation.candidate_id, paths[0], Verdict.helped, Confidence.high, "isolated_single_edit")
            ]
        elif selection.decision is Decision.KEEP_BASE and selection.reason_code == "chose_base":
            classifications = [
                Classification(observation.candidate_id, paths[0], Verdict.hurt, Confidence.medium, "isolated_single_edit_hurt")
            ]
        else:
            classifications = [
                Classification(observation.candidate_id, paths[0], Verdict.confounded, Confidence.low, "no_preference")
            ]

        reason = classifications[0].reason if classifications else "no_change"
        reflections.append(
            CandidateReflection(observation.candidate_id, observation.harness.harness_id, tuple(classifications), reason)
        )
        if len(classifications) == 1 and classifications[0].classification in {Verdict.helped, Verdict.hurt}:
            possible_lessons.append((observation, classifications[0]))

    verified_lessons: list[tuple[CandidateObservation, Lesson]] = []
    for observation, classification in possible_lessons:
        try:
            lesson = _build_lesson(
                observation,
                base_probe,
                classification,
                domain=domain,
                run_id=run_id,
                round_number=selected_round,
                created_at=created_at,
            )
        except EvidenceError:
            evidence_failed = True
        else:
            verified_lessons.append((observation, lesson))

    chosen_observation = next(
        (item for item, _ in verified_lessons
         if selection.decision is Decision.CANDIDATE
         and selection.reason_code == "llm"
         and selection.chosen_label == item.candidate_id),
        None,
    )
    if chosen_observation is not None:
        lesson = next(lesson for item, lesson in verified_lessons if item.candidate_id == chosen_observation.candidate_id)
    elif verified_lessons:
        lesson = min((lesson for _, lesson in verified_lessons), key=lambda item: item.candidate_id)
    else:
        lesson = None

    if lesson is not None:
        return Reflection(selected_round, base.harness_id, tuple(reflections), lesson, False, "lesson_recorded")
    if evidence_failed:
        reason = "evidence_mismatch"
    elif reflections:
        reason = min(reflections, key=lambda item: item.candidate_id).reason
    else:
        reason = "no_change"
    return Reflection(selected_round, base.harness_id, tuple(reflections), None, True, reason)
