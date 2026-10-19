"""Offline-first Harness² evolution orchestration."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Callable, Sequence

from . import harness, judge, library, probe, store, validate
from .llm import AuxLlmClient, Budget, BudgetExceeded, FakeLlm, InvalidPayload, LlmError, parse_json_object
from .models import (
    Decision,
    Dimension,
    Edit,
    Harness,
    Ledger,
    LlmClient,
    LlmRequest,
    LlmResult,
    Op,
    Proposal,
    ProbeResult,
    RoundSummary,
    RunResult,
    Selection,
    StageUsage,
    Status,
    Usage,
    render_history,
    valid_id,
)

PROPOSER_SYSTEM = (
    "You edit a portable agent harness for ONE task. A harness is overlay.md plus on-demand skills.\n"
    'Return JSON only: {"dimension":"INV"|"DOM","edits":[{"op":"add"|"replace"|"remove",'
    '"path":"...","new_text":"..."}],"rationale":"<=300 chars","keep_base":false}\n'
    "Rules: relative paths only; no traversal, absolute paths, backslashes, or secrets. "
    "Every added skill must be named in overlay.md as a trigger. Prefer one minimal edit."
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _hex_token(chars: int) -> str:
    return secrets.token_hex((chars + 1) // 2)[:chars]


@dataclass(frozen=True, slots=True)
class Deps:
    llm: LlmClient | None = None
    llm_factory: Callable[[str | None], LlmClient] | None = None
    runner: object | None = None
    now: Callable[[], datetime] = _utcnow
    monotonic: Callable[[], float] = time.monotonic
    token: Callable[[int], str] = _hex_token
    price_fn: Callable[[str, int, int], float | None] | None = None


@dataclass(frozen=True, slots=True)
class EvolveOptions:
    task_path: Path
    domain: str = "default"
    mode: str = "parallel"
    k: int | None = None
    rounds: int | None = None
    proposer: str = "template"
    probe: str = "none"
    composer: str = "none"
    base_dir: Path | None = None
    seed: int = 0
    apply: bool = False
    dry_run: bool = False
    model: str | None = None
    max_llm_calls: int = 32


class _MeteredClient:
    def __init__(self, client: LlmClient):
        self.client = client
        self.calls = 0
        self.successes = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.stage = "propose"
        self.stage_calls = {name: 0 for name in ("propose", "probe", "compose", "reflect")}
        self.stage_usage = {name: [0, 0, 0] for name in self.stage_calls}

    def complete(self, req: LlmRequest) -> LlmResult:
        self.calls += 1
        self.stage_calls[self.stage] += 1
        response = self.client.complete(req)
        self.successes += 1
        usage = (response.input_tokens, response.output_tokens, response.input_tokens + response.output_tokens)
        self.input_tokens += usage[0]
        self.output_tokens += usage[1]
        self.total_tokens += usage[2]
        counters = self.stage_usage[self.stage]
        for index, count in enumerate(usage):
            counters[index] += count
        return response


def _write_text(writer: store.Store, rel: str, text: str) -> Path:
    # Delegate every write through Store without duplicating its writer surface.
    writer_write = writer.write_text
    return writer_write(rel, text)


def _created_at(deps: Deps) -> str:
    now = deps.now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _options_error(options: EvolveOptions) -> tuple[str, int, int]:
    if not valid_id("domain", options.domain):
        return "E_USAGE: invalid domain", 0, 0
    if options.mode not in {"parallel", "sequential"}:
        return "E_USAGE: invalid mode", 0, 0
    if options.proposer not in {"template", "llm"} or options.probe not in {"none", "live"} or options.composer not in {"none", "llm"}:
        return "E_USAGE: invalid stage option", 0, 0
    if options.mode == "parallel":
        if options.rounds is not None:
            return "E_USAGE: rounds is only valid in sequential mode", 0, 0
        count = 3 if options.k is None else options.k
        if type(count) is not int or not 1 <= count <= 5:
            return "E_USAGE: k must be in 1..5", 0, 0
        rounds = 1
    else:
        if options.k is not None:
            return "E_USAGE: k is only valid in parallel mode", 0, 0
        rounds = 2 if options.rounds is None else options.rounds
        if type(rounds) is not int or not 1 <= rounds <= 5:
            return "E_USAGE: rounds must be in 1..5", 0, 0
        count = 1
    if type(options.seed) is not int or type(options.max_llm_calls) is not int or options.max_llm_calls < 0:
        return "E_USAGE: invalid numeric option", 0, 0
    return "", count, rounds


def _fallback(proposal_id: str, round_number: int, dimension: Dimension, error: str) -> Proposal:
    return Proposal(proposal_id, round_number, dimension, (), "", True, True, error)


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:max(0, cap - 3)] + "..."


def build_proposal_payload(
    task: str,
    base: harness.Harness,
    context,
    prior_rounds: Sequence[RoundSummary],
    dimension: Dimension,
) -> str:
    task_text = _truncate(task, 4000)
    overlay_text = base.overlay
    skill_texts = [
        f"{path}\n{base.files[path]}"
        for path in sorted(base.files)
        if path.endswith("/SKILL.md")
    ]
    context_text = context.render(cap_chars=4000) if context else "(none)"
    history = render_history(prior_rounds, cap_chars=1200) or "(none)"
    fixed = [
        f"DIMENSION / {dimension.value}",
        f"TASK / {task_text}",
        f"LIBRARY CONTEXT / {context_text}",
        f"PRIOR ROUNDS / {history}",
    ]
    def render() -> str:
        return "\n\n".join([*fixed, f"BASE OVERLAY / {overlay_text}", "BASE SKILLS", *skill_texts])
    payload = render()
    if len(payload) > 20_000 and overlay_text:
        overlay_text = _truncate(overlay_text, max(0, len(overlay_text) - (len(payload) - 20_000)))
        payload = render()
    while len(payload) > 20_000 and skill_texts:
        skill_texts.pop()
        payload = render()
    return payload


def _propose_template(
    proposal_id: str,
    round_number: int,
    dimension: Dimension,
    base: harness.Harness,
    domain: str,
    context,
    prior_rounds: Sequence[RoundSummary],
) -> Proposal:
    if dimension is Dimension.INV:
        lesson_lines = [f"- Apply prior lesson: {item.cue}" for item in (context.lessons[-3:] if context else ())]
        text = "\n".join([
            base.overlay.rstrip("\n"),
            "## Operating protocol",
            "- State the next concrete action before acting.",
            "- Verify each change with a command or a read-back.",
            *lesson_lines,
        ]).strip() + "\n"
        if len(text) > validate.MAX_OVERLAY_CHARS:
            return _fallback(proposal_id, round_number, dimension, "overlay_cap")
        edits = (Edit(Op.replace, "overlay.md", text),)
    else:
        skill_paths = sorted(path for path in base.files if path.startswith("skills/") and path.endswith("/SKILL.md"))
        cue = f"When {domain} work stalls, restate the goal and the smallest verifiable step."
        if skill_paths:
            path = skill_paths[0]
            text = base.files[path].rstrip() + f"\n\n## Practice cue\n- {cue}\n"
            if len(text.encode("utf-8")) > validate.MAX_SKILL_BYTES:
                return _fallback(proposal_id, round_number, dimension, "skill_cap")
            edits = (Edit(Op.replace, path, text),)
        else:
            skill_name = f"{domain[:55]}-practice"
            description = f"Use when {domain} work needs a practice cue"[:60]
            skill_text = f"---\nname: {skill_name}\ndescription: {description}\n---\n\n## Practice cue\n- {cue}\n"
            overlay = base.overlay.rstrip("\n") + f"\n- {skill_name}: use when {domain} work needs a practice cue.\n"
            if len(overlay) > validate.MAX_OVERLAY_CHARS:
                return _fallback(proposal_id, round_number, dimension, "overlay_cap")
            if len(skill_text.encode("utf-8")) > validate.MAX_SKILL_BYTES:
                return _fallback(proposal_id, round_number, dimension, "skill_cap")
            edits = (
                Edit(Op.add, f"skills/{skill_name}/SKILL.md", skill_text),
                Edit(Op.replace, "overlay.md", overlay),
            )
    return Proposal(proposal_id, round_number, dimension, edits, f"template {dimension.value.lower()} proposal", False, False, None)


def _propose_llm(
    proposal_id: str,
    round_number: int,
    dimension: Dimension,
    task: str,
    base: harness.Harness,
    context,
    prior_rounds: Sequence[RoundSummary],
    client: LlmClient,
    budget: Budget,
) -> Proposal:
    from .models import parse_proposal_payload
    from .store import redact_strict

    user = build_proposal_payload(task, base, context, prior_rounds, dimension)
    request = LlmRequest("propose", PROPOSER_SYSTEM, user, 2048, 0.0)
    for attempt in range(2):
        try:
            budget.reserve(request.max_tokens)
            response = client.complete(request)
            payload = parse_proposal_payload(parse_json_object(response.text), expected_dimension=dimension)
            edits = tuple(Edit(item.op, item.path, item.new_text) for item in payload.edits)
            return Proposal(proposal_id, round_number, dimension, edits, redact_strict(payload.rationale),
                            payload.keep_base, False, None)
        except BudgetExceeded:
            return _fallback(proposal_id, round_number, dimension, "budget")
        except (InvalidPayload, ValueError) as exc:
            if attempt == 1:
                return _fallback(proposal_id, round_number, dimension, "invalid_payload")
            detail = getattr(exc, "detail", "invalid payload")
            request = LlmRequest(request.purpose, request.system,
                                 request.user + f"\n\nYour previous reply was invalid: {detail}. Reply with JSON only.",
                                 request.max_tokens, request.temperature, request.json_mode)
        except Exception:  # health: allow BLE001 -- malformed LLM replies fall back without aborting the run
            return _fallback(proposal_id, round_number, dimension, "invalid_payload")
    return _fallback(proposal_id, round_number, dimension, "invalid_payload")


def _write_proposal(writer: store.Store, rel: str, proposal: Proposal) -> None:
    writer.write_json(rel, {"schema_version": 1, **proposal.to_dict()})


def _probe_one(
    writer: store.Store,
    rel: str,
    bundle: harness.Harness,
    task: str,
    runner,
    *,
    task_id: str,
    label: str,
    deps: Deps,
    meter: _MeteredClient,
) -> ProbeResult:
    meter.stage = "probe"
    observed = probe.run_probe(bundle, task, runner, task_id=task_id, label=label, monotonic=deps.monotonic)
    writer.write_json(rel, {"schema_version": 1, **observed.to_dict()})
    return observed


def _contrast(
    round_number: int,
    task_id: str,
    base_probe: ProbeResult,
    observations: Sequence[judge.CandidateObservation],
    reflection,
    selection: Selection,
) -> str:
    lines = [
        f"# Contrast — round {round_number:02d}",
        f"task: {task_id}",
        f"base:    {base_probe.harness_id} status={base_probe.status.value} behavior={base_probe.behavior_signature} model={base_probe.model} provider={base_probe.provider}",
    ]
    by_id = {item.candidate_id: item for item in observations}
    for candidate in reflection.candidates:
        probe_result = by_id.get(candidate.candidate_id)
        if probe_result:
            result = probe_result.probe
            lines.append(f"{candidate.candidate_id}: {result.harness_id} status={result.status.value} behavior={result.behavior_signature} model={result.model} provider={result.provider}")
    lines.extend(["| candidate | path | classification | confidence |", "|---|---|---|---|"])
    for candidate in reflection.candidates:
        for item in candidate.classifications:
            lines.append(f"| {item.candidate_id} | {item.path} | {item.classification.value} | {item.confidence.value} |")
    for candidate in reflection.candidates:
        lines.append(f"reason: {candidate.reason}")
        if candidate.candidate_id in by_id:
            lines.append(f"diff: {by_id[candidate.candidate_id].diff_path}")
    lines.append(f"selection: {selection.reason_code}")
    return "\n".join(lines) + "\n"


def _round_selection(selection: Selection, run_id: str, base: harness.Harness, final: harness.Harness,
                     round_decisions: Sequence[Selection]) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "decision": selection.decision.value,
        "changed": final.harness_id != base.harness_id,
        "initial_harness_id": base.harness_id,
        "final_harness_id": final.harness_id,
        "reason_code": selection.reason_code,
        "rounds": [{"round": item.round, "decision": item.decision.value} for item in round_decisions],
    }


def _run_parallel(
    writer: store.Store,
    options: EvolveOptions,
    deps: Deps,
    task: str,
    task_id: str,
    base: harness.Harness,
    run_id: str,
    created_at: str,
    k: int,
    client: LlmClient | None,
    meter: _MeteredClient,
    budget: Budget,
    runner,
) -> RunResult:
    from .models import CandidateReflection

    run_rel = f"runs/{run_id}"
    round_rel = f"{run_rel}/rounds/round-01"
    started = deps.monotonic()
    options_payload = {"proposer": options.proposer, "probe": options.probe, "composer": options.composer,
                       "seed": options.seed, "model": options.model}
    manifest = {
        "schema_version": 1, "run_id": run_id, "task_id": task_id, "domain": options.domain,
        "created_at": created_at, "mode": options.mode, "k": k, "rounds": 1,
        "base_harness_id": base.harness_id, "final_harness_id": base.harness_id,
        "decision": Decision.KEEP_BASE.value, "options": options_payload,
        "apply_requested": options.apply, "dry_run": options.dry_run,
    }
    writer.write_json(f"{run_rel}/manifest.json", manifest)
    _write_text(writer, f"{run_rel}/TASK.md", task)
    harness.snapshot_round(writer, f"{round_rel}/base", base)
    writer.write_json(f"{round_rel}/round.json", {"schema_version": 1, "round": 1, "base_harness_id": base.harness_id})

    context = library.build_context(writer, options.domain)
    proposals: list[Proposal] = []
    observations: list[judge.CandidateObservation] = []
    invalid_count = 0
    stage_propose_calls = 0
    propose_ms = 0

    for index in range(1, k + 1):
        pair: list[Proposal] = []
        current = base
        for dimension in (Dimension.INV, Dimension.DOM):
            proposal_number = (index - 1) * 2 + len(pair) + 1
            proposal_id = f"p-{proposal_number:02d}"
            if dimension is Dimension.DOM and pair and not pair[0].keep_base and pair[0].edits:
                try:
                    current, _ = harness.apply_edits(base, pair[0].edits)
                except harness.EditError:
                    current = base
            stage_propose_calls += 1
            before = deps.monotonic()
            if options.proposer == "template":
                proposed = _propose_template(proposal_id, 1, dimension, current, options.domain, context, ())
            else:
                assert client is not None
                meter.stage = "propose"
                proposed = _propose_llm(proposal_id, 1, dimension, task, current, context, (), client, budget)
            propose_ms += max(0, int((deps.monotonic() - before) * 1000))
            proposals.append(proposed)
            pair.append(proposed)
            _write_proposal(writer, f"{round_rel}/proposals/{proposal_id}.json", proposed)
        all_edits = tuple(edit for proposal in pair if not proposal.keep_base for edit in proposal.edits)
        if not all_edits:
            continue
        candidate_id = f"cand-{index:02d}"
        try:
            first, applied_inv = harness.apply_edits(base, pair[0].edits if not pair[0].keep_base else ())
            candidate, applied_dom = harness.apply_edits(first, pair[1].edits if not pair[1].keep_base else ())
            applied_by_path = {edit.path: edit for edit in (*applied_inv, *applied_dom)}
            applied = tuple(applied_by_path.values())
        except harness.EditError as exc:
            invalid_count += 1
            writer.write_json(f"{round_rel}/candidates/{candidate_id}.errors.json",
                              {"candidate_id": candidate_id, "errors": [{"code": exc.code, "path": exc.path}]})
            continue
        validation = validate.validate_harness(candidate)
        if not validation.valid:
            invalid_count += 1
            writer.write_json(f"{round_rel}/candidates/{candidate_id}.errors.json",
                              {"candidate_id": candidate_id,
                               "errors": [{"code": error.code, "path": error.path, "detail": error.detail}
                                          for error in validation.errors]})
            continue
        harness.snapshot_round(writer, f"{round_rel}/candidates/{candidate_id}", candidate)
        diff_text = harness.diff_harness(base, candidate)
        diff_rel = f"{round_rel}/diffs/{candidate_id}.diff"
        _write_text(writer, diff_rel, diff_text)
        probe_rel = f"{round_rel}/probes/{candidate_id}.json"
        cand_probe = _probe_one(writer, probe_rel, candidate, task, runner, task_id=task_id,
                                label=candidate_id, deps=deps, meter=meter)
        observations.append(judge.CandidateObservation(candidate_id, candidate, applied, cand_probe, diff_rel, diff_text))

    base_probe = _probe_one(writer, f"{round_rel}/probes/base.json", base, task, runner,
                            task_id=task_id, label="base", deps=deps, meter=meter)
    # Store and probe the base before candidate probes in the on-disk contract.
    # Reorder lexically through a stable replacement after the actual observations.
    candidate_views = [(item.candidate_id, item.probe) for item in observations]
    compose_started = deps.monotonic()
    if invalid_count and not observations:
        selection = Selection(1, Decision.KEEP_BASE, None, "all candidates invalid", "no_usable_candidate", (), {})
    elif not proposals or not any(not proposal.keep_base and proposal.edits for proposal in proposals):
        selection = Selection(1, Decision.KEEP_BASE, None, "no edits proposed", "no_change", (), {})
    else:
        meter.stage = "compose"
        selection = judge.compose(
            task, base_probe, candidate_views, composer=options.composer, llm_client=client,
            k=k if options.composer == "llm" else 1, round_number=1, prior_rounds=(), seed=options.seed, budget=budget,
        )
    compose_ms = max(0, int((deps.monotonic() - compose_started) * 1000))
    chosen = next((item.harness for item in observations if selection.decision is Decision.CANDIDATE
                   and selection.chosen_label == item.candidate_id), base)
    reflect_started = deps.monotonic()
    reflection = judge.reflect(base, base_probe, observations, selection, task_id=task_id,
                               domain=options.domain, run_id=run_id, round_number=1, created_at=created_at)
    reflect_ms = max(0, int((deps.monotonic() - reflect_started) * 1000))
    summary = RoundSummary(
        1, base.harness_id, tuple(item.candidate_id for item in observations), selection.decision,
        selection.chosen_label, tuple(classification for item in reflection.candidates for classification in item.classifications),
        reflection.lesson.lesson_id if reflection.lesson else None, selection.reason_code,
    )
    writer.write_json(f"{round_rel}/selection.json", selection)
    writer.write_json(f"{round_rel}/reflection.json", reflection)
    _write_text(writer, f"{round_rel}/contrast.md", _contrast(1, task_id, base_probe, observations, reflection, selection))
    writer.write_json(f"{run_rel}/selection.json", _round_selection(selection, run_id, base, chosen, (selection,)))
    writer.write_json(f"{run_rel}/reflection.json", {
        "schema_version": 1, "run_id": run_id,
        "lessons": [reflection.lesson.to_dict()] if reflection.lesson else [],
        "rounds": [{"round": 1, "no_entry": reflection.no_entry, "reason": reflection.reason,
                    "candidates": [item.to_dict() for item in reflection.candidates]}],
    })

    actual_probe_calls = sum(item.probe.status is not Status.skipped for item in observations) + int(base_probe.status is not Status.skipped)
    probe_ms = sum(item.probe.wall_ms for item in observations) + base_probe.wall_ms
    stages = {
        "propose": StageUsage(stage_propose_calls, propose_ms),
        "probe": StageUsage(actual_probe_calls, probe_ms),
        "compose": StageUsage(len(selection.votes), compose_ms),
        "reflect": StageUsage(1, reflect_ms),
    }
    extra_probe_usage = [item.probe.usage for item in observations] + [base_probe.usage]
    input_tokens = meter.input_tokens + sum(item.input_tokens for item in extra_probe_usage if options.probe != "live")
    output_tokens = meter.output_tokens + sum(item.output_tokens for item in extra_probe_usage if options.probe != "live")
    tokens = Usage(input_tokens, output_tokens, input_tokens + output_tokens)
    cost = None
    if deps.price_fn:
        price_values = [deps.price_fn(options.model or "", input_tokens, output_tokens)]
        if all(value is not None for value in price_values):
            cost = sum(value for value in price_values if value is not None)
    wall_ms = max(0, int((deps.monotonic() - started) * 1000))
    ledger = Ledger(run_id, options.mode, 1, stages, actual_probe_calls, meter.calls, tokens, cost, wall_ms)
    writer.write_json(f"{run_rel}/ledger.json", ledger)
    manifest.update({"final_harness_id": chosen.harness_id, "decision": selection.decision.value})
    writer.write_json(f"{run_rel}/manifest.json", manifest)

    if invalid_count and not observations:
        exit_code = 5
        message = "E_NO_CANDIDATES: all created candidates were invalid"
    elif meter.calls and meter.successes == 0 and (options.proposer == "llm" or options.composer == "llm" or options.probe == "live"):
        exit_code = 7
        message = "E_LLM: no LLM call completed successfully"
    else:
        exit_code = 0
        message = selection.reason_code

    if not options.dry_run:
        library.record_run(writer, options.domain, chosen, [reflection.lesson] if reflection.lesson else [],
                           run_id=run_id, decision=selection.decision, rounds=(summary,), created_at=created_at)
        if options.apply and selection.decision is Decision.CANDIDATE:
            from .overlay import apply_overlay
            apply_overlay(writer, chosen)
    else:
        message = "dry-run: application simulated; active overlay untouched"
    return RunResult(run_id, selection.decision, run_rel, exit_code, message, chosen.harness_id)


def _build_sequential_candidate(
    writer: store.Store,
    current: Harness,
    after_inv: Harness,
    inv: Proposal,
    dom: Proposal,
    inv_error: harness.EditError | None,
    applied_inv: tuple[harness.AppliedEdit, ...],
    round_rel: str,
) -> tuple[Harness | None, tuple[harness.AppliedEdit, ...], str, str, bool]:
    candidate_id = "cand-01"
    candidate = None
    applied = ()
    diff_text = ""
    diff_rel = f"{round_rel}/diffs/{candidate_id}.diff"
    invalid = inv_error is not None
    if inv_error is not None:
        writer.write_json(f"{round_rel}/candidates/{candidate_id}.errors.json",
                          {"candidate_id": candidate_id,
                           "errors": [{"code": inv_error.code, "path": inv_error.path}]})
    if not invalid and (not inv.keep_base and inv.edits or not dom.keep_base and dom.edits):
        try:
            candidate, applied_dom = harness.apply_edits(after_inv, dom.edits if not dom.keep_base else ())
            applied = tuple(applied_inv) + tuple(applied_dom)
        except harness.EditError as exc:
            invalid = True
            writer.write_json(f"{round_rel}/candidates/{candidate_id}.errors.json",
                              {"candidate_id": candidate_id, "errors": [{"code": exc.code, "path": exc.path}]})
        if candidate is not None and not invalid:
            validation = validate.validate_harness(candidate)
            if not validation.valid:
                invalid = True
                writer.write_json(f"{round_rel}/candidates/{candidate_id}.errors.json",
                                  {"candidate_id": candidate_id,
                                   "errors": [{"code": error.code, "path": error.path, "detail": error.detail}
                                              for error in validation.errors]})
            else:
                harness.snapshot_round(writer, f"{round_rel}/candidates/{candidate_id}", candidate)
                diff_text = harness.diff_harness(current, candidate)
                _write_text(writer, diff_rel, diff_text)
    if invalid:
        candidate = None
    return candidate, applied, diff_text, diff_rel, invalid


def _compose_sequential_round(
    task: str,
    base_probe: ProbeResult,
    candidate_probe: ProbeResult | None,
    candidate_id: str,
    invalid: bool,
    options: EvolveOptions,
    client: LlmClient | None,
    meter: _MeteredClient,
    budget: Budget,
    round_number: int,
    history: Sequence[RoundSummary],
    seed: int,
    monotonic: Callable[[], float],
) -> tuple[Selection, int]:
    started = monotonic()
    if candidate_probe is None:
        reason = "no_usable_candidate" if invalid else "no_change"
        selection = Selection(round_number, Decision.KEEP_BASE, None, reason, reason, (), {})
    else:
        meter.stage = "compose"
        selection = judge.compose(
            task,
            base_probe,
            [(candidate_id, candidate_probe)],
            composer=options.composer,
            llm_client=client,
            k=1,
            round_number=round_number,
            prior_rounds=history,
            seed=seed,
            budget=budget,
        )
    elapsed = max(0, int((monotonic() - started) * 1000))
    return selection, elapsed


def _run_sequential(
    writer: store.Store,
    options: EvolveOptions,
    deps: Deps,
    task: str,
    task_id: str,
    base: Harness,
    run_id: str,
    created_at: str,
    rounds_count: int,
    client: LlmClient | None,
    meter: _MeteredClient,
    budget: Budget,
    runner,
) -> RunResult:
    run_rel = f"runs/{run_id}"
    started = deps.monotonic()
    manifest = {
        "schema_version": 1, "run_id": run_id, "task_id": task_id, "domain": options.domain,
        "created_at": created_at, "mode": options.mode, "k": None, "rounds": rounds_count,
        "base_harness_id": base.harness_id, "final_harness_id": base.harness_id,
        "decision": Decision.KEEP_BASE.value,
        "options": {"proposer": options.proposer, "probe": options.probe, "composer": options.composer,
                     "seed": options.seed, "model": options.model},
        "apply_requested": options.apply, "dry_run": options.dry_run,
    }
    writer.write_json(f"{run_rel}/manifest.json", manifest)
    _write_text(writer, f"{run_rel}/TASK.md", task)

    context = library.build_context(writer, options.domain)
    history: list[RoundSummary] = []
    selections: list[Selection] = []
    reflections = []
    lessons = []
    probe_results: list[ProbeResult] = []
    current = base
    last_selected = None
    invalid_count = 0
    valid_count = 0
    propose_ms = compose_ms = reflect_ms = 0
    probe_calls = probe_ms = propose_calls = compose_calls = 0

    for round_number in range(1, rounds_count + 1):
        round_rel = f"{run_rel}/rounds/round-{round_number:02d}"
        harness.snapshot_round(writer, f"{round_rel}/base", current)
        writer.write_json(f"{round_rel}/round.json", {
            "schema_version": 1, "round": round_number, "base_harness_id": current.harness_id,
        })
        pair: list[Proposal] = []
        proposal_ids = (f"p-{2 * round_number - 1:02d}", f"p-{2 * round_number:02d}")
        before_stage = deps.monotonic()
        inv_id = proposal_ids[0]
        if options.proposer == "template":
            inv = _propose_template(inv_id, round_number, Dimension.INV, current, options.domain, context, history)
        else:
            assert client is not None
            meter.stage = "propose"
            inv = _propose_llm(inv_id, round_number, Dimension.INV, task, current, context, history, client, budget)
        pair.append(inv)
        inv_error = None
        try:
            after_inv, applied_inv = harness.apply_edits(current, inv.edits if not inv.keep_base else ())
        except harness.EditError as exc:
            inv_error = exc
            after_inv, applied_inv = current, ()

        dom_id = proposal_ids[1]
        if options.proposer == "template":
            dom = _propose_template(dom_id, round_number, Dimension.DOM, after_inv, options.domain, context, history)
        else:
            assert client is not None
            meter.stage = "propose"
            dom = _propose_llm(dom_id, round_number, Dimension.DOM, task, after_inv, context, history, client, budget)
        pair.append(dom)
        propose_calls += 2
        propose_ms += max(0, int((deps.monotonic() - before_stage) * 1000))
        for proposed in pair:
            _write_proposal(writer, f"{round_rel}/proposals/{proposed.proposal_id}.json", proposed)

        candidate_id = "cand-01"
        candidate, applied, diff_text, diff_rel, invalid = _build_sequential_candidate(
            writer, current, after_inv, inv, dom, inv_error, tuple(applied_inv), round_rel,
        )
        if invalid:
            invalid_count += 1
        elif candidate is not None:
            valid_count += 1

        base_probe = _probe_one(writer, f"{round_rel}/probes/base.json", current, task, runner,
                                task_id=task_id, label="base", deps=deps, meter=meter)
        probe_results.append(base_probe)
        if base_probe.status is not Status.skipped:
            probe_calls += 1
        probe_ms += base_probe.wall_ms
        observations = []
        if candidate is not None:
            candidate_probe = _probe_one(writer, f"{round_rel}/probes/{candidate_id}.json", candidate, task, runner,
                                         task_id=task_id, label=candidate_id, deps=deps, meter=meter)
            probe_results.append(candidate_probe)
            if candidate_probe.status is not Status.skipped:
                probe_calls += 1
            probe_ms += candidate_probe.wall_ms
            observations.append(judge.CandidateObservation(candidate_id, candidate, applied, candidate_probe,
                                                           diff_rel, diff_text))

        selection, elapsed = _compose_sequential_round(
            task, base_probe, observations[0].probe if observations else None, candidate_id, invalid,
            options, client, meter, budget, round_number, history, options.seed, deps.monotonic,
        )
        compose_ms += elapsed
        compose_calls += len(selection.votes)

        reflect_started = deps.monotonic()
        meter.stage = "reflect"
        reflection = judge.reflect(current, base_probe, observations, selection, task_id=task_id,
                                   domain=options.domain, run_id=run_id,
                                   round_number=round_number, created_at=created_at)
        reflect_ms += max(0, int((deps.monotonic() - reflect_started) * 1000))
        selections.append(selection)
        reflections.append(reflection)
        if reflection.lesson:
            lessons.append(reflection.lesson)
        summary = RoundSummary(
            round_number,
            current.harness_id,
            (candidate_id,) if candidate is not None else (),
            selection.decision,
            selection.chosen_label,
            tuple(item for candidate_reflection in reflection.candidates
                  for item in candidate_reflection.classifications),
            reflection.lesson.lesson_id if reflection.lesson else None,
            selection.reason_code,
        )
        history.append(summary)
        writer.write_json(f"{round_rel}/selection.json", selection)
        writer.write_json(f"{round_rel}/reflection.json", reflection)
        _write_text(writer, f"{round_rel}/contrast.md",
                    _contrast(round_number, task_id, base_probe, observations, reflection, selection))
        if candidate is not None and selection.decision is Decision.CANDIDATE and selection.chosen_label == candidate_id:
            current = candidate
            last_selected = candidate_id

    global_decision = Decision.CANDIDATE if current.harness_id != base.harness_id else Decision.KEEP_BASE
    final_reason = "sequential_changed" if global_decision is Decision.CANDIDATE else selections[-1].reason_code
    final_selection = Selection(
        rounds_count, global_decision, last_selected if global_decision is Decision.CANDIDATE else None,
        selections[-1].rationale, final_reason, selections[-1].votes, selections[-1].mapping,
    )
    writer.write_json(f"{run_rel}/selection.json", _round_selection(final_selection, run_id, base, current, selections))
    writer.write_json(f"{run_rel}/reflection.json", {
        "schema_version": 1,
        "run_id": run_id,
        "lessons": [lesson.to_dict() for lesson in lessons],
        "rounds": [{"round": item.round, "no_entry": item.no_entry, "reason": item.reason,
                    "candidates": [candidate.to_dict() for candidate in item.candidates]}
                   for item in reflections],
    })
    tokens = Usage(meter.input_tokens, meter.output_tokens, meter.total_tokens)
    if meter.stage_calls["probe"] == 0:
        tokens = Usage(
            tokens.input_tokens + sum(item.usage.input_tokens for item in probe_results),
            tokens.output_tokens + sum(item.usage.output_tokens for item in probe_results),
            tokens.total_tokens + sum(item.usage.total_tokens for item in probe_results),
        )
    cost = None
    if deps.price_fn:
        value = deps.price_fn(options.model or "", tokens.input_tokens, tokens.output_tokens)
        if value is not None:
            cost = value
    ledger = Ledger(
        run_id, options.mode, 1,
        {"propose": StageUsage(propose_calls, propose_ms), "probe": StageUsage(probe_calls, probe_ms),
         "compose": StageUsage(compose_calls, compose_ms), "reflect": StageUsage(rounds_count, reflect_ms)},
        probe_calls, meter.calls, tokens, cost, max(0, int((deps.monotonic() - started) * 1000)),
    )
    writer.write_json(f"{run_rel}/ledger.json", ledger)
    manifest.update({"final_harness_id": current.harness_id, "decision": global_decision.value})
    writer.write_json(f"{run_rel}/manifest.json", manifest)

    if invalid_count and valid_count == 0:
        exit_code, message = 5, "E_NO_CANDIDATES: all created candidates were invalid"
    elif meter.calls and meter.successes == 0 and (options.proposer == "llm" or options.composer == "llm" or options.probe == "live"):
        exit_code, message = 7, "E_LLM: no LLM call completed successfully"
    else:
        exit_code, message = 0, final_reason
    if options.dry_run:
        message = "dry-run: application simulated; active overlay untouched"
    else:
        library.record_run(writer, options.domain, current, lessons, run_id=run_id,
                           decision=global_decision, rounds=tuple(history), created_at=created_at)
        if options.apply and global_decision is Decision.CANDIDATE:
            from .overlay import apply_overlay
            apply_overlay(writer, current)
    return RunResult(run_id, global_decision, run_rel, exit_code, message, current.harness_id)


def run_evolve(options: EvolveOptions, *, home: Path, deps: Deps | None = None) -> RunResult:
    deps = deps or Deps()
    placeholder = RunResult("run-000000000000", Decision.KEEP_BASE, "", 2, "E_USAGE", "h-000000000000")
    message, k, rounds = _options_error(options)
    if message:
        return RunResult(placeholder.run_id, placeholder.decision, "", 2, message, placeholder.final_harness_id)
    try:
        writer = store.Store(Path(home))
        run_id = "run-" + deps.token(12)
        if not valid_id("run", run_id):
            return RunResult(run_id, Decision.KEEP_BASE, "", 8, "E_IO: invalid run token", "h-000000000000")
        run_path = writer.run_dir(run_id)
        writer._confine(run_path, leaf="dir")
        if run_path.exists():
            return RunResult(run_id, Decision.KEEP_BASE, f"runs/{run_id}", 8, "E_IO: run already exists", "h-000000000000")
        try:
            task_path = Path(options.task_path)
            if task_path.stat().st_size > validate.MAX_TASK_BYTES:
                return RunResult(run_id, Decision.KEEP_BASE, "", 3, "E_TASK: task exceeds size limit", "h-000000000000")
            task = task_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return RunResult(run_id, Decision.KEEP_BASE, "", 3, "E_TASK: task unreadable", "h-000000000000")
        if not task.strip():
            return RunResult(run_id, Decision.KEEP_BASE, "", 3, "E_TASK: task is empty", "h-000000000000")
        if store.contains_secret(task):
            return RunResult(run_id, Decision.KEEP_BASE, "", 6, "E_SECRET: task rejected", "h-000000000000")
        task_id = task_path.stem.lower()
        if not valid_id("task", task_id):
            task_id = "default"
        if options.base_dir is not None:
            try:
                base = harness.load_bundle(options.base_dir)
            except (OSError, ValueError):
                return RunResult(run_id, Decision.KEEP_BASE, "", 4, "E_BASE: base bundle unreadable", "h-000000000000")
        else:
            from .overlay import load_active

            active_overlay = load_active(Path(home))
            base = Harness.make(active_overlay.text, {}) if active_overlay is not None else Harness.empty()
        base_validation = validate.validate_harness(base)
        if not base_validation.valid:
            codes = {error.code for error in base_validation.errors}
            code = 6 if codes == {"V_SECRET"} else 4
            return RunResult(run_id, Decision.KEEP_BASE, "", code, "E_SECRET: base rejected" if code == 6 else "E_BASE: base invalid", base.harness_id)
        created_at = _created_at(deps)
        client: LlmClient | None = None
        if options.proposer == "llm" or options.probe == "live" or options.composer == "llm":
            if deps.llm is not None:
                client = deps.llm
            elif deps.llm_factory is not None:
                client = deps.llm_factory(options.model)
            else:
                client = AuxLlmClient(model=options.model)
        meter = _MeteredClient(client) if client is not None else _MeteredClient(FakeLlm([]))
        budget = Budget(max_calls=options.max_llm_calls)
        if deps.runner is not None:
            runner = deps.runner
        elif options.probe == "live":
            assert client is not None
            runner = probe.LiveProbeRunner(meter)
        else:
            runner = probe.NullRunner()
        if options.mode == "sequential":
            return _run_sequential(writer, options, deps, task, task_id, base, run_id, created_at,
                                  rounds, meter if client is not None else None, meter, budget, runner)
        return _run_parallel(writer, options, deps, task, task_id, base, run_id, created_at,
                             k, meter if client is not None else None, meter, budget, runner)
    except store.SecretError:
        return RunResult(placeholder.run_id, Decision.KEEP_BASE, "", 6, "E_SECRET: rejected content", placeholder.final_harness_id)
    except (store.PathError, OSError):
        return RunResult(placeholder.run_id, Decision.KEEP_BASE, "", 8, "E_IO: confined storage error", placeholder.final_harness_id)
