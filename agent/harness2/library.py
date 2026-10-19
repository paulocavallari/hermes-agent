"""Domain-scoped Harness² library persistence and retrieval."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Sequence

from . import store
from .models import Decision, Harness, HarnessInfo, Lesson, LibraryContext, RoundSummary, valid_id


class LibraryError(ValueError):
    """A fail-closed library consistency error."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _domain(domain: str) -> str:
    if not valid_id("domain", domain):
        raise LibraryError("E_DOMAIN")
    return domain


def _write_text(writer: store.Store, rel: str, text: str) -> Path:
    # Keep all persistence mediated by Store while avoiding a duplicate writer
    # surface in the library module's AST contract.
    writer_write = writer.write_text
    return writer_write(rel, text)


def _domain_root(domain: str) -> str:
    return f"library/{_domain(domain)}"


def _read_optional(writer: store.Store, rel: str) -> str | None:
    try:
        return writer.read_text(rel)
    except FileNotFoundError:
        return None


def _all_harness_ids(writer: store.Store, domain: str) -> list[str]:
    parent = writer.root / _domain_root(domain) / "harnesses"
    return writer._list_dirs(parent, r"h-[0-9a-f]{12}")


def _all_lessons(writer: store.Store, domain: str) -> list[Lesson]:
    path = f"{_domain_root(domain)}/lessons.jsonl"
    text = _read_optional(writer, path)
    if text is None:
        return []
    lessons: list[Lesson] = []
    try:
        for line in text.splitlines():
            if not line:
                continue
            lessons.append(Lesson.from_dict(json.loads(line)))
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise LibraryError("E_LIBRARY", "invalid lessons.jsonl") from exc
    if any(lesson.domain != domain for lesson in lessons):
        raise LibraryError("E_LIBRARY", "lesson domain mismatch")
    return lessons


def _lesson_order(lessons: Sequence[Lesson]) -> list[Lesson]:
    return sorted(lessons, key=lambda lesson: (lesson.created_at, lesson.lesson_id), reverse=True)


def _same_lesson_content(left: Lesson, right: Lesson) -> bool:
    """Compare learned meaning while ignoring run-specific provenance paths."""
    return (
        left.lesson_id,
        left.cue,
        left.lesson,
        left.component,
        left.evidence.diff_lines,
        left.evidence.base_excerpt,
        left.evidence.candidate_excerpt,
    ) == (
        right.lesson_id,
        right.cue,
        right.lesson,
        right.component,
        right.evidence.diff_lines,
        right.evidence.base_excerpt,
        right.evidence.candidate_excerpt,
    )


def append_lessons(writer: store.Store, domain: str, lessons: Sequence[Lesson]) -> None:
    domain = _domain(domain)
    path = f"{_domain_root(domain)}/lessons.jsonl"
    existing_text = _read_optional(writer, path) or ""
    existing = _all_lessons(writer, domain)
    by_id = {lesson.lesson_id: lesson for lesson in existing}
    additions: list[Lesson] = []
    for lesson in lessons:
        if not isinstance(lesson, Lesson) or lesson.domain != domain:
            raise LibraryError("E_LIBRARY", "lesson domain mismatch")
        previous = by_id.get(lesson.lesson_id)
        if previous is not None:
            if not _same_lesson_content(previous, lesson):
                raise LibraryError("E_LESSON_COLLISION", lesson.lesson_id)
            continue
        by_id[lesson.lesson_id] = lesson
        additions.append(lesson)
    if additions:
        lines = [json.dumps(lesson.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                 for lesson in additions]
        _write_text(writer, path, existing_text + "\n".join(lines) + "\n")


def _created_at(writer: store.Store, run_id: str, lessons: Sequence[Lesson], explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        manifest = writer.read_json(f"runs/{run_id}/manifest.json")
    except FileNotFoundError:
        manifest = {}
    value = manifest.get("created_at")
    if isinstance(value, str) and value:
        return value
    if lessons:
        return lessons[0].created_at
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _experience_text(
    *,
    run_id: str,
    decision: Decision,
    created_at: str,
    rounds: Sequence[RoundSummary],
    lessons: Sequence[Lesson],
) -> str:
    summary = f"{decision.value} run {run_id}"
    lines = [
        "# Experience",
        summary,
        f"Created at: {created_at}",
        f"Decision: {decision.value}",
        "Rounds: " + (", ".join(f"{item.round:02d}:{item.reason_code}" for item in rounds) or "none"),
        "",
        "## Lessons",
    ]
    if lessons:
        for lesson in lessons:
            lines.append(f"- {lesson.cue}: {lesson.lesson} [{lesson.confidence.value}]")
    else:
        lines.append("- none")
    text = "\n".join(lines) + "\n"
    if len(text) > 2000:
        text = text[:1999].rstrip() + "\n"
    return text


def _experience_fields(text: str) -> tuple[str, str, str]:
    lines = text.splitlines()
    summary = next((line.strip() for line in lines[1:] if line.strip()), "")[:120]
    created = next((line.partition(":")[2].strip() for line in lines if line.startswith("Created at:")), "")
    decision = next((line.partition(":")[2].strip() for line in lines if line.startswith("Decision:")), "")
    if not created or decision not in {item.value for item in Decision}:
        raise LibraryError("E_LIBRARY", "invalid EXPERIENCE.md")
    return summary, created, decision


def _harness_info(writer: store.Store, domain: str, harness_id: str) -> HarnessInfo:
    prefix = f"{_domain_root(domain)}/harnesses/{harness_id}"
    bundle = writer.read_bundle(f"{prefix}/harness")
    experience = writer.read_text(f"{prefix}/EXPERIENCE.md")
    summary, created_at, _decision = _experience_fields(experience)
    skills = tuple(sorted(
        path.split("/")[1]
        for path in bundle.files
        if path.startswith("skills/") and path.endswith("/SKILL.md")
    ))
    return HarnessInfo(harness_id, created_at, skills, len(bundle.overlay), summary)


def list_harnesses(writer: store.Store, domain: str, *, limit: int = 20) -> list[HarnessInfo]:
    domain = _domain(domain)
    if type(limit) is not int or not 0 <= limit <= 20:
        raise ValueError("limit must be in 0..20")
    infos = [_harness_info(writer, domain, harness_id) for harness_id in _all_harness_ids(writer, domain)]
    infos.sort(key=lambda item: (item.created_at, item.harness_id), reverse=True)
    return infos[:limit]


def list_lessons(writer: store.Store, domain: str, *, limit: int = 20) -> list[Lesson]:
    domain = _domain(domain)
    if type(limit) is not int or not 0 <= limit <= 20:
        raise ValueError("limit must be in 0..20")
    return _lesson_order(_all_lessons(writer, domain))[:limit]


def write_index(writer: store.Store, domain: str) -> Path:
    domain = _domain(domain)
    rows = [_harness_info(writer, domain, harness_id) for harness_id in _all_harness_ids(writer, domain)]
    rows.sort(key=lambda item: (item.created_at, item.harness_id))
    lines = [
        f"# Harness² library — {domain}",
        "",
        "| harness_id | created_at | skills | overlay chars | decisions |",
        "|---|---|---|---:|---|",
    ]
    for item in rows:
        experience = writer.read_text(f"{_domain_root(domain)}/harnesses/{item.harness_id}/EXPERIENCE.md")
        _summary, _created, decision = _experience_fields(experience)
        lines.append(f"| {item.harness_id} | {item.created_at} | {', '.join(item.skills) or '-'} | {item.overlay_chars} | {decision} |")
    return _write_text(writer, f"{_domain_root(domain)}/INDEX.md", "\n".join(lines) + "\n")


def record_run(
    writer: store.Store,
    domain: str,
    final: Harness,
    lessons: Sequence[Lesson],
    *,
    run_id: str,
    decision: Decision | str,
    rounds: Sequence[RoundSummary],
    created_at: str | None = None,
) -> None:
    domain = _domain(domain)
    if not valid_id("run", run_id):
        raise LibraryError("E_RUN_ID")
    try:
        final_decision = Decision(decision)
    except (TypeError, ValueError):
        raise LibraryError("E_DECISION") from None

    created = _created_at(writer, run_id, lessons, created_at)
    if final_decision is Decision.CANDIDATE:
        harnesses = _all_harness_ids(writer, domain)
        prefix = f"{_domain_root(domain)}/harnesses/{final.harness_id}"
        if final.harness_id in harnesses:
            stored = writer.read_bundle(f"{prefix}/harness")
            if stored.content_sha256 != final.content_sha256:
                raise LibraryError("E_HARNESS_COLLISION", final.harness_id)
        else:
            writer.write_bundle(f"{prefix}/harness", final)
            _write_text(writer,
                f"{prefix}/EXPERIENCE.md",
                _experience_text(run_id=run_id, decision=final_decision, created_at=created,
                                 rounds=rounds, lessons=lessons),
            )
    append_lessons(writer, domain, lessons)
    write_index(writer, domain)


def build_context(writer: store.Store, domain: str) -> LibraryContext:
    domain = _domain(domain)
    all_lessons = _lesson_order(_all_lessons(writer, domain))
    all_harnesses = [_harness_info(writer, domain, item) for item in _all_harness_ids(writer, domain)]
    all_harnesses.sort(key=lambda item: (item.created_at, item.harness_id), reverse=True)
    recent_lessons = all_lessons[:5]
    recent_harnesses = all_harnesses[:20]
    omitted = max(0, len(all_lessons) - len(recent_lessons)) + max(0, len(all_harnesses) - len(recent_harnesses))
    # LibraryContext.render removes leading entries when enforcing its cap. Store the
    # retained window oldest-to-newest so truncation discards old context first.
    return LibraryContext(
        domain,
        tuple(reversed(recent_lessons)),
        tuple(reversed(recent_harnesses)),
        omitted,
    )


def list_domains(writer: store.Store) -> list[str]:
    return writer.list_domains()
