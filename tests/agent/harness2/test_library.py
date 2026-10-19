"""Persistence and same-domain retrieval tests for Harness²."""

import json

import pytest

from agent.harness2 import library, store
from agent.harness2.models import Confidence, Decision, Evidence, Harness, Lesson


def _lesson(index: int, domain: str = "demo", *, text: str | None = None) -> Lesson:
    created = f"2025-02-01T00:{index:02d}:00Z"
    return Lesson(
        f"l-{index:08x}", domain, f"run-{index:012x}", 1, "cand-01", "overlay.md",
        f"editing overlay.md for {domain} tasks", text or f"lesson marker {index}",
        Evidence("diffs/cand-01.diff", ("+new",), "probes/base.json", "base output",
                 "probes/cand-01.json", "candidate output"), Confidence.high, created,
    )


def _harness(index: int = 1, *, support: bool = False) -> Harness:
    files = {}
    if support:
        files = {
            "skills/demo/SKILL.md": "---\nname: demo\ndescription: Use demo\n---\nSkill body",
            "skills/demo/references/guide.md": "SUPPORT_CONTENT_MUST_NOT_BE_INLINED",
        }
    return Harness.make(f"overlay version {index}", files)


def _record(writer, domain="demo", *, index=1, lessons=(), decision=Decision.CANDIDATE, created_at=None):
    return library.record_run(
        writer, domain, _harness(index, support=index == 1), tuple(lessons),
        run_id=f"run-{index:012x}", decision=decision, rounds=(), created_at=created_at,
    )


def test_record_writes_full_tree_including_support_files(tmp_path):
    writer = store.Store(tmp_path)
    bundle = _harness(support=True)
    library.record_run(writer, "demo", bundle, (), run_id="run-000000000001",
                       decision=Decision.CANDIDATE, rounds=(), created_at="2025-01-01T00:00:00Z")
    root = "library/demo/harnesses/{}/harness".format(bundle.harness_id)
    assert writer.read_bundle(root).content_sha256 == bundle.content_sha256
    assert writer.read_text(f"{root}/skills/demo/references/guide.md") == "SUPPORT_CONTENT_MUST_NOT_BE_INLINED\n"


def test_record_writes_experience_and_index(tmp_path):
    writer = store.Store(tmp_path)
    _record(writer, index=1, lessons=(_lesson(1),), created_at="2025-01-01T00:00:00Z")
    bundle = _harness(1, support=True)
    experience = writer.read_text(f"library/demo/harnesses/{bundle.harness_id}/EXPERIENCE.md")
    index = writer.read_text("library/demo/INDEX.md")
    assert "run-000000000001" in experience
    assert "CANDIDATE" in experience
    assert "editing overlay.md for demo tasks" in experience
    assert "| harness_id | created_at | skills | overlay chars | decisions |" in index
    assert bundle.harness_id in index
    assert "demo" in index


def test_record_is_idempotent_for_same_hash(tmp_path):
    writer = store.Store(tmp_path)
    bundle = _harness(1)
    library.record_run(writer, "demo", bundle, (), run_id="run-000000000001",
                       decision=Decision.CANDIDATE, rounds=(), created_at="2025-01-01T00:00:00Z")
    path = f"library/demo/harnesses/{bundle.harness_id}/EXPERIENCE.md"
    before = writer.read_text(path)
    library.record_run(writer, "demo", bundle, (), run_id="run-000000000002",
                       decision=Decision.CANDIDATE, rounds=(), created_at="2026-01-01T00:00:00Z")
    assert writer.read_text(path) == before
    assert writer.read_bundle(f"library/demo/harnesses/{bundle.harness_id}/harness").content_sha256 == bundle.content_sha256


def test_record_rejects_hash_collision(tmp_path):
    writer = store.Store(tmp_path)
    bundle = _harness(1)
    library.record_run(writer, "demo", bundle, (), run_id="run-000000000001",
                       decision=Decision.CANDIDATE, rounds=(), created_at="2025-01-01T00:00:00Z")
    root = f"library/demo/harnesses/{bundle.harness_id}/harness"
    writer.write_bundle(root, _harness(2))
    with pytest.raises(library.LibraryError) as exc:
        library.record_run(writer, "demo", bundle, (), run_id="run-000000000002",
                           decision=Decision.CANDIDATE, rounds=())
    assert exc.value.code == "E_HARNESS_COLLISION"


def test_lessons_jsonl_appends_and_dedupes(tmp_path):
    writer = store.Store(tmp_path)
    lesson = _lesson(1)
    library.append_lessons(writer, "demo", (lesson, lesson))
    library.append_lessons(writer, "demo", (lesson,))
    rows = [json.loads(line) for line in writer.read_text("library/demo/lessons.jsonl").splitlines()]
    assert len(rows) == 1
    assert rows[0]["lesson_id"] == lesson.lesson_id
    assert library.list_lessons(writer, "demo") == [lesson]


def test_retrieval_is_domain_scoped(tmp_path):
    writer = store.Store(tmp_path)
    _record(writer, "alpha", index=1, lessons=(_lesson(1, "alpha"),), created_at="2025-01-01T00:00:00Z")
    _record(writer, "beta", index=2, lessons=(_lesson(2, "beta"),), created_at="2025-01-02T00:00:00Z")
    alpha = library.build_context(writer, "alpha")
    assert alpha.domain == "alpha"
    assert {item.domain for item in alpha.lessons} == {"alpha"}
    assert all(item.domain == "alpha" for item in library.list_lessons(writer, "alpha"))
    assert all(item.harness_id == _harness(1, support=True).harness_id for item in alpha.inventory)


def test_retrieval_caps_five_lessons_and_twenty_harnesses(tmp_path):
    writer = store.Store(tmp_path)
    for index in range(1, 26):
        created = f"2025-02-01T00:{index:02d}:00Z"
        bundle = Harness.make(f"unique bundle {index}")
        library.record_run(writer, "demo", bundle, (_lesson(index),),
                           run_id=f"run-{index:012x}", decision=Decision.CANDIDATE,
                           rounds=(), created_at=created)
    assert len(library.list_lessons(writer, "demo", limit=20)) == 20
    assert len(library.list_harnesses(writer, "demo", limit=20)) == 20
    context = library.build_context(writer, "demo")
    assert len(context.lessons) == 5
    assert len(context.inventory) == 20
    assert context.omitted == 25


def test_rendered_context_never_inlines_harness_files(tmp_path):
    writer = store.Store(tmp_path)
    bundle = _harness(support=True)
    library.record_run(writer, "demo", bundle, (), run_id="run-000000000001",
                       decision=Decision.CANDIDATE, rounds=(), created_at="2025-01-01T00:00:00Z")
    rendered = library.build_context(writer, "demo").render()
    assert "SUPPORT_CONTENT_MUST_NOT_BE_INLINED" not in rendered
    assert "skills=demo" in rendered
    assert "Full harness files are on disk" in rendered


def test_context_hard_cap_drops_inventory_then_oldest_lessons(tmp_path):
    writer = store.Store(tmp_path)
    for index in range(1, 6):
        bundle = Harness.make(f"inventory marker {index}")
        library.record_run(writer, "demo", bundle, (_lesson(index, text=f"lesson-{index} " + "x" * 1000),),
                           run_id=f"run-{index:012x}", decision=Decision.CANDIDATE,
                           rounds=(), created_at=f"2025-02-01T00:0{index}:00Z")
    context = library.build_context(writer, "demo")
    rendered = context.render(cap_chars=4000)
    assert len(rendered) <= 4000
    assert "lesson-5" in rendered
    assert "lesson-1" not in rendered
    assert "run-000000000005" not in rendered
