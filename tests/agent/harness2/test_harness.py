"""Bundle editing, rendering, and path-loading tests."""

import os
from pathlib import Path

import pytest

from agent.harness2 import harness, store
from agent.harness2.models import Edit, Harness, Op


def test_apply_add_replace_remove():
    base = Harness.make("base", {"skills/demo/SKILL.md": "old"})
    added, changes = harness.apply_edits(base, (Edit(Op.add, "skills/new/SKILL.md", "new skill"),))
    assert added.files["skills/new/SKILL.md"] == "new skill\n"
    assert changes[0].op is Op.add
    replaced, _ = harness.apply_edits(base, (Edit(Op.replace, "skills/demo/SKILL.md", "updated"),))
    assert replaced.files["skills/demo/SKILL.md"] == "updated\n"
    removed, _ = harness.apply_edits(base, (Edit(Op.remove, "skills/demo/SKILL.md", None),))
    assert "skills/demo/SKILL.md" not in removed.files


def test_apply_rejects_add_existing_remove_missing_and_missing_text():
    base = Harness.make("base")
    cases = (
        (Edit(Op.add, "overlay.md", "again"), "E_EDIT_EXISTS"),
        (Edit(Op.remove, "skills/no/SKILL.md", None), "E_EDIT_MISSING"),
        (Edit(Op.replace, "overlay.md", None), "E_EDIT_NO_TEXT"),
        (Edit(Op.remove, "overlay.md", "not omitted"), "E_EDIT_TEXT"),
        (Edit(Op.add, "skills/new/SKILL.md", ""), "E_EDIT_EMPTY"),
    )
    for edit, code in cases:
        with pytest.raises(harness.EditError) as exc:
            harness.apply_edits(base, (edit,))
        assert exc.value.code == code


def test_apply_rejects_bad_paths():
    base = Harness.empty()
    for path in ("../x", "/tmp/x", "skills/.hidden/SKILL.md", "C:/x", "overlay.md/../x"):
        with pytest.raises(harness.EditError):
            harness.apply_edits(base, (Edit(Op.add, path, "safe"),))


def test_render_system_prompt_is_deterministic_and_lists_support_files():
    bundle = Harness.make(
        "always do work",
        {
            "skills/beta/SKILL.md": "---\nname: beta\ndescription: Use beta\n---\nBody beta",
            "skills/beta/references/guide.md": "support text",
            "skills/alpha/SKILL.md": "---\nname: alpha\ndescription: Use alpha\n---\nBody alpha",
        },
    )
    rendered = harness.render_system_prompt(bundle)
    assert rendered == harness.render_system_prompt(bundle)
    assert rendered.index("## Skill: alpha") < rendered.index("## Skill: beta")
    assert "support text" not in rendered
    assert "skills/beta/references/guide.md" in rendered


def test_render_returns_none_over_cap():
    bundle = Harness.make("x" * 40)
    assert harness.render_system_prompt(bundle, cap=10) is None


def test_diff_is_unified_and_verbatim():
    before = Harness.make("before")
    after = Harness.make("after")
    diff = harness.diff_harness(before, after)
    assert "--- a/overlay.md" in diff
    assert "+++ b/overlay.md" in diff
    assert "@@" in diff
    assert "-before" in diff
    assert "+after" in diff


def test_snapshot_roundtrip_preserves_harness_id(home):
    with home() as selected:
        writer = store.Store(selected)
        original = Harness.make("base", {"skills/demo/SKILL.md": "body"})
        harness.snapshot_round(writer, "runs/run-0123456789ab/rounds/round-01/base", original)
        loaded = writer.read_bundle("runs/run-0123456789ab/rounds/round-01/base")
        assert loaded.harness_id == original.harness_id


def test_frontmatter_parsed_and_body_extracted():
    metadata, body = harness.parse_skill_frontmatter(
        "---\nname: demo\ndescription: Use this\n---\n\nBody text\n"
    )
    assert metadata == {"name": "demo", "description": "Use this"}
    assert body == "Body text\n"


def test_edits_are_normalized_before_hashing():
    base = Harness.empty()
    first, _ = harness.apply_edits(base, (Edit(Op.replace, "overlay.md", "one\r\n\r\n\r\ntwo  \r\n"),))
    second, _ = harness.apply_edits(base, (Edit(Op.replace, "overlay.md", "one\n\ntwo\n"),))
    assert first.overlay == "one\n\ntwo\n"
    assert first.harness_id == second.harness_id


def test_load_bundle_walks_real_dirs_with_scandir(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "skills/demo/references").mkdir(parents=True)
    (bundle / "overlay.md").write_text("hello\n", encoding="utf-8")
    (bundle / "skills/demo/SKILL.md").write_text("body\n", encoding="utf-8")
    (bundle / "skills/demo/references/guide.md").write_text("guide\n", encoding="utf-8")
    loaded = harness.load_bundle(bundle)
    assert loaded.overlay == "hello\n"
    assert loaded.files["skills/demo/references/guide.md"] == "guide\n"


def test_load_bundle_rejects_symlinked_root_and_ancestor(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "overlay.md").write_text("secret sentinel\n", encoding="utf-8")
    link = tmp_path / "linked-parent"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(store.PathError):
        harness.load_bundle(link / "bundle")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(store.PathError):
        harness.load_bundle(root_link)
    assert (outside / "overlay.md").read_text(encoding="utf-8") == "secret sentinel\n"


def test_load_bundle_rejects_dotdot_before_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("unchanged", encoding="utf-8")
    linked = tmp_path / "linked-parent"
    linked.symlink_to(outside, target_is_directory=True)
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/overlay.md").write_text("valid cwd bundle", encoding="utf-8")
    before = (tmp_path / "bundle/overlay.md").read_bytes()
    with pytest.raises(store.PathError):
        harness.load_bundle("linked-parent/../bundle")
    assert (tmp_path / "bundle/overlay.md").read_bytes() == before
    assert (outside / "sentinel").read_text(encoding="utf-8") == "unchanged"
