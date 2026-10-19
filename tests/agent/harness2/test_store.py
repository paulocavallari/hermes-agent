"""Confinement, redaction, and persistence tests for the Harness² store."""

import ast
import os
import shutil
from pathlib import Path

import pytest

from agent.harness2 import store
from agent.harness2.models import Harness


def test_safe_relpath_rejects_traversal_absolute_and_colon():
    for path in ("../x", "/tmp/x", "C:/x", "a:b", "a\\b", ".hidden/x", ""):
        with pytest.raises(store.PathError):
            store.assert_safe_relpath(path)


def test_safe_relpath_accepts_generated_components():
    assert store.assert_safe_relpath("runs/run-0123456789ab/manifest.json") == "runs/run-0123456789ab/manifest.json"
    assert store.assert_safe_relpath("active/overlay.md") == "active/overlay.md"


@pytest.mark.parametrize("secret", [
    "SERVICE_API_KEY=unit-test-secret-value",
    '{"api_key": "abc123"}',
    "password: abc123",
    "https://abc123@example.test/path",
])
def test_write_text_rejects_secret_in_env_json_yaml_url(home, secret):
    with home() as selected:
        writer = store.Store(selected)
        with pytest.raises(store.SecretError):
            writer.write_text("runs/run-0123456789ab/secret.txt", secret)
        assert not (writer.root / "runs/run-0123456789ab/secret.txt").exists()


def test_scanner_and_redactor_agree_on_corpus():
    sensitive = (
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "SERVICE_API_KEY=unit-test-secret-value",
        '{"api_key":"abc123"}',
        "https://abc123@example.test/path",
        "https://example.test/?token=abc123",
    )
    for text in sensitive:
        assert store.contains_secret(text)
        redacted = store.redact_strict(text)
        assert redacted != text
        assert not store.scan_secrets(redacted)
    for text in ("token budget is limited", "secret sauce", "MAX_OVERLAY_CHARS=4000"):
        assert not store.contains_secret(text)


def test_accepted_bundle_roundtrip_is_byte_identical(home):
    with home() as selected:
        writer = store.Store(selected)
        harness = Harness.make("text\r\n", {"skills/demo/SKILL.md": "content\r\n"})
        writer.write_bundle("runs/run-0123456789ab/base", harness)
        restored = writer.read_bundle("runs/run-0123456789ab/base")
        assert restored.content_sha256 == harness.content_sha256
        assert restored.overlay == harness.overlay
        assert restored.files == harness.files


def test_secret_never_reaches_disk(home):
    with home() as selected:
        writer = store.Store(selected)
        with pytest.raises(store.SecretError):
            writer.write_text("active/overlay.md", "token=unit-test-secret-value")
        assert not (writer.root / "active/overlay.md").exists()


def test_atomic_write_leaves_no_tmp_file(home):
    with home() as selected:
        writer = store.Store(selected)
        writer.write_text("active/overlay.md", "safe content")
        active = writer.root / "active"
        assert sorted(path.name for path in active.iterdir()) == ["overlay.md"]


def test_write_json_is_sorted_utf8_and_no_encoding_kwarg(home, monkeypatch):
    with home() as selected:
        writer = store.Store(selected)
        calls = []
        original = store.atomic_write_text

        def observed(path, text, **kwargs):
            calls.append((Path(path), text, kwargs))
            return original(path, text, **kwargs)

        monkeypatch.setattr(store, "atomic_write_text", observed)
        writer.write_json("runs/run-0123456789ab/meta.json", {"z": "ç", "a": 1})
        target, text, kwargs = calls[0]
        assert target.read_text(encoding="utf-8") == text
        assert text.index('"a"') < text.index('"z"')
        assert "ç" in text
        assert "encoding" not in kwargs


def test_read_bundle_rejects_symlink_and_non_regular(home, tmp_path):
    with home() as selected:
        writer = store.Store(selected)
        writer.write_bundle("runs/run-0123456789ab/base", Harness.empty())
        base = writer.root / "runs/run-0123456789ab/base"
        external = tmp_path / "external.txt"
        external.write_text("sentinel", encoding="utf-8")
        (base / "leak.txt").symlink_to(external)
        with pytest.raises(store.PathError):
            writer.read_bundle("runs/run-0123456789ab/base")
        (base / "leak.txt").unlink()
        os.mkfifo(base / "pipe")
        with pytest.raises(store.PathError):
            writer.read_bundle("runs/run-0123456789ab/base")


def test_read_bundle_enforces_size_caps(home, tmp_path):
    with home() as selected:
        writer = store.Store(selected)
        bundle = writer.root / "runs/run-0123456789ab/base"
        bundle.mkdir(parents=True)
        (bundle / "overlay.md").write_text("x" * 65537, encoding="utf-8")
        with pytest.raises(store.PathError):
            writer.read_bundle("runs/run-0123456789ab/base")


def test_read_plain_tree_rejects_linked_parent_bundle(home, tmp_path, monkeypatch):
    with home() as selected:
        writer = store.Store(selected)
        outside = tmp_path / "outside"
        bundle = outside / "bundle"
        bundle.mkdir(parents=True)
        sentinel = bundle / "overlay.md"
        sentinel.write_text("outside sentinel", encoding="utf-8")
        link = tmp_path / "linked-parent"
        link.symlink_to(outside, target_is_directory=True)
        opened = []
        original = os.open

        def observed(path, *args, **kwargs):
            opened.append(Path(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", observed)
        with pytest.raises(store.PathError):
            store.read_plain_tree(link / "bundle", caps=store.DEFAULT_TREE_CAPS, anchor=tmp_path)
        assert not any(path.is_relative_to(outside) for path in opened)
        assert sentinel.read_text(encoding="utf-8") == "outside sentinel"
        assert sorted(path.name for path in bundle.iterdir()) == ["overlay.md"]


def test_read_plain_tree_requires_explicit_anchor(tmp_path):
    with pytest.raises(TypeError):
        store.read_plain_tree(tmp_path, caps=store.DEFAULT_TREE_CAPS)


def test_confine_ancestors_rejects_links_and_is_read_only(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(store.PathError):
        store.confine_ancestors(link / "not-created", anchor=tmp_path)
    assert not (outside / "not-created").exists()


def test_read_bundle_rejects_root_link_and_linked_runs(home, tmp_path):
    with home() as selected:
        writer = store.Store(selected)
        writer.write_bundle("runs/run-0123456789ab/base", Harness.empty())
        base = writer.root / "runs/run-0123456789ab/base"
        moved = writer.root / "runs/run-0123456789ab/moved"
        base.rename(moved)
        base.symlink_to(moved, target_is_directory=True)
        with pytest.raises(store.PathError):
            writer.read_bundle("runs/run-0123456789ab/base")
        base.unlink()
        runs = writer.root / "runs"
        shutil.rmtree(runs)
        outside = tmp_path / "outside-runs"
        outside.mkdir()
        (outside / "marker").write_text("sentinel", encoding="utf-8")
        runs.symlink_to(outside, target_is_directory=True)
        with pytest.raises(store.PathError):
            writer.read_bundle("runs/run-0123456789ab/moved")
        assert (outside / "marker").read_text(encoding="utf-8") == "sentinel"


def test_store_refuses_symlinked_root_runs_active(tmp_path):
    home_path = tmp_path / "home"
    home_path.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (home_path / "harness2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(store.PathError):
        store.Store(home_path)
    safe_home = tmp_path / "safe-home"
    safe_home.mkdir()
    writer = store.Store(safe_home)
    for name in ("runs", "active"):
        target = outside / name
        target.mkdir(exist_ok=True)
        (writer.root / name).symlink_to(target, target_is_directory=True)
        with pytest.raises(store.PathError):
            writer.write_text(f"{name}/x.txt", "safe")
        (writer.root / name).unlink()


def test_only_store_writes():
    package = Path(__file__).resolve().parents[3] / "agent/harness2"
    forbidden = {"write_text", "write_bytes", "atomic_write_text", "atomic_write_bytes", "atomic_json_write"}
    for path in package.glob("*.py"):
        if path.name in {"store.py", "__init__.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden, f"writer outside Store: {path.name}:{node.lineno}"


def test_read_errors_propagate_permission_error(home, monkeypatch):
    with home() as selected:
        writer = store.Store(selected)
        writer.write_text("active/overlay.md", "safe")

        def denied(*_args, **_kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(os, "open", denied)
        with pytest.raises(PermissionError):
            writer.read_text("active/overlay.md")
