"""Portable Harness² bundle operations."""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import store
from .models import Edit, Harness, Op, normalize_text


@dataclass(frozen=True, slots=True)
class AppliedEdit:
    op: Op
    path: str
    before_chars: int
    after_chars: int


class EditError(ValueError):
    def __init__(self, code: str, path: str):
        self.code = code
        self.path = path
        super().__init__(f"{code}: {path}")


def _bundle_path(path: str) -> str:
    try:
        normalized = store.assert_safe_relpath(path)
    except store.PathError:
        raise EditError("E_EDIT_PATH", path) from None
    if normalized != "overlay.md":
        try:
            Harness.make("", {normalized: ""})
        except (TypeError, ValueError):
            raise EditError("E_EDIT_PATH", path) from None
    return normalized


def apply_edits(harness: Harness, edits: Iterable[Edit]) -> tuple[Harness, tuple[AppliedEdit, ...]]:
    overlay = harness.overlay
    files = dict(harness.files)
    applied: list[AppliedEdit] = []
    for edit in edits:
        if not isinstance(edit.op, Op):
            raise EditError("E_EDIT_OP", edit.path)
        path = _bundle_path(edit.path)
        exists = path == "overlay.md" or path in files
        old = overlay if path == "overlay.md" else files.get(path)
        if edit.op is Op.add:
            if exists:
                raise EditError("E_EDIT_EXISTS", path)
            if edit.new_text is None:
                raise EditError("E_EDIT_NO_TEXT", path)
            if not edit.new_text:
                raise EditError("E_EDIT_EMPTY", path)
            value = normalize_text(edit.new_text)
            files[path] = value
            applied.append(AppliedEdit(edit.op, path, 0, len(value)))
        elif edit.op is Op.replace:
            if not exists:
                raise EditError("E_EDIT_MISSING", path)
            if edit.new_text is None:
                raise EditError("E_EDIT_NO_TEXT", path)
            value = normalize_text(edit.new_text)
            if path == "overlay.md":
                overlay = value
            else:
                files[path] = value
            applied.append(AppliedEdit(edit.op, path, len(old or ""), len(value)))
        elif edit.op is Op.remove:
            if not exists:
                raise EditError("E_EDIT_MISSING", path)
            if edit.new_text is not None:
                raise EditError("E_EDIT_TEXT", path)
            if path == "overlay.md":
                overlay = ""
            else:
                del files[path]
            applied.append(AppliedEdit(edit.op, path, len(old or ""), 0))
        else:
            raise EditError("E_EDIT_OP", path)
    try:
        result = Harness.make(overlay, files)
    except (TypeError, ValueError):
        raise EditError("E_EDIT_PATH", "bundle") from None
    return result, tuple(applied)


def render_system_prompt(harness: Harness, *, cap: int = 24_000) -> str | None:
    parts: list[str] = []
    if harness.overlay:
        parts.append(harness.overlay.rstrip("\n"))
    skill_names = sorted(
        path.split("/")[1]
        for path in harness.files
        if path.startswith("skills/") and path.endswith("/SKILL.md")
    )
    for name in skill_names:
        skill_path = f"skills/{name}/SKILL.md"
        parts.append(f"## Skill: {name}\n{harness.files[skill_path].rstrip()}" )
        supports = sorted(path for path in harness.files if path.startswith(f"skills/{name}/") and path != skill_path)
        if supports:
            parts.append("Support files: " + ", ".join(supports))
    rendered = "\n\n".join(parts)
    if len(rendered) > cap:
        return None
    return rendered


def diff_harness(before: Harness, after: Harness) -> str:
    old_files = {"overlay.md": before.overlay, **before.files}
    new_files = {"overlay.md": after.overlay, **after.files}
    chunks: list[str] = []
    for path in sorted(set(old_files) | set(new_files)):
        old = old_files.get(path, "")
        new = new_files.get(path, "")
        if old == new:
            continue
        chunks.extend(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        ))
    return "".join(chunks)


def parse_skill_frontmatter(text: str) -> tuple[dict[str, str], str]:
    normalized = normalize_text(text)
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        raise ValueError("missing frontmatter")
    closing = next((index for index in range(1, len(lines)) if lines[index].rstrip("\n") == "---"), None)
    if closing is None:
        raise ValueError("unterminated frontmatter")
    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        stripped = line.rstrip("\n")
        if not stripped or ":" not in stripped:
            raise ValueError("invalid frontmatter line")
        key, value = stripped.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or key in metadata:
            raise ValueError("invalid frontmatter key")
        metadata[key] = value
    body = "".join(lines[closing + 1:]).lstrip("\n")
    return metadata, body


def snapshot_round(writer: store.Store, rel_dir: str | Path, harness: Harness) -> None:
    writer.write_bundle(rel_dir, harness)


def load_bundle(path: str | Path) -> Harness:
    raw = Path(path)
    anchor = Path(raw.anchor) if raw.is_absolute() else Path.cwd()
    # Validate the caller's original lexical path before abspath can erase '..'.
    store.confine_ancestors(raw, anchor=anchor)
    root = Path(os.path.abspath(os.fspath(raw)))
    tree = store.read_plain_tree(root, caps=store.DEFAULT_TREE_CAPS, anchor=anchor)
    if "overlay.md" not in tree:
        raise store.PathError("E_IO", root / "overlay.md")
    overlay = tree.pop("overlay.md")
    try:
        return Harness.make(overlay, tree)
    except (TypeError, ValueError):
        raise store.PathError("E_PATH", root) from None
