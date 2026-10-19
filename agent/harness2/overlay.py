"""Opt-in active-overlay loading and confined one-level apply/rollback."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from . import store
from .models import Harness

MAX_ACTIVE_CHARS = 4000


@dataclass(frozen=True, slots=True)
class ActiveOverlay:
    text: str
    chars: int
    path: Path


class ApplyError(ValueError):
    """An overlay failed validation before apply or rollback."""

    code = "E_APPLY"

    def __init__(self, detail: str = "invalid overlay"):
        super().__init__(f"E_APPLY: {detail}")


def _valid_overlay(text: str) -> bool:
    try:
        return store.safe_overlay_text(Harness.make(text, {}).overlay, max_chars=MAX_ACTIVE_CHARS)
    except (TypeError, ValueError):
        return False


def _write_text(writer: store.Store, rel: str, text: str):
    writer_write = writer.write_text
    return writer_write(rel, text)


def load_active(home: Path) -> ActiveOverlay | None:
    """Read a valid, bounded active overlay without following any symlink."""
    target = Path(home) / "harness2" / "active" / "overlay.md"
    try:
        store.confine_ancestors(target, anchor=Path(home))
        if not stat.S_ISREG(os.lstat(target).st_mode):
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(target, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                fd = -1
                text = stream.read(MAX_ACTIVE_CHARS + 1)
        finally:
            if fd >= 0:
                os.close(fd)
        if len(text) > MAX_ACTIVE_CHARS or not _valid_overlay(text):
            return None
        return ActiveOverlay(text, len(text), target)
    except Exception:  # health: allow BLE001 -- malformed or unreadable active state is rejected fail-closed
        return None


def load_injected_overlay(home: Path, *, gate: bool) -> str | None:
    """Return active text only when the caller's explicit injection gate is on."""
    if not gate:
        return None
    active = load_active(home)
    return active.text if active is not None and active.text else None


def apply_overlay(writer: store.Store, harness: Harness) -> None:
    """Apply only overlay.md, retaining one previous overlay for rollback."""
    if not isinstance(harness, Harness) or not _valid_overlay(harness.overlay):
        raise ApplyError()
    active_rel = "active/overlay.md"
    previous_rel = "active/overlay.prev"
    try:
        previous_text = writer.read_text(active_rel)
    except FileNotFoundError:
        writer.unlink(previous_rel)
    else:
        _write_text(writer, previous_rel, previous_text)
    _write_text(writer, active_rel, harness.overlay)


def rollback_overlay(writer: store.Store) -> bool:
    """Restore the one-level backup; return False when no backup exists."""
    previous_rel = "active/overlay.prev"
    try:
        previous_text = writer.read_text(previous_rel)
    except FileNotFoundError:
        return False
    if not _valid_overlay(previous_text):
        raise ApplyError("invalid backup")
    _write_text(writer, "active/overlay.md", previous_text)
    writer.unlink(previous_rel)
    return True
