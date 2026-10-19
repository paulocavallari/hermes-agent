"""Fail-closed validation for harness bundles and reflection evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from . import store
from .harness import parse_skill_frontmatter
from .models import Evidence, Harness, Op, ProbeResult

MAX_OVERLAY_CHARS = 4000
MAX_FILES = 64
MAX_SKILL_BYTES = 65536
MAX_SUPPORT_BYTES = 65536
MAX_TOTAL_BYTES = 1_048_576
MAX_RENDER_CHARS = 24000
MAX_TASK_BYTES = 65536
_SUPPORT_DIRS = {"references", "templates", "assets", "scripts"}
_MD_LINK = re.compile(r"\]\(\s*(?P<dest><[^>\n]*>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_DIR_TOKEN = re.compile(r"(?P<ref>[^\s()\[\]`'\"]*?(?:references|templates|assets|scripts)[\\/][^\s()\[\]`'\"]*)")
_ABSOLUTE_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class ValidationError:
    code: str
    path: str
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: tuple[ValidationError, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


class EvidenceError(ValueError):
    def __init__(self, code: str = "evidence_mismatch"):
        self.code = code
        super().__init__(code)


def _error(errors: list[ValidationError], code: str, path: str, detail: str = "") -> None:
    errors.append(ValidationError(code, path, detail[:200]))


def _extract_references(text: str) -> tuple[str, ...]:
    if not any(name in text for name in _SUPPORT_DIRS):
        return ()
    candidates: list[tuple[int, str]] = []
    for match in _MD_LINK.finditer(text):
        token = match.group("dest")
        if any(name in token for name in _SUPPORT_DIRS):
            candidates.append((match.start("dest"), token.rstrip(".,;")))
    for match in _DIR_TOKEN.finditer(text):
        token = match.group("ref")
        if any(name in token for name in _SUPPORT_DIRS):
            candidates.append((match.start("ref"), token.rstrip(".,;")))
    result: list[str] = []
    seen: set[str] = set()
    for _, token in sorted(candidates, key=lambda item: item[0]):
        if token and token not in seen:
            result.append(token)
            seen.add(token)
    return tuple(result)


def _classify_reference(ref: str, skill_name: str, files: dict[str, str]) -> str | None:
    if ref.startswith("/") or ref.startswith("\\") or _ABSOLUTE_DRIVE.match(ref):
        return "V_REF_ABSOLUTE"
    if ".." in re.split(r"[\\/]", ref):
        return "V_REF_TRAVERSAL"
    if any(char in ref for char in "{}<>*$\\"):
        return "V_REF_PLACEHOLDER"
    normalized = ref.removeprefix("./")
    first = re.split(r"[\\/]", normalized, maxsplit=1)[0]
    if first not in _SUPPORT_DIRS:
        return "V_REF_INVALID"
    target = f"skills/{skill_name}/{normalized}"
    if target not in files:
        return "V_REF_MISSING"
    return None


def _has_control(text: str) -> bool:
    return store.contains_control(text)


def _valid_path(path: str) -> tuple[str | None, str | None]:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or ":" in path:
        return None, None
    parts = path.split("/")
    if any(not part or part in (".", "..") or part.startswith(".") for part in parts):
        return None, None
    if parts[0] != "skills" or len(parts) < 3:
        return None, None
    skill = parts[1]
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", skill):
        return None, None
    if len(parts) == 3 and parts[2] == "SKILL.md":
        return skill, "skill"
    if len(parts) >= 4 and parts[2] in _SUPPORT_DIRS:
        return skill, "support"
    return None, None


def _validate_bundle_limits(harness: Harness, errors: list[ValidationError]) -> set[str]:
    overlay = harness.overlay
    files = harness.files
    if len(overlay) > MAX_OVERLAY_CHARS:
        _error(errors, "V_OVERLAY_SIZE", "overlay.md", "overlay exceeds character limit")
    if _has_control(overlay):
        _error(errors, "V_OVERLAY_CONTROL", "overlay.md", "control character")
    if len(files) > MAX_FILES:
        _error(errors, "V_TOO_MANY_FILES", "", "too many files")
    total = len(overlay.encode("utf-8"))
    skill_names: set[str] = set()
    for path, text in files.items():
        if not isinstance(path, str) or not isinstance(text, str):
            _error(errors, "V_UNKNOWN_FILE", str(path), "invalid path or content type")
            continue
        skill_name, kind = _valid_path(path)
        if skill_name is None:
            code = "V_PATH" if any(x in path for x in ("..", ":", "\\")) or path.startswith("/") else "V_UNKNOWN_FILE"
            _error(errors, code, path, "invalid harness path")
            continue
        skill_names.add(skill_name)
        file_size = len(text.encode("utf-8"))
        if kind == "skill" and file_size > MAX_SKILL_BYTES:
            _error(errors, "V_SKILL_BYTES", path, "skill file exceeds byte limit")
        elif kind == "support" and file_size > MAX_SUPPORT_BYTES:
            _error(errors, "V_FILE_SIZE", path, "support file exceeds byte limit")
        if _has_control(text):
            _error(errors, "V_OVERLAY_CONTROL", path, "control character")
        total += file_size
    if total > MAX_TOTAL_BYTES:
        _error(errors, "V_TOTAL_SIZE", "", "bundle exceeds total byte limit")
    return skill_names


def _validate_skill_metadata(
    overlay: str,
    files: dict[str, str],
    skill_names: set[str],
    trigger_policy: str,
    errors: list[ValidationError],
) -> None:
    for name in sorted(skill_names):
        skill_path = f"skills/{name}/SKILL.md"
        text = files.get(skill_path)
        if text is None:
            _error(errors, "V_SKILL_NO_SKILL_MD", f"skills/{name}", "support directory needs root SKILL.md")
            continue
        try:
            metadata, body = parse_skill_frontmatter(text)
        except (TypeError, ValueError):
            _error(errors, "V_FRONTMATTER", skill_path, "invalid frontmatter")
            metadata, body = {}, ""
        if metadata.get("name") != name:
            _error(errors, "V_NAME_MISMATCH", skill_path, "frontmatter name must match directory")
        description = metadata.get("description", "")
        if not description or len(description) > 60:
            _error(errors, "V_DESCRIPTION", skill_path, "description must contain 1..60 characters")
        if not body.strip():
            _error(errors, "V_BODY_EMPTY", skill_path, "skill body is empty")
        if trigger_policy == "all":
            trigger = re.compile(rf"(?<![A-Za-z0-9._-]){re.escape(name)}(?![A-Za-z0-9._-])")
            if not trigger.search(overlay):
                _error(errors, "V_TRIGGER_MISSING", "overlay.md", f"missing trigger for {name}")


def _validate_references(harness: Harness, errors: list[ValidationError]) -> None:
    for _ref in _extract_references(harness.overlay):
        _error(errors, "V_REF_IN_OVERLAY", "overlay.md", "support-file references belong in a skill")
    for path, text in harness.files.items():
        skill_name, _ = _valid_path(path) if isinstance(path, str) else (None, None)
        if skill_name is None:
            continue
        for ref in _extract_references(text):
            code = _classify_reference(ref, skill_name, harness.files)
            if code:
                _error(errors, code, path, "invalid support-file reference")


def _validate_content_safety(harness: Harness, errors: list[ValidationError]) -> None:
    for path, text in [("overlay.md", harness.overlay), *harness.files.items()]:
        encoded_size = len(text.encode("utf-8"))
        # Oversize inputs are already rejected above; run the fast strict scanner on
        # them without invoking the repository redactor's substantially slower pass.
        secret = bool(store.scan_secrets(text)) if encoded_size > MAX_SKILL_BYTES else store.contains_secret(text)
        if secret:
            _error(errors, "V_SECRET", path, "secret-like content rejected")
        if store.contains_invisible(text) or store.contains_prompt_injection(text):
            _error(errors, "V_INJECTION", path, "unsafe instruction content")


def validate_harness(harness: Harness, *, trigger_policy: str = "all") -> ValidationResult:
    errors: list[ValidationError] = []
    skill_names = _validate_bundle_limits(harness, errors)
    _validate_skill_metadata(harness.overlay, harness.files, skill_names, trigger_policy, errors)
    _validate_references(harness, errors)
    _validate_content_safety(harness, errors)
    if trigger_policy not in {"all", "none"}:
        _error(errors, "V_PATH", "", "unknown trigger policy")
    return ValidationResult(tuple(errors))


def verify_evidence(
    ev: Evidence,
    *,
    base_probe: ProbeResult,
    cand_probe: ProbeResult,
    diff_text: str,
    edited_paths: Iterable[str],
    op: Op | str,
) -> None:
    if not ev.base_excerpt.strip() or not ev.candidate_excerpt.strip():
        raise EvidenceError()
    if ev.base_excerpt not in base_probe.text or ev.candidate_excerpt not in cand_probe.text:
        raise EvidenceError()
    diff_lines = diff_text.splitlines()
    if not ev.diff_lines or any(line not in diff_lines for line in ev.diff_lines):
        raise EvidenceError()
    content_lines = [
        line for line in ev.diff_lines
        if not line.startswith(("--- ", "+++ ", "@@ ", "diff --git ", "index "))
        and line != r"\ No newline at end of file"
    ]
    try:
        operation = Op(op)
    except (TypeError, ValueError):
        raise EvidenceError() from None
    additions = [line for line in content_lines if line.startswith("+")]
    removals = [line for line in content_lines if line.startswith("-")]
    if operation is Op.add and not additions:
        raise EvidenceError()
    if operation is Op.remove and not removals:
        raise EvidenceError()
    if operation is Op.replace and (not additions or not removals):
        raise EvidenceError()
    if ev.diff_lines and any(line not in diff_lines for line in ev.diff_lines):
        raise EvidenceError()
    paths = tuple(edited_paths)
    if len(paths) != 1:
        raise EvidenceError()
    component = paths[0]
    headers = [line for line in diff_lines if line.startswith(("--- ", "+++ ", "diff --git "))]
    if not any(f"a/{component}" in line or f"b/{component}" in line for line in headers):
        raise EvidenceError()
