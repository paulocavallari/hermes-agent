"""Confined, atomic, and secret-aware persistence for Harness²."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Literal, Mapping

from agent.redact import redact_sensitive_text
from utils import atomic_write_text

from .models import Harness


class PathError(OSError):
    def __init__(self, code: str = "E_PATH", path: object = ""):
        self.code = code
        self.path = str(path)
        super().__init__(f"{code}: {self.path}")


class SecretError(ValueError):
    def __init__(self, code: str = "E_SECRET", path: object = ""):
        self.code = code
        self.path = str(path)
        super().__init__(f"{code}: {self.path}")


@dataclass(frozen=True, slots=True)
class SecretMatch:
    family: str
    start: int
    end: int


DEFAULT_TREE_CAPS = {"files": 64, "file_bytes": 65536, "total_bytes": 1_048_576}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MASK_PREFIXES = ("«", "***", "<", "$", "${")
_PLACEHOLDERS = {"redacted", "none", "null", "true", "false", "changeme", "example", "placeholder"}

_PREFIX_PATTERNS = (
    ("github", re.compile(r"(?<![A-Za-z0-9_])(?:ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]{12,}")),
    ("gitlab", re.compile(r"(?<![A-Za-z0-9_])glpat-[A-Za-z0-9_-]{12,}")),
    ("slack", re.compile(r"(?<![A-Za-z0-9_])xox[baprs]-[A-Za-z0-9-]{12,}")),
    ("openai", re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9]{20,}")),
    ("aws", re.compile(r"(?<![A-Za-z0-9_])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("google", re.compile(r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])")),
    ("huggingface", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}")),
    ("npm", re.compile(r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{30,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
)
_ASSIGNMENT = re.compile(
    r"(?i)(?P<key>(?:[A-Za-z0-9_.-]*[_-])?(?:api[_-]?key|apikey|secret|token|password|passwd|authorization|bearer|access[_-]?key|private[_-]?key|client[_-]?secret))"
    r"\s*['\"]?\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9_\-./+=]{6,})"
)
_ENV_ASSIGNMENT = re.compile(
    r"(?m)^(?P<key>[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|KEY|PASSWORD|PASSWD|CREDENTIAL|AUTH|PRIVATE)[A-Z0-9_]*)"
    r"\s*=\s*(?P<value>\S{4,})$"
)
_URL_USERINFO = re.compile(r"://(?P<value>[^/\s@]+)@")
_URL_QUERY = re.compile(r"(?i)[?&](?P<key>token|key|api_key|access_token|sig|signature|password|secret)=(?P<value>[^&\s]{3,})")


def _placeholder(value: str) -> bool:
    value = value.strip().strip("'\"")
    return value.startswith(_MASK_PREFIXES) or value.lower() in _PLACEHOLDERS


def scan_secrets(text: str) -> tuple[SecretMatch, ...]:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    found: list[SecretMatch] = []
    for family, pattern in _PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            if not _placeholder(match.group(0)):
                found.append(SecretMatch(family, match.start(), match.end()))
    lowered = text.lower()
    for family, pattern, group in (
        ("assignment", _ASSIGNMENT, "value"),
        ("env", _ENV_ASSIGNMENT, "value"),
        ("url_userinfo", _URL_USERINFO, "value"),
        ("url_query", _URL_QUERY, "value"),
    ):
        if family == "assignment" and not any(token in lowered for token in (
            "api", "key", "secret", "token", "password", "passwd", "authorization", "bearer", "credential", "private"
        )):
            continue
        if family == "env" and "=" not in text:
            continue
        if family == "url_userinfo" and "://" not in text:
            continue
        if family == "url_query" and not any(token in lowered for token in (
            "?token=", "&token=", "?key=", "&key=", "api_key=", "access_token=", "signature=", "?sig=", "&sig=", "password=", "secret="
        )):
            continue
        for match in pattern.finditer(text):
            if not _placeholder(match.group(group)):
                found.append(SecretMatch(family, match.start(), match.end()))
    return tuple(sorted(found, key=lambda item: (item.start, item.end, item.family)))


def _mask_spans(text: str, matches: tuple[SecretMatch, ...]) -> str:
    if not matches:
        return text
    merged: list[tuple[int, int, str]] = []
    for match in matches:
        if merged and match.start <= merged[-1][1]:
            start, end, families = merged[-1]
            merged[-1] = (start, max(end, match.end), families)
        else:
            merged.append((match.start, match.end, match.family))
    result = text
    for start, end, family in reversed(merged):
        result = result[:start] + f"«redacted:{family}»" + result[end:]
    return result


def contains_secret(text: str) -> bool:
    if scan_secrets(text):
        return True
    return redact_sensitive_text(
        text, force=True, file_read=True, secret_file=True, redact_url_credentials=True
    ) != text


def redact_strict(text: str) -> str:
    output = redact_sensitive_text(
        text, force=True, file_read=True, secret_file=True, redact_url_credentials=True
    )
    return _mask_spans(output, scan_secrets(output))


def assert_safe_relpath(path: str | Path) -> str:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw or ":" in raw:
        raise PathError("E_PATH", raw)
    candidate = PurePath(raw)
    if candidate.is_absolute():
        raise PathError("E_PATH", raw)
    parts = raw.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") or not _SAFE_COMPONENT.fullmatch(part) for part in parts):
        raise PathError("E_PATH", raw)
    return "/".join(parts)


def _raw_has_dotdot(path: Path) -> bool:
    return ".." in path.parts


def _lexical_abs(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def confine_ancestors(path: Path, *, anchor: Path) -> None:
    """Read-only lstat walk that rejects traversal and every existing symlink."""
    raw_path, raw_anchor = Path(path), Path(anchor)
    if _raw_has_dotdot(raw_path) or _raw_has_dotdot(raw_anchor):
        raise PathError("E_PATH", path)
    absolute_anchor = _lexical_abs(raw_anchor)
    absolute_path = _lexical_abs(raw_path if raw_path.is_absolute() else Path.cwd() / raw_path)
    try:
        common = Path(os.path.commonpath((absolute_anchor, absolute_path)))
    except (ValueError, OSError):
        raise PathError("E_PATH", path) from None
    if common != absolute_anchor:
        raise PathError("E_PATH", path)
    try:
        relative = absolute_path.relative_to(absolute_anchor)
    except ValueError:
        raise PathError("E_PATH", path) from None
    current = absolute_anchor
    components = [current]
    for part in relative.parts:
        current = current / part
        components.append(current)
    for index, component in enumerate(components):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PathError("E_PATH", component)
        if index < len(components) - 1 and not stat.S_ISDIR(info.st_mode):
            raise PathError("E_PATH", component)


def _read_nofollow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PathError("E_PATH", path)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return stream.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _lstat_optional(path: Path):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value) if item.init}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _walk_tree(root: Path, prefix: str, out: dict[str, str], totals: list[int], caps: Mapping[str, int]) -> None:
    root_info = os.lstat(root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise PathError("E_PATH", root)
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise PathError("E_PATH", entry.path)
            if entry.is_dir(follow_symlinks=False):
                _walk_tree(Path(entry.path), prefix + entry.name + "/", out, totals, caps)
            elif entry.is_file(follow_symlinks=False):
                info = entry.stat(follow_symlinks=False)
                if info.st_size > caps["file_bytes"]:
                    raise PathError("E_IO", entry.path)
                if len(out) + 1 > caps["files"] or totals[0] + info.st_size > caps["total_bytes"]:
                    raise PathError("E_IO", entry.path)
                content = _read_nofollow(Path(entry.path))
                totals[0] += info.st_size
                out[prefix + entry.name] = content
            else:
                raise PathError("E_PATH", entry.path)


def read_plain_tree(root: Path, *, caps: Mapping[str, int], anchor: Path) -> dict[str, str]:
    root = Path(root)
    confine_ancestors(root, anchor=Path(anchor))
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PathError("E_PATH", root)
    out: dict[str, str] = {}
    _walk_tree(root, "", out, [0], caps)
    return out


class Store:
    def __init__(self, home: Path):
        self.home = _lexical_abs(Path(home))
        self.root = self.home / "harness2"
        self._ensure_root()

    def _ensure_root(self) -> None:
        if not self.home.exists():
            raise PathError("E_PATH", self.home)
        confine_ancestors(self.home, anchor=Path(self.home.anchor or "/"))
        home_info = os.lstat(self.home)
        if stat.S_ISLNK(home_info.st_mode) or not stat.S_ISDIR(home_info.st_mode):
            raise PathError("E_PATH", self.home)
        self._confine(self.root, leaf="dir", create=True)

    def _confine(self, path: Path, *, leaf: Literal["file", "dir"], create: bool = False) -> None:
        path = Path(path)
        confine_ancestors(path.parent, anchor=self.home)
        info = _lstat_optional(path)
        if info is not None:
            if stat.S_ISLNK(info.st_mode):
                raise PathError("E_PATH", path)
            if leaf == "file" and not stat.S_ISREG(info.st_mode):
                raise PathError("E_PATH", path)
            if leaf == "dir" and not stat.S_ISDIR(info.st_mode):
                raise PathError("E_PATH", path)
            return
        if not create:
            return
        if leaf == "dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        confine_ancestors(path.parent, anchor=self.home)
        info = _lstat_optional(path)
        if info is not None:
            if stat.S_ISLNK(info.st_mode):
                raise PathError("E_PATH", path)
            if leaf == "file" and not stat.S_ISREG(info.st_mode):
                raise PathError("E_PATH", path)
            if leaf == "dir" and not stat.S_ISDIR(info.st_mode):
                raise PathError("E_PATH", path)

    def _target(self, rel: str | Path) -> Path:
        return self.root / assert_safe_relpath(rel)

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"run-[0-9a-f]{12}", run_id):
            raise PathError("E_PATH", run_id)
        return self.root / "runs" / run_id

    def round_dir(self, run_id: str, n: int) -> Path:
        if not re.fullmatch(r"run-[0-9a-f]{12}", run_id) or type(n) is not int or not 1 <= n <= 5:
            raise PathError("E_PATH", f"{run_id}/{n}")
        return self.run_dir(run_id) / "rounds" / f"round-{n:02d}"

    def write_text(self, rel: str | Path, text: str) -> Path:
        safe = assert_safe_relpath(rel)
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if contains_secret(text):
            raise SecretError("E_SECRET", safe)
        path = self.root / safe
        self._confine(path, leaf="file", create=True)
        atomic_write_text(path, text, tmp_prefix=f".{path.stem}_")
        self._confine(path, leaf="file", create=False)
        return path

    def write_json(self, rel: str | Path, obj: Any) -> Path:
        text = json.dumps(_jsonable(obj), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        return self.write_text(rel, text)

    def read_text(self, rel: str | Path) -> str:
        path = self._target(rel)
        self._confine(path, leaf="file", create=False)
        return _read_nofollow(path)

    def read_json(self, rel: str | Path) -> dict[str, Any]:
        value = json.loads(self.read_text(rel))
        if not isinstance(value, dict):
            raise ValueError("stored JSON must be an object")
        return value

    def write_bundle(self, rel_dir: str | Path, harness: Harness) -> None:
        directory = assert_safe_relpath(rel_dir)
        self.write_text(f"{directory}/overlay.md", harness.overlay)
        for path, content in sorted(harness.files.items()):
            self.write_text(f"{directory}/{path}", content)

    def read_bundle(self, rel: str | Path) -> Harness:
        safe = assert_safe_relpath(rel)
        bundle_root = self.root / safe
        confine_ancestors(bundle_root, anchor=self.home)
        tree = read_plain_tree(bundle_root, caps=DEFAULT_TREE_CAPS, anchor=self.root)
        if "overlay.md" not in tree:
            raise PathError("E_IO", bundle_root / "overlay.md")
        overlay = tree.pop("overlay.md")
        return Harness.make(overlay, tree)

    def list_run_ids(self) -> list[str]:
        return self._list_dirs(self.root / "runs", r"run-[0-9a-f]{12}")

    def list_domains(self) -> list[str]:
        return self._list_dirs(self.root / "library", r"[a-z0-9][a-z0-9._-]{0,63}")

    def _list_dirs(self, path: Path, pattern: str) -> list[str]:
        confine_ancestors(path, anchor=self.home)
        info = _lstat_optional(path)
        if info is None:
            return []
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PathError("E_PATH", path)
        result = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise PathError("E_PATH", entry.path)
                if entry.is_dir(follow_symlinks=False) and re.fullmatch(pattern, entry.name):
                    result.append(entry.name)
        return sorted(result)

    def unlink(self, rel: str | Path) -> None:
        path = self._target(rel)
        self._confine(path, leaf="file", create=False)
        try:
            os.unlink(path)
        except FileNotFoundError:
            return
