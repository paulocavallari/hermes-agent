"""Pure Harness² contracts, canonicalization, and strict wire parsers."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence, get_args, get_origin, get_type_hints


class Mode(str, Enum):
    parallel = "parallel"
    sequential = "sequential"


class Dimension(str, Enum):
    INV = "INV"
    DOM = "DOM"


class Op(str, Enum):
    add = "add"
    replace = "replace"
    remove = "remove"


class Status(str, Enum):
    ok = "ok"
    error = "error"
    skipped = "skipped"


class Decision(str, Enum):
    CANDIDATE = "CANDIDATE"
    KEEP_BASE = "KEEP_BASE"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Verdict(str, Enum):
    helped = "helped"
    hurt = "hurt"
    no_observed_effect = "no_observed_effect"
    confounded = "confounded"


class InvalidPayload(ValueError):
    """A bounded, non-echoing wire-payload validation failure."""

    def __init__(self, detail: str = "invalid payload"):
        super().__init__(detail[:200])
        self.detail = detail[:200]


_ID_PATTERNS = {
    "run": re.compile(r"^run-[0-9a-f]{12}$"),
    "harness": re.compile(r"^h-[0-9a-f]{12}$"),
    "domain": re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$"),
    "task": re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$"),
    "skill": re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$"),
    "proposal": re.compile(r"^p-[0-9]{2}$"),
    "candidate": re.compile(r"^cand-[0-9]{2}$"),
    "label": re.compile(r"^opt-[0-9]{1,2}$"),
    "lesson": re.compile(r"^l-[0-9a-f]{8}$"),
}


def valid_id(kind: str, value: str) -> bool:
    pattern = _ID_PATTERNS.get(kind)
    return bool(pattern and isinstance(value, str) and pattern.fullmatch(value))


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        return ""
    return text.rstrip("\n") + "\n"


def normalize_for_signature(text: str) -> str:
    normalized = normalize_text(text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines())


def behavior_signature(status: Status | str, text: str, harness_id: str | None = None) -> str:
    del harness_id
    status_value = status.value if isinstance(status, Status) else str(status)
    body = f"{status_value}\n{normalize_for_signature(text)}"
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _valid_harness_path(path: str) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or path.startswith("/"):
        return False
    parts = path.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        return False
    if path == "overlay.md" or len(parts) < 3 or parts[0] != "skills":
        return False
    if not valid_id("skill", parts[1]):
        return False
    if len(parts) == 3:
        return parts[2] == "SKILL.md"
    return parts[2] in {"references", "templates", "assets", "scripts"} and all(parts[3:])


def canonical_bytes(harness: Harness) -> bytes:
    payload = {"overlay": harness.overlay, "files": {key: harness.files[key] for key in sorted(harness.files)}}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


@dataclass(frozen=True, slots=True)
class Harness:
    overlay: str
    files: dict[str, str]
    harness_id: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        overlay = normalize_text(self.overlay)
        if not isinstance(self.files, Mapping):
            raise TypeError("files must be a mapping")
        normalized: dict[str, str] = {}
        for path, text in self.files.items():
            if not _valid_harness_path(path):
                raise ValueError("invalid harness path")
            if not isinstance(text, str):
                raise TypeError("harness file contents must be str")
            normalized[path] = normalize_text(text)
        object.__setattr__(self, "overlay", overlay)
        object.__setattr__(self, "files", normalized)
        blob = canonical_bytes(self)
        digest = hashlib.sha256(blob).hexdigest()
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(self, "harness_id", "h-" + digest[:12])

    @classmethod
    def make(cls, overlay: str, files: Mapping[str, str] | None = None) -> Harness:
        return cls(overlay, dict(files or {}))

    @classmethod
    def empty(cls) -> Harness:
        return cls.make("")

    def to_dict(self) -> dict[str, Any]:
        return {"overlay": self.overlay, "files": dict(self.files)}


class _StrictModel:
    """Dataclass serializer whose persisted parser rejects key drift."""

    SCHEMA_KEY: str

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _encode(getattr(self, item.name)) for item in fields(self) if item.init}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]):
        if not isinstance(obj, dict):
            raise TypeError("persisted value must be an object")
        required = SCHEMAS[cls.SCHEMA_KEY]
        if set(obj) != set(required):
            missing = set(required) - set(obj)
            extra = set(obj) - set(required)
            raise ValueError(f"persisted schema mismatch (missing={len(missing)}, extra={len(extra)})")
        hints = get_type_hints(cls)
        values = {key: _decode(hints[key], obj[key]) for key in required}
        return cls(**values)


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _encode(getattr(value, item.name)) for item in fields(value) if item.init}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def _decode(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is None and isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise TypeError("expected array")
        if origin is tuple:
            subtype = args[0] if args else Any
            return tuple(_decode(subtype, item) for item in value)
        subtype = args[0] if args else Any
        return [_decode(subtype, item) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("expected object")
        key_type, val_type = args or (Any, Any)
        return {_decode(key_type, key): _decode(val_type, item) for key, item in value.items()}
    if origin is not None and args:
        # Optional / PEP 604 union.
        if type(None) in args and value is None:
            return None
        candidates = [arg for arg in args if arg is not type(None)]
        last_error = None
        for candidate in candidates:
            try:
                return _decode(candidate, value)
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error:
            raise last_error
    if isinstance(tp, type) and is_dataclass(tp):
        return tp.from_dict(value)
    if tp is Any:
        return value
    if tp is bool and type(value) is not bool:
        raise TypeError("expected bool")
    if tp is int and type(value) is not int:
        raise TypeError("expected int")
    if tp is float and type(value) not in (int, float):
        raise TypeError("expected number")
    if tp is str and not isinstance(value, str):
        raise TypeError("expected string")
    if tp in (str, int, float, bool) and not isinstance(value, tp):
        raise TypeError(f"expected {tp.__name__}")
    return value


@dataclass(frozen=True, slots=True)
class Edit(_StrictModel):
    op: Op
    path: str
    new_text: str | None
    SCHEMA_KEY = "edit"


@dataclass(frozen=True, slots=True)
class Proposal(_StrictModel):
    proposal_id: str
    round: int
    dimension: Dimension
    edits: tuple[Edit, ...]
    rationale: str
    keep_base: bool
    fell_back: bool
    error: str | None
    SCHEMA_KEY = "proposal"


@dataclass(frozen=True, slots=True)
class Usage(_StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    SCHEMA_KEY = "usage"


@dataclass(frozen=True, slots=True)
class ProbeResult(_StrictModel):
    task_id: str
    label: str
    status: Status
    text: str
    error: str | None
    usage: Usage
    model: str
    provider: str
    behavior_signature: str
    probe_mode: str
    harness_id: str
    wall_ms: int
    SCHEMA_KEY = "probe_result"


@dataclass(frozen=True, slots=True)
class Vote(_StrictModel):
    voter: int
    decision: Decision
    chosen: str | None
    rationale: str
    error: str | None
    SCHEMA_KEY = "vote"


@dataclass(frozen=True, slots=True)
class Selection(_StrictModel):
    round: int
    decision: Decision
    chosen_label: str | None
    rationale: str
    reason_code: str
    votes: tuple[Vote, ...]
    mapping: dict[str, str]
    SCHEMA_KEY = "selection"


@dataclass(frozen=True, slots=True)
class Evidence(_StrictModel):
    diff_path: str
    diff_lines: tuple[str, ...]
    base_probe_path: str
    base_excerpt: str
    candidate_probe_path: str
    candidate_excerpt: str
    SCHEMA_KEY = "evidence"


@dataclass(frozen=True, slots=True)
class Lesson(_StrictModel):
    lesson_id: str
    domain: str
    run_id: str
    round: int
    candidate_id: str
    component: str
    cue: str
    lesson: str
    evidence: Evidence
    confidence: Confidence
    created_at: str
    SCHEMA_KEY = "lesson"


@dataclass(frozen=True, slots=True)
class Classification(_StrictModel):
    candidate_id: str
    path: str
    classification: Verdict
    confidence: Confidence
    reason: str
    SCHEMA_KEY = "classification"


@dataclass(frozen=True, slots=True)
class CandidateReflection(_StrictModel):
    candidate_id: str
    harness_id: str
    classifications: tuple[Classification, ...]
    reason: str
    SCHEMA_KEY = "candidate_reflection"


@dataclass(frozen=True, slots=True)
class Reflection(_StrictModel):
    round: int
    base_harness_id: str
    candidates: tuple[CandidateReflection, ...]
    lesson: Lesson | None
    no_entry: bool
    reason: str
    SCHEMA_KEY = "reflection"


@dataclass(frozen=True, slots=True)
class RoundSummary(_StrictModel):
    round: int
    base_harness_id: str
    candidate_ids: tuple[str, ...]
    decision: Decision
    chosen_label: str | None
    classifications: tuple[Classification, ...]
    lesson_id: str | None
    reason_code: str
    SCHEMA_KEY = "round_summary"


@dataclass(frozen=True, slots=True)
class HarnessInfo(_StrictModel):
    harness_id: str
    created_at: str
    skills: tuple[str, ...]
    overlay_chars: int
    summary: str
    SCHEMA_KEY = "harness_info"


@dataclass(frozen=True, slots=True)
class LibraryContext(_StrictModel):
    domain: str
    lessons: tuple[Lesson, ...]
    inventory: tuple[HarnessInfo, ...]
    omitted: int
    SCHEMA_KEY = "library_context"

    def render(self, *, cap_chars: int = 4000) -> str:
        lines = [f"LESSONS (same domain, max 5): {self.domain}"]
        for lesson in self.lessons[:5]:
            lines.append(f"- {lesson.cue}: {lesson.lesson} [{lesson.confidence.value}]")
        lines.append("HARNESS INVENTORY (metadata only; full files live under library/<domain>/harnesses/)")
        for item in self.inventory[:20]:
            lines.append(f"- {item.harness_id} | {item.created_at} | skills={','.join(item.skills)} | overlay={item.overlay_chars} | {item.summary[:120]}")
        lines.append("Full harness files are on disk under library/<domain>/harnesses/; this prompt contains lessons and inventory metadata only.")
        rendered = "\n".join(lines)
        if len(rendered) <= cap_chars:
            return rendered
        inventory_start = next((i for i, line in enumerate(lines) if line.startswith("HARNESS INVENTORY")), len(lines))
        while len("\n".join(lines)) > cap_chars and len(lines) > inventory_start + 2:
            lines.pop(inventory_start + 1)
        while len("\n".join(lines)) > cap_chars and self.lessons:
            lesson_line = next((i for i, line in enumerate(lines) if line.startswith("- ")), None)
            if lesson_line is None:
                break
            lines.pop(lesson_line)
        marker = "\n...[context truncated]"
        while len("\n".join(lines)) + len(marker) > cap_chars and len(lines) > 2:
            lines.pop(-2 if len(lines) > 1 else -1)
        base = "\n".join(lines)
        return (base[:max(0, cap_chars - len(marker))] + marker) if len(base) > cap_chars or len(lines) < len(self.inventory) + len(self.lessons) + 3 else base


@dataclass(frozen=True, slots=True)
class StageUsage(_StrictModel):
    calls: int
    ms: int
    SCHEMA_KEY = "stage_usage"


@dataclass(frozen=True, slots=True)
class Ledger(_StrictModel):
    run_id: str
    mode: str
    runs: int
    stages: dict[str, StageUsage]
    probes: int
    llm_calls: int
    tokens: Usage
    cost_usd: float | None
    wall_ms: int
    SCHEMA_KEY = "ledger"


@dataclass(frozen=True, slots=True)
class RunResult(_StrictModel):
    run_id: str
    decision: Decision
    run_dir: str
    exit_code: int
    message: str
    final_harness_id: str
    SCHEMA_KEY = "run_result"


@dataclass(frozen=True, slots=True)
class WireEdit:
    op: Op
    path: str
    new_text: str | None


@dataclass(frozen=True, slots=True)
class ProposalPayload:
    dimension: Dimension
    edits: tuple[WireEdit, ...]
    rationale: str
    keep_base: bool


@dataclass(frozen=True, slots=True)
class SelectionPayload:
    decision: Decision
    chosen: str | None
    rationale: str


@dataclass(frozen=True, slots=True)
class LlmRequest:
    purpose: str
    system: str
    user: str
    max_tokens: int
    temperature: float
    json_mode: bool = True


@dataclass(frozen=True, slots=True)
class LlmResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LlmClient(Protocol):
    def complete(self, req: LlmRequest) -> LlmResult: ...


class Runner(Protocol):
    probe_mode: str

    def run(self, *, task_id: str, system: str, task: str, label: str, harness_id: str) -> ProbeResult: ...


def _wire_object(obj: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != keys:
        raise InvalidPayload(f"invalid {label} keys")
    return obj


def _wire_path(path: Any) -> str:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or path.startswith("/"):
        raise InvalidPayload("invalid edit path")
    parts = path.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        raise InvalidPayload("invalid edit path")
    if path != "overlay.md" and not _valid_harness_path(path):
        raise InvalidPayload("invalid edit path")
    return path


def parse_proposal_payload(obj: dict[str, Any], expected_dimension: Dimension | str | None = None) -> ProposalPayload:
    obj = _wire_object(obj, {"dimension", "edits", "rationale", "keep_base"}, "proposal")
    try:
        dimension = Dimension(obj["dimension"])
    except (TypeError, ValueError):
        raise InvalidPayload("invalid proposal dimension") from None
    if expected_dimension is not None and dimension != Dimension(expected_dimension):
        raise InvalidPayload("proposal dimension mismatch")
    rationale = obj["rationale"]
    if not isinstance(rationale, str) or len(rationale) > 300:
        raise InvalidPayload("invalid proposal rationale")
    if type(obj["keep_base"]) is not bool or not isinstance(obj["edits"], list):
        raise InvalidPayload("invalid proposal fields")
    edits: list[WireEdit] = []
    for raw in obj["edits"]:
        if not isinstance(raw, dict) or "op" not in raw or "path" not in raw:
            raise InvalidPayload("invalid edit fields")
        try:
            op = Op(raw["op"])
        except (TypeError, ValueError):
            raise InvalidPayload("invalid edit operation") from None
        expected = {"op", "path"} if op is Op.remove else {"op", "path", "new_text"}
        if set(raw) != expected:
            raise InvalidPayload("invalid edit keys")
        path = _wire_path(raw["path"])
        new_text = raw.get("new_text")
        if op is Op.remove:
            new_text = None
        elif not isinstance(new_text, str) or (op is Op.add and not new_text):
            raise InvalidPayload("invalid edit text")
        edits.append(WireEdit(op, path, new_text))
    if obj["keep_base"] and edits:
        raise InvalidPayload("keep_base requires no edits")
    return ProposalPayload(dimension, tuple(edits), rationale, obj["keep_base"])


def parse_selection_payload(obj: dict[str, Any]) -> SelectionPayload:
    obj = _wire_object(obj, {"decision", "chosen", "rationale"}, "selection")
    try:
        decision = Decision(obj["decision"])
    except (TypeError, ValueError):
        raise InvalidPayload("invalid selection decision") from None
    chosen = obj["chosen"]
    rationale = obj["rationale"]
    if not isinstance(rationale, str) or len(rationale) > 300:
        raise InvalidPayload("invalid selection rationale")
    if decision is Decision.CANDIDATE:
        if not valid_id("label", chosen):
            raise InvalidPayload("invalid selected label")
    elif chosen is not None:
        raise InvalidPayload("KEEP_BASE requires null choice")
    return SelectionPayload(decision, chosen, rationale)


def render_history(summaries: Sequence[RoundSummary], *, cap_chars: int = 1200) -> str:
    if not summaries:
        return ""
    lines = ["PRIOR ROUNDS"]
    for summary in summaries[-9:]:
        candidates = ",".join(summary.candidate_ids)
        classifications = ", ".join(
            f"{item.candidate_id}: {item.path}={item.classification.value}"
            for item in summary.classifications
        )
        lesson = summary.lesson_id or "none"
        line = f"- round {summary.round:02d}: {summary.decision.value} {candidates}"
        if classifications:
            line += f" | {classifications}"
        line += f" | lesson={lesson} | reason={summary.reason_code}"
        lines.append(line)
    if len(lines) > 10:
        lines = [lines[0], *lines[-9:]]
    while len("\n".join(lines)) > cap_chars and len(lines) > 2:
        lines.pop(1)
    result = "\n".join(lines)
    if len(result) > cap_chars:
        return result[:max(0, cap_chars - 3)] + "..."
    return result


def _schema(cls: type) -> tuple[str, ...]:
    return tuple(item.name for item in fields(cls) if item.init)


PERSISTED_TYPES = {
    "edit": Edit,
    "proposal": Proposal,
    "usage": Usage,
    "probe_result": ProbeResult,
    "vote": Vote,
    "selection": Selection,
    "evidence": Evidence,
    "lesson": Lesson,
    "classification": Classification,
    "candidate_reflection": CandidateReflection,
    "reflection": Reflection,
    "round_summary": RoundSummary,
    "harness_info": HarnessInfo,
    "library_context": LibraryContext,
    "stage_usage": StageUsage,
    "ledger": Ledger,
    "run_result": RunResult,
}
SCHEMAS = {key: _schema(value) for key, value in PERSISTED_TYPES.items()}
SCHEMAS["proposal_wire"] = ("dimension", "edits", "rationale", "keep_base")
SCHEMAS["selection_wire"] = ("decision", "chosen", "rationale")
