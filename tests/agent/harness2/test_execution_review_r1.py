"""Regressions for the independent Harness² execution review r1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, UTC
from itertools import count
from pathlib import Path

import pytest

from agent.harness2 import judge, library, orchestrator, store
from agent.harness2.llm import FakeLlm
from agent.harness2.models import Decision
from agent.harness2.probe import FakeRunner
from hermes_cli import harness2_cmd

REPO = Path(__file__).resolve().parents[3]
_SYNTHETIC_SECRET = "SERVICE_API_KEY=unit-test-secret-value"


def _task(tmp_path: Path) -> Path:
    path = tmp_path / "task.md"
    path.write_text("Write a concise summary.", encoding="utf-8")
    return path


def _base_dir(tmp_path: Path, overlay: str, files: dict[str, str] | None = None) -> Path:
    root = tmp_path / "base"
    root.mkdir()
    (root / "overlay.md").write_text(overlay, encoding="utf-8")
    for relative, text in (files or {}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def _proposer_and_composer(*, dom_edit: tuple[str, str] | None = None, rationale: str = "preferred output"):
    def respond(request):
        if request.purpose == "propose":
            if "DIMENSION / INV" in request.user:
                payload = {
                    "dimension": "INV",
                    "edits": [{"op": "replace", "path": "overlay.md",
                               "new_text": "Improved operating protocol. Use example when needed."}],
                    "rationale": "single INV edit",
                    "keep_base": False,
                }
            elif dom_edit is None:
                payload = {"dimension": "DOM", "edits": [], "rationale": "keep DOM base", "keep_base": True}
            else:
                path, text = dom_edit
                payload = {
                    "dimension": "DOM",
                    "edits": [{"op": "replace", "path": path, "new_text": text}],
                    "rationale": "single DOM edit",
                    "keep_base": False,
                }
            return json.dumps(payload)
        options = re.findall(r"(opt-\d+) \[status=ok\]\n([^\n]+)", request.user)
        chosen = next(label for label, text in options if "PREFERRED" in text)
        return json.dumps({"decision": "CANDIDATE", "chosen": chosen, "rationale": rationale})

    return respond


def _deps(*, dom_edit: tuple[str, str] | None = None, rationale: str = "preferred output",
          runner: FakeRunner | None = None, token=None, now=None) -> orchestrator.Deps:
    ticks = iter(range(10000))
    return orchestrator.Deps(
        llm=FakeLlm(_proposer_and_composer(dom_edit=dom_edit, rationale=rationale)),
        runner=runner or FakeRunner({"base": "baseline answer", "cand-01": "PREFERRED candidate answer"}),
        now=now or (lambda: datetime(2025, 1, 1, tzinfo=UTC)),
        monotonic=ticks.__next__,
        token=token or (lambda size: "a" * size),
    )


def _parallel_options(task: Path, base_dir: Path, **overrides) -> orchestrator.EvolveOptions:
    values = {
        "task_path": task,
        "domain": "demo",
        "mode": "parallel",
        "k": 1,
        "proposer": "llm",
        "probe": "live",
        "composer": "llm",
        "base_dir": base_dir,
        "dry_run": True,
    }
    values.update(overrides)
    return orchestrator.EvolveOptions(**values)


def test_parallel_reflects_inv_only_edit_and_records_verified_lesson(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    base_dir = _base_dir(tmp_path, "Original protocol.")
    result = orchestrator.run_evolve(
        _parallel_options(_task(tmp_path), base_dir), home=home, deps=_deps()
    )

    writer = store.Store(home)
    root = f"runs/{result.run_id}/rounds/round-01"
    reflection = writer.read_json(f"{root}/reflection.json")
    candidate = writer.read_bundle(f"{root}/candidates/cand-01")
    classified = reflection["candidates"][0]["classifications"]

    assert result.exit_code == 0 and result.decision is Decision.CANDIDATE
    assert candidate.overlay == "Improved operating protocol. Use example when needed.\n"
    assert [(item["path"], item["classification"], item["reason"]) for item in classified] == [
        ("overlay.md", "helped", "isolated_single_edit")
    ]
    lesson = reflection["lesson"]
    assert lesson is not None and lesson["component"] == "overlay.md"
    assert "-Original protocol." in lesson["evidence"]["diff_lines"]
    assert "+Improved operating protocol. Use example when needed." in lesson["evidence"]["diff_lines"]
    assert lesson["evidence"]["base_excerpt"] == "baseline answer"
    assert lesson["evidence"]["candidate_excerpt"] == "PREFERRED candidate answer"


def test_parallel_inv_and_dom_paths_are_both_confounded_without_lesson(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    base_skill = "---\nname: example\ndescription: Use example skill\n---\nOriginal skill body.\n"
    new_skill = "---\nname: example\ndescription: Use example skill\n---\nImproved skill body.\n"
    base_dir = _base_dir(tmp_path, "Original protocol. Use example when needed.",
                         {"skills/example/SKILL.md": base_skill})
    result = orchestrator.run_evolve(
        _parallel_options(_task(tmp_path), base_dir),
        home=home,
        deps=_deps(dom_edit=("skills/example/SKILL.md", new_skill)),
    )

    writer = store.Store(home)
    root = f"runs/{result.run_id}/rounds/round-01"
    reflection = writer.read_json(f"{root}/reflection.json")
    classified = reflection["candidates"][0]["classifications"]

    assert result.exit_code == 0 and result.decision is Decision.CANDIDATE
    assert [(item["path"], item["classification"], item["reason"]) for item in classified] == [
        ("overlay.md", "confounded", "not_isolated"),
        ("skills/example/SKILL.md", "confounded", "not_isolated"),
    ]
    assert reflection["lesson"] is None and reflection["no_entry"] is True


def _invoke_cli_evolve(home: Path, task: Path, deps: orchestrator.Deps) -> int:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    harness2_cmd.build_harness2_parser(subparsers, cmd_harness2=harness2_cmd.cmd_harness2)
    args = parser.parse_args([
        "harness2", "evolve", "--task", str(task), "--domain", "repeat-demo",
        "--mode", "sequential", "--rounds", "1", "--proposer", "llm",
        "--probe", "live", "--composer", "llm", "--json",
    ])
    token = set_hermes_home_override(home)
    try:
        return harness2_cmd.cmd_harness2(args, deps=deps)
    finally:
        reset_hermes_home_override(token)


def test_repeated_sequential_cli_runs_dedupe_semantic_lesson(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    active = home / "harness2" / "active" / "overlay.md"
    active.parent.mkdir(parents=True)
    active.write_text("Existing baseline instructions.\n", encoding="utf-8")
    task = _task(tmp_path)
    ids = count(1)
    seconds = count()

    def token(size):
        return f"{next(ids):0{size}x}"

    def now():
        return datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=next(seconds))

    deps = _deps(token=token, now=now)
    first_code = _invoke_cli_evolve(home, task, deps)
    first = json.loads(capsys.readouterr().out)
    second_code = _invoke_cli_evolve(home, task, deps)
    second_output = capsys.readouterr().out

    writer = store.Store(home)
    lessons = library.list_lessons(writer, "repeat-demo")
    assert first_code == second_code == 0
    second = json.loads(second_output)
    assert first["exit_code"] == second["exit_code"] == 0
    assert first["run_id"] != second["run_id"]
    assert len(lessons) == 1
    assert lessons[0].run_id == first["run_id"]
    second_lesson = writer.read_json(f"runs/{second['run_id']}/reflection.json")["lessons"][0]
    assert second_lesson["lesson_id"] == lessons[0].lesson_id
    assert second_lesson["evidence"]["diff_lines"] == list(lessons[0].evidence.diff_lines)
    assert second_lesson["evidence"]["diff_path"] != lessons[0].evidence.diff_path
    assert second_lesson["created_at"] != lessons[0].created_at


def test_same_lesson_id_with_incompatible_evidence_still_collides(tmp_path):
    from dataclasses import replace

    lesson = library_test_lesson()
    writer = store.Store(tmp_path)
    library.append_lessons(writer, "demo", (lesson,))
    incompatible = replace(
        lesson,
        run_id="run-000000000099",
        evidence=replace(lesson.evidence, diff_lines=("+different content",)),
    )

    with pytest.raises(library.LibraryError) as error:
        library.append_lessons(writer, "demo", (incompatible,))
    assert error.value.code == "E_LESSON_COLLISION"


def library_test_lesson():
    from agent.harness2.models import Confidence, Evidence, Lesson

    return Lesson(
        "l-00000001", "demo", "run-000000000001", 1, "cand-01", "overlay.md",
        "editing overlay.md for demo tasks", "same learning", Evidence(
            "runs/run-000000000001/diffs/cand-01.diff", ("-old", "+new"),
            "runs/run-000000000001/probes/base.json", "base answer",
            "runs/run-000000000001/probes/cand-01.json", "candidate answer",
        ), Confidence.high, "2025-01-01T00:00:00Z",
    )


def _fifo_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    active = home / "harness2" / "active"
    active.mkdir(parents=True)
    os.mkfifo(active / "overlay.md")
    return home


def _subprocess_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_HOME"] = str(home)
    return env


def test_load_active_fifo_returns_fail_closed_in_timed_subprocess(tmp_path):
    home = _fifo_home(tmp_path)
    script = (
        "import os; from pathlib import Path; from agent.harness2.overlay import load_active; "
        "assert load_active(Path(os.environ['HERMES_HOME'])) is None"
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=_subprocess_env(home),
                                text=True, capture_output=True, timeout=5, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("load_active blocked while opening a FIFO") from exc
    assert result.returncode == 0, result.stderr


def test_status_fifo_returns_invalid_without_blocking_in_timed_subprocess(tmp_path):
    home = _fifo_home(tmp_path)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "harness2", "status", "--json"],
            cwd=REPO,
            env=_subprocess_env(home),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("harness2 status blocked while opening a FIFO") from exc
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["active_overlay"] == {"present": True, "chars": 0, "valid": False}


def test_composer_rationale_is_redacted_and_run_artifacts_complete(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    base_dir = _base_dir(tmp_path, "Original protocol.")
    raw_rationale = f"preferred output; {_SYNTHETIC_SECRET}"
    result = orchestrator.run_evolve(
        _parallel_options(_task(tmp_path), base_dir),
        home=home,
        deps=_deps(rationale=raw_rationale),
    )

    run_dir = store.Store(home).root / "runs" / result.run_id
    writer = store.Store(home)
    round_root = f"runs/{result.run_id}/rounds/round-01"
    selection = writer.read_json(f"{round_root}/selection.json")
    reflection_path = run_dir / "rounds" / "round-01" / "reflection.json"
    ledger_path = run_dir / "ledger.json"

    assert result.exit_code == 0 and result.decision is Decision.CANDIDATE
    assert judge.redact_strict is store.redact_strict
    for rationale in [selection["rationale"], selection["votes"][0]["rationale"]]:
        assert _SYNTHETIC_SECRET not in rationale
        assert "redacted-secret" in rationale
    assert reflection_path.is_file() and ledger_path.is_file()
    assert writer.read_json(f"{round_root}/reflection.json")["candidates"]
    assert writer.read_json(f"runs/{result.run_id}/reflection.json")["rounds"]
    assert writer.read_json(f"runs/{result.run_id}/ledger.json")["stages"]
    assert writer.read_json(f"runs/{result.run_id}/selection.json")["decision"] == "CANDIDATE"
    assert all(
        _SYNTHETIC_SECRET not in path.read_bytes().decode("utf-8")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
