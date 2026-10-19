"""Harness² CLI contracts, offline behavior, and integration checks."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.harness2 import library, orchestrator, store
from agent.harness2.llm import FakeLlm
from agent.harness2.probe import FakeRunner
from hermes_cli import harness2_cmd
from tests._fixtures.harness2_guard import block_llm, block_network

REPO = Path(__file__).resolve().parents[2]
_BOOTSTRAP_RUNNER = "import runpy; runpy.run_path('tests/_fixtures/harness2_bootstrap.py')"


@pytest.fixture(autouse=True)
def _block_network_and_real_llm(monkeypatch):
    block_network(monkeypatch)
    block_llm(monkeypatch)


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    harness2_cmd.build_harness2_parser(subparsers, cmd_harness2=harness2_cmd.cmd_harness2)
    return parser


def _invoke(argv, home, *, deps=None, capsys=None):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        args = _parser().parse_args(["harness2", *argv])
        code = harness2_cmd.cmd_harness2(args, deps=deps)
        output = capsys.readouterr().out if capsys is not None else ""
        return code, output
    finally:
        reset_hermes_home_override(token)


def _env(home, *, guarded=False, selftest=None):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HERMES_HOME"] = str(home)
    if guarded:
        env["HARNESS2_FORBID_AUX"] = "1"
        env["HARNESS2_ARGV"] = json.dumps(["harness2", "status", "--json"])
    if selftest:
        env["HARNESS2_GUARD_SELFTEST"] = selftest
    return env


def _run_guarded(args, home, *, selftest=None):
    env = _env(home, guarded=True, selftest=selftest)
    env["HARNESS2_ARGV"] = json.dumps(args)
    return subprocess.run([sys.executable, "-c", _BOOTSTRAP_RUNNER], cwd=REPO, env=env,
                          text=True, capture_output=True, timeout=90, check=False)


def _run_literal(args, home):
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], cwd=REPO,
                          env=_env(home), text=True, capture_output=True, timeout=90, check=False)


def _task(tmp_path, text="Summarize the subject clearly."):
    task = tmp_path / "task.md"
    task.write_text(text, encoding="utf-8")
    return task


def _proposal_and_selection(request):
    if request.purpose == "propose":
        if "DIMENSION / INV" in request.user:
            payload = {"dimension": "INV", "edits": [{"op": "replace", "path": "overlay.md",
                       "new_text": "Use the improved operating protocol."}], "rationale": "isolated edit",
                       "keep_base": False}
        else:
            payload = {"dimension": "DOM", "edits": [], "rationale": "no additional domain edit",
                       "keep_base": True}
        return json.dumps(payload)
    options = re.findall(r"(opt-\d+) \[status=ok\]\n([^\n]+)", request.user)
    chosen = next(label for label, text in options if "PREFERRED" in text)
    return json.dumps({"decision": "CANDIDATE", "chosen": chosen, "rationale": "preferred output"})


def _candidate_deps():
    client = FakeLlm(_proposal_and_selection)
    runner = FakeRunner({"base": "ordinary baseline answer", "cand-01": "PREFERRED improved answer"})
    return orchestrator.Deps(llm=client, runner=runner), client, runner


def _json_from(output):
    value = json.loads(output)
    assert isinstance(value, dict)
    return value


def test_help_lists_all_commands(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run_literal(["harness2", "--help"], home)
    assert result.returncode == 0, result.stderr
    for name in ("status", "evolve", "library", "lessons", "rollback"):
        assert name in result.stdout


def test_status_json(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run_literal(["harness2", "status", "--json"], home)
    assert result.returncode == 0, result.stderr
    value = _json_from(result.stdout)
    assert value["home"] == str(home)
    assert value["inject_overlay"] is False
    assert value["active_overlay"] == {"present": False, "chars": 0, "valid": False}
    assert value["runs"] == 0 and value["domains"] == 0 and value["last_run"] is None


def test_child_guard_blocks_socket_and_aux_import(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    for selftest in ("socket", "aux"):
        result = _run_guarded([], home, selftest=selftest)
        assert result.returncode == 0, result.stderr


def test_cli_status_under_child_guard(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run_guarded(["harness2", "status", "--json"], home)
    assert result.returncode == 0, result.stderr
    assert _json_from(result.stdout)["runs"] == 0


def test_evolve_dry_run_offline_json(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    task = _task(tmp_path)
    result = _run_guarded(["harness2", "evolve", "--task", str(task), "--dry-run", "--json"], home)
    assert result.returncode == 0, result.stderr
    value = _json_from(result.stdout)
    assert value["decision"] == "KEEP_BASE" and value["exit_code"] == 0
    writer = store.Store(home)
    root = value["run_dir"]
    assert len(list((writer.root / root / "rounds/round-01/proposals").glob("*.json"))) == 6
    assert writer.read_json(f"{root}/ledger.json")["llm_calls"] == 0
    assert "composer_disabled" in writer.read_text(f"{root}/selection.json")


def test_evolve_composer_llm_with_fake_deps_selects_candidate(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    task = _task(tmp_path)
    deps, client, runner = _candidate_deps()
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "cli-test", "--proposer", "llm",
                            "--probe", "live", "--composer", "llm", "--k", "1", "--dry-run", "--json"],
                           home, deps=deps, capsys=capsys)
    value = _json_from(output)
    assert code == 0 and value["decision"] == "CANDIDATE"
    assert len(client.requests) == 3 and len(runner.calls) == 2


def test_cli_model_reaches_client_factory(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    task = _task(tmp_path)
    received = []
    monkeypatch.setattr(harness2_cmd, "_make_llm_client", lambda model: received.append(model) or FakeLlm(_proposal_and_selection))
    code, output = _invoke(["evolve", "--task", str(task), "--proposer", "llm", "--model", "chosen-model",
                            "--dry-run", "--json"], home, capsys=capsys)
    assert code == 0 and _json_from(output)["exit_code"] == 0
    assert received == ["chosen-model"]


def test_cli_apply_dry_run_preserves_active_bytes(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    active = home / "harness2" / "active" / "overlay.md"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"old active overlay\n")
    before = active.read_bytes()
    deps, _, _ = _candidate_deps()
    task = _task(tmp_path)
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "dry-apply", "--proposer", "llm",
                            "--probe", "live", "--composer", "llm", "--k", "1", "--apply", "--dry-run", "--json"],
                           home, deps=deps, capsys=capsys)
    value = _json_from(output)
    assert code == 0 and value["decision"] == "CANDIDATE"
    assert active.read_bytes() == before
    assert not (active.parent / "overlay.prev").exists()
    assert not (home / "harness2" / "library").exists()


def test_unknown_run_exit_9(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    code, output = _invoke(["lessons", "--domain", "not-here", "--json"], home, capsys=capsys)
    assert code == 9
    assert _json_from(output)["error"] == "E_NOT_FOUND"


def test_lessons_requires_known_domain(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    code, output = _invoke(["lessons", "--domain", "missing-domain", "--json"], home, capsys=capsys)
    assert code == 9
    assert _json_from(output)["error"] == "E_NOT_FOUND"


def test_rollback_without_prev_exit_9(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    code, output = _invoke(["rollback", "--json"], home, capsys=capsys)
    assert code == 9 and _json_from(output) == {"restored": False}


def test_no_print_in_package():
    package = REPO / "agent" / "harness2"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
                       for node in ast.walk(tree)), path.name


def test_module_import_graph_matches_dag():
    package = REPO / "agent" / "harness2"
    modules = {path.stem for path in package.glob("*.py") if path.name != "__init__.py"}
    allowed = {
        "models": set(), "store": {"models"}, "harness": {"models", "store"},
        "validate": {"models", "harness", "store"}, "llm": {"models", "store"},
        "probe": {"models", "harness", "store"}, "judge": {"models", "harness", "llm", "validate"},
        "library": {"models", "harness", "store"},
        "orchestrator": {"models", "store", "harness", "validate", "llm", "probe", "judge", "library", "overlay"},
        "overlay": {"models", "store"},
    }
    graph = {name: set() for name in modules}
    for name in modules:
        tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                if node.module:
                    imported = node.module.split(".")[0]
                    if imported in modules:
                        graph[name].add(imported)
                else:
                    graph[name].update(alias.name for alias in node.names if alias.name in modules)
            elif node.module and node.module.startswith("agent.harness2."):
                imported = node.module.rsplit(".", 1)[-1]
                if imported in modules:
                    graph[name].add(imported)
    assert set(graph) == set(allowed)
    assert all(edges <= allowed[name] for name, edges in graph.items())
    visiting, visited = set(), set()

    def visit(name):
        if name in visiting:
            raise AssertionError(f"import cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for child in graph[name]:
            visit(child)
        visiting.remove(name)
        visited.add(name)

    for module in graph:
        visit(module)


def test_end_to_end_offline_parallel_layout_and_ledger(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    task = _task(tmp_path)
    code, output = _invoke(["evolve", "--task", str(task), "--dry-run", "--json"], home, capsys=capsys)
    value = _json_from(output)
    writer = store.Store(home)
    assert code == 0 and value["decision"] == "KEEP_BASE"
    assert writer.read_text(f"{value['run_dir']}/TASK.md")
    assert writer.read_json(f"{value['run_dir']}/ledger.json")["runs"] == 1
    assert len(list((writer.root / value["run_dir"] / "rounds/round-01/candidates").glob("cand-*"))) == 3


def test_end_to_end_sequential_records_lesson_in_library(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    active = home / "harness2" / "active" / "overlay.md"
    active.parent.mkdir(parents=True)
    active.write_text("Existing baseline instructions.\n", encoding="utf-8")
    task = _task(tmp_path)
    deps, _, _ = _candidate_deps()
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "same-domain", "--mode", "sequential",
                            "--rounds", "1", "--proposer", "llm", "--probe", "live", "--composer", "llm",
                            "--dry-run", "--json"], home, deps=deps, capsys=capsys)
    value = _json_from(output)
    assert code == 0 and value["decision"] == "CANDIDATE"
    deps, _, _ = _candidate_deps()
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "same-domain", "--mode", "sequential",
                            "--rounds", "1", "--proposer", "llm", "--probe", "live", "--composer", "llm",
                            "--json"], home, deps=deps, capsys=capsys)
    value = _json_from(output)
    writer = store.Store(home)
    lessons = library.list_lessons(writer, "same-domain")
    assert code == 0 and value["decision"] == "CANDIDATE"
    assert len(lessons) == 1 and lessons[0].evidence.diff_lines
    assert "-Existing baseline instructions." in lessons[0].evidence.diff_lines
    assert "+Use the improved operating protocol." in lessons[0].evidence.diff_lines


def test_end_to_end_apply_then_rollback_restores_bytes(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    active = home / "harness2" / "active" / "overlay.md"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"previous active overlay\n")
    before = active.read_bytes()
    deps, _, _ = _candidate_deps()
    task = _task(tmp_path)
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "apply-demo", "--proposer", "llm",
                            "--probe", "live", "--composer", "llm", "--k", "1", "--apply", "--json"],
                           home, deps=deps, capsys=capsys)
    assert code == 0 and _json_from(output)["decision"] == "CANDIDATE"
    assert active.read_bytes() != before
    code, output = _invoke(["rollback", "--json"], home, capsys=capsys)
    assert code == 0 and _json_from(output) == {"restored": True}
    assert active.read_bytes() == before


def test_end_to_end_secret_in_probe_text_is_redacted_on_disk(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    task = _task(tmp_path)
    secret = "API_KEY=abcdef123456"
    deps = orchestrator.Deps(llm=FakeLlm(_proposal_and_selection),
                             runner=FakeRunner(lambda **args: secret if args["label"] == "base" else "safe candidate"))
    code, output = _invoke(["evolve", "--task", str(task), "--domain", "redact-demo", "--probe", "live",
                            "--dry-run", "--json"], home, deps=deps, capsys=capsys)
    value = _json_from(output)
    writer = store.Store(home)
    probe_files = list((writer.root / value["run_dir"] / "rounds/round-01/probes").glob("*.json"))
    assert code == 0 and probe_files
    assert all(secret not in path.read_text(encoding="utf-8") for path in probe_files)
    assert "redacted" in " ".join(path.read_text(encoding="utf-8") for path in probe_files)


def test_docs_exist_and_cover_limits_costs_and_exit_codes():
    doc = REPO / "docs" / "HARNESS2.md"
    text = doc.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 200
    for term in ("quickstart", "cost", "4000", "skills", "exit code", "dry-run"):
        assert term in text.lower()
