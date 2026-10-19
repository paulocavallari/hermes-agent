"""Parallel evolve orchestration tests (all dependencies are offline fakes)."""

import json
import re
from datetime import datetime, timezone, UTC

from agent.harness2 import harness, library, orchestrator, store
from agent.harness2.llm import FakeLlm
from agent.harness2.models import Decision, Dimension, Harness, Op, Proposal, Status
from agent.harness2.probe import FakeRunner
from agent.harness2.validate import validate_harness


def _task(tmp_path, text="Write a concise summary."):
    path = tmp_path / "task.md"
    path.write_text(text, encoding="utf-8")
    return path


def _deps(*, llm=None, runner=None):
    return orchestrator.Deps(
        llm=llm,
        runner=runner,
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
        monotonic=iter(range(10000)).__next__,
        token=lambda n: "a" * n,
    )


def _options(tmp_path, **kwargs):
    return orchestrator.EvolveOptions(task_path=_task(tmp_path), **kwargs)


def _reply_proposal(request):
    dimension = "INV" if "DIMENSION / INV" in request.user else "DOM"
    text = "protocol proposal" if dimension == "INV" else "domain proposal"
    return json.dumps({
        "dimension": dimension,
        "edits": [{"op": "replace", "path": "overlay.md", "new_text": text}],
        "rationale": "one textual edit",
        "keep_base": False,
    })


def _choose_preferred(request):
    options = re.findall(r"(opt-\d+) \[status=ok\]\n([^\n]+)", request.user)
    label = next(label for label, text in options if "PREFERRED" in text)
    return json.dumps({"decision": "CANDIDATE", "chosen": label, "rationale": "preferred"})


def test_parallel_writes_complete_layout(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path), home=home, deps=_deps())
    writer = store.Store(home)
    root = f"runs/{result.run_id}"
    assert result.exit_code == 0 and result.decision is Decision.KEEP_BASE
    assert writer.read_json(f"{root}/manifest.json")["run_id"] == result.run_id
    assert writer.read_text(f"{root}/TASK.md").startswith("Write a concise summary.")
    assert writer.read_bundle(f"{root}/rounds/round-01/base").harness_id == result.final_harness_id
    assert len([p for p in (writer.root / root / "rounds/round-01/proposals").glob("*.json")]) == 6
    assert len([p for p in (writer.root / root / "rounds/round-01/candidates").glob("cand-*") if p.is_dir()]) == 3
    assert len([p for p in (writer.root / root / "rounds/round-01/probes").glob("*.json")]) == 4
    for rel in ("selection.json", "reflection.json", "ledger.json",
                "rounds/round-01/selection.json", "rounds/round-01/reflection.json",
                "rounds/round-01/contrast.md", "rounds/round-01/round.json", "rounds/round-01/probes/base.json"):
        assert writer.read_text(f"{root}/{rel}")
    assert "composer_disabled" in writer.read_text(f"{root}/selection.json")


def test_parallel_makes_two_k_proposals_and_k_candidates(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path, k=2), home=home, deps=_deps())
    root = store.Store(home).root / "runs" / result.run_id / "rounds/round-01"
    assert len(list((root / "proposals").glob("*.json"))) == 4
    assert len(list((root / "candidates").glob("cand-*") )) == 2


def test_template_proposals_are_valid_on_empty_base():
    base = Harness.empty()
    context = library.LibraryContext("demo", (), (), 0)
    inv = orchestrator._propose_template("p-01", 1, Dimension.INV, base, "demo", context, ())
    after_inv, _ = harness.apply_edits(base, inv.edits)
    dom = orchestrator._propose_template("p-02", 1, Dimension.DOM, after_inv, "demo", context, ())
    candidate, _ = harness.apply_edits(after_inv, dom.edits)
    assert validate_harness(after_inv).valid
    assert validate_harness(candidate).valid


def test_proposer_wire_json_drives_positive_path(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    client = FakeLlm(_reply_proposal)
    result = orchestrator.run_evolve(_options(tmp_path, proposer="llm", k=1), home=home,
                                     deps=_deps(llm=client))
    assert result.exit_code == 0
    assert len(client.requests) == 2
    writer = store.Store(home)
    saved = writer.read_json(f"runs/{result.run_id}/rounds/round-01/proposals/p-01.json")
    assert saved["fell_back"] is False
    assert saved["edits"][0]["new_text"] == "protocol proposal"


def test_invalid_proposal_retries_then_valid_succeeds(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    calls = []

    def invalid_once(request):
        calls.append(request)
        return "not json" if len(calls) == 1 else _reply_proposal(request)

    client = FakeLlm(invalid_once)
    result = orchestrator.run_evolve(_options(tmp_path, proposer="llm", k=1), home=home,
                                     deps=_deps(llm=client))
    assert result.exit_code == 0
    assert len(calls) == 3
    assert "previous reply was invalid" in calls[1].user


def test_two_invalid_proposals_fall_back(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    client = FakeLlm(lambda request: "not json")
    result = orchestrator.run_evolve(_options(tmp_path, proposer="llm", k=1), home=home,
                                     deps=_deps(llm=client))
    writer = store.Store(home)
    proposal = writer.read_json(f"runs/{result.run_id}/rounds/round-01/proposals/p-01.json")
    assert proposal["keep_base"] is True and proposal["fell_back"] is True
    assert proposal["error"] == "invalid_payload"
    assert len(client.requests) == 4


def test_parallel_majority_selects_candidate(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    runner = FakeRunner({"base": "BASE", "cand-01": "ordinary one", "cand-02": "PREFERRED output", "cand-03": "ordinary three"})
    client = FakeLlm(_choose_preferred)
    result = orchestrator.run_evolve(_options(tmp_path, probe="live", composer="llm", k=3), home=home,
                                     deps=_deps(llm=client, runner=runner))
    assert result.exit_code == 0
    assert result.decision is Decision.CANDIDATE
    assert len(client.requests) == 3
    assert len(runner.calls) == 4


def test_default_offline_is_keep_base_exit_zero(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path), home=home, deps=_deps())
    assert result.exit_code == 0
    assert result.decision is Decision.KEEP_BASE
    assert "composer_disabled" in result.message


def test_default_offline_makes_no_llm_and_no_runner_calls(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    class NoCallRunner:
        probe_mode = "none"
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise AssertionError("probe runner must not be called")

    runner = NoCallRunner()
    client = FakeLlm([])
    result = orchestrator.run_evolve(_options(tmp_path), home=home, deps=_deps(llm=client, runner=runner))
    assert result.exit_code == 0
    assert client.requests == []
    assert runner.calls == 0


def test_base_invalid_aborts_before_any_run_write(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    base_dir = tmp_path / "invalid-base"
    base_dir.mkdir()
    (base_dir / "overlay.md").write_text("x" * 4001, encoding="utf-8")
    result = orchestrator.run_evolve(_options(tmp_path, base_dir=base_dir), home=home, deps=_deps())
    assert result.exit_code == 4
    assert store.Store(home).list_run_ids() == []


def test_secret_task_exits_6_and_writes_nothing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    path = _task(tmp_path, "API_KEY=unit-test-secret-value")
    result = orchestrator.run_evolve(orchestrator.EvolveOptions(path), home=home, deps=_deps())
    assert result.exit_code == 6
    assert store.Store(home).list_run_ids() == []
    assert list((home / "harness2").rglob("*")) == []


def test_all_candidates_invalid_exits_5(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    def invalid_candidate(request):
        if "DIMENSION / INV" in request.user:
            return json.dumps({"dimension": "INV", "edits": [{"op": "replace", "path": "overlay.md", "new_text": "no trigger"}],
                               "rationale": "remove trigger", "keep_base": False})
        return json.dumps({"dimension": "DOM", "edits": [{"op": "add", "path": "skills/bad/SKILL.md",
                              "new_text": "---\nname: bad\ndescription: Use bad\n---\nBody"}],
                           "rationale": "missing trigger", "keep_base": False})

    result = orchestrator.run_evolve(_options(tmp_path, proposer="llm", k=1), home=home,
                                     deps=_deps(llm=FakeLlm(invalid_candidate)))
    assert result.exit_code == 5
    writer = store.Store(home)
    assert writer.read_text(f"runs/{result.run_id}/rounds/round-01/candidates/cand-01.errors.json")


def test_reflection_covers_every_candidate(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    runner = FakeRunner(lambda **args: f"output for {args['label']}")
    result = orchestrator.run_evolve(_options(tmp_path, probe="live"), home=home,
                                     deps=_deps(runner=runner))
    reflection = store.Store(home).read_json(f"runs/{result.run_id}/rounds/round-01/reflection.json")
    assert [item["candidate_id"] for item in reflection["candidates"]] == ["cand-01", "cand-02", "cand-03"]
    assert all(item["classifications"] for item in reflection["candidates"])


def test_dry_run_suppresses_apply_even_when_requested(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    writer = store.Store(home)
    writer.write_text("active/overlay.md", "preserve active\n")
    before = writer.read_text("active/overlay.md")
    runner = FakeRunner({"base": "base output", "cand-01": "candidate one", "cand-02": "PREFERRED candidate two", "cand-03": "candidate three"})
    result = orchestrator.run_evolve(_options(tmp_path, probe="live", composer="llm", k=3, apply=True, dry_run=True),
                                     home=home, deps=_deps(llm=FakeLlm(_choose_preferred), runner=runner))
    assert result.decision is Decision.CANDIDATE
    assert writer.read_text("active/overlay.md") == before
    assert writer.list_domains() == []
    manifest = writer.read_json(f"runs/{result.run_id}/manifest.json")
    assert manifest["apply_requested"] is True and manifest["dry_run"] is True
    assert "active overlay untouched" in result.message


def test_ledger_counts_stages_probes_calls_and_tokens(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path), home=home, deps=_deps())
    ledger = store.Store(home).read_json(f"runs/{result.run_id}/ledger.json")
    assert ledger["stages"]["propose"]["calls"] == 6
    assert ledger["stages"]["probe"]["calls"] == 0
    assert ledger["stages"]["compose"]["calls"] == 0
    assert ledger["stages"]["reflect"]["calls"] == 1
    assert ledger["probes"] == 0 and ledger["llm_calls"] == 0
    assert ledger["tokens"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_proposer_receives_domain_lessons_and_inventory(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    writer = store.Store(home)
    prior = Harness.make("learned base")
    library.record_run(writer, "demo", prior, (), run_id="run-000000000099", decision=Decision.CANDIDATE,
                       rounds=(), created_at="2024-12-31T00:00:00Z")
    from tests.agent.harness2.test_library import _lesson
    library.append_lessons(writer, "demo", (_lesson(98, "demo"),))
    client = FakeLlm(_reply_proposal)
    result = orchestrator.run_evolve(_options(tmp_path, domain="demo", proposer="llm", k=1), home=home,
                                     deps=_deps(llm=client))
    assert result.exit_code == 0
    assert "lesson marker 98" in client.requests[0].user
    assert prior.harness_id in client.requests[0].user


def test_all_llm_failures_exit_7(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    class FailedClient:
        def complete(self, request):
            raise RuntimeError("provider unavailable")

    result = orchestrator.run_evolve(_options(tmp_path, proposer="llm", k=1), home=home,
                                     deps=_deps(llm=FailedClient()))
    assert result.exit_code == 7
    assert "E_LLM" in result.message


def test_dry_run_does_not_record_library(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path, dry_run=True), home=home, deps=_deps())
    assert result.exit_code == 0
    assert store.Store(home).list_domains() == []
    assert store.Store(home).read_text(f"runs/{result.run_id}/reflection.json")


def _sequential_llm(request):
    if request.purpose == "propose":
        if "DIMENSION / INV" in request.user:
            return json.dumps({"dimension": "INV", "edits": [{"op": "replace", "path": "overlay.md", "new_text": "sequential protocol"}],
                               "rationale": "one path", "keep_base": False})
        return json.dumps({"dimension": "DOM", "edits": [], "rationale": "keep practice unchanged", "keep_base": True})
    return json.dumps({"decision": "KEEP_BASE", "chosen": None, "rationale": "retain base"})


def test_sequential_two_rounds_reflect_at_end_of_each(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    runner = FakeRunner(lambda **args: f"output {args['label']}")
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=2, probe="live"),
                                     home=home, deps=_deps(runner=runner))
    writer = store.Store(home)
    for round_number in (1, 2):
        rel = f"runs/{result.run_id}/rounds/round-{round_number:02d}"
        reflection = writer.read_json(f"{rel}/reflection.json")
        assert len(reflection["candidates"]) == 1
        assert reflection["candidates"][0]["candidate_id"] == "cand-01"
        assert writer.read_text(f"{rel}/contrast.md")
    assert result.exit_code == 0


def test_sequential_default_offline_writes_complete_run_without_llm_or_probe(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=2), home=home, deps=_deps())
    writer = store.Store(home)
    assert result.exit_code == 0 and result.decision is Decision.KEEP_BASE
    for round_number in (1, 2):
        root = f"runs/{result.run_id}/rounds/round-{round_number:02d}"
        assert len(list((writer.root / root / "proposals").glob("*.json"))) == 2
        assert writer.read_json(f"{root}/probes/base.json")["status"] == "skipped"
        assert writer.read_json(f"{root}/selection.json")["reason_code"] == "composer_disabled"
    assert writer.read_json(f"runs/{result.run_id}/ledger.json")["llm_calls"] == 0


def test_round_summary_reaches_next_proposer_and_composer(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    client = FakeLlm(_sequential_llm)
    runner = FakeRunner(lambda **args: f"probe {args['label']}")
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=2, proposer="llm",
                                               probe="live", composer="llm"), home=home,
                                     deps=_deps(llm=client, runner=runner))
    assert result.exit_code == 0
    proposals = [request for request in client.requests if request.purpose == "propose"]
    compositions = [request for request in client.requests if request.purpose == "compose"]
    assert len(proposals) == 4 and len(compositions) == 2
    assert "round 01" in proposals[2].user
    assert "round 01" in compositions[1].user


def test_sequential_reflection_per_round_covers_all_candidates(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    runner = FakeRunner(lambda **args: f"result {args['label']}")
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=3, probe="live"),
                                     home=home, deps=_deps(runner=runner))
    writer = store.Store(home)
    reflections = [writer.read_json(f"runs/{result.run_id}/rounds/round-{n:02d}/reflection.json")
                   for n in range(1, 4)]
    assert all(len(item["candidates"]) == 1 for item in reflections)
    assert len(runner.calls) == 6


def test_sequential_carries_selected_harness_forward_and_final_decision_global(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    compose_calls = 0

    def choose_first_then_keep(request):
        nonlocal compose_calls
        if request.purpose == "propose":
            return _sequential_llm(request)
        compose_calls += 1
        if compose_calls == 1:
            return _choose_preferred(request)
        return json.dumps({"decision": "KEEP_BASE", "chosen": None, "rationale": "retain adopted base"})

    client = FakeLlm(choose_first_then_keep)
    runner = FakeRunner({"base": "base output", "cand-01": "PREFERRED output"})
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=2, proposer="llm",
                                               probe="live", composer="llm"), home=home,
                                     deps=_deps(llm=client, runner=runner))
    writer = store.Store(home)
    first = writer.read_bundle(f"runs/{result.run_id}/rounds/round-01/candidates/cand-01")
    second_base = writer.read_bundle(f"runs/{result.run_id}/rounds/round-02/base")
    run_selection = writer.read_json(f"runs/{result.run_id}/selection.json")
    assert first.harness_id == second_base.harness_id
    assert result.decision is Decision.CANDIDATE
    assert run_selection["decision"] == "CANDIDATE" and run_selection["changed"] is True


def test_sequential_keep_base_stays_on_same_base(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    client = FakeLlm(_sequential_llm)
    runner = FakeRunner(lambda **args: f"probe {args['label']}")
    result = orchestrator.run_evolve(_options(tmp_path, mode="sequential", rounds=2, proposer="llm",
                                               probe="live", composer="llm"), home=home,
                                     deps=_deps(llm=client, runner=runner))
    writer = store.Store(home)
    base1 = writer.read_bundle(f"runs/{result.run_id}/rounds/round-01/base")
    base2 = writer.read_bundle(f"runs/{result.run_id}/rounds/round-02/base")
    assert base1.harness_id == base2.harness_id == result.final_harness_id
    assert result.decision is Decision.KEEP_BASE


def test_rounds_and_k_bounds_rejected(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    cases = (
        (orchestrator.EvolveOptions(_task(tmp_path), mode="parallel", rounds=2), 2),
        (orchestrator.EvolveOptions(_task(tmp_path), mode="sequential", k=1), 2),
        (orchestrator.EvolveOptions(_task(tmp_path), mode="sequential", rounds=0), 2),
        (orchestrator.EvolveOptions(_task(tmp_path), mode="parallel", k=6), 2),
    )
    for options, expected in cases:
        assert orchestrator.run_evolve(options, home=home, deps=_deps()).exit_code == expected
