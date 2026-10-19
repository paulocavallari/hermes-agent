"""CLI surface for the opt-in Harness² workflow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def build_harness2_parser(subparsers, *, cmd_harness2):
    parser = subparsers.add_parser("harness2", help="evolve and inspect portable agent harnesses")
    commands = parser.add_subparsers(dest="harness2_action", required=True)

    status = commands.add_parser("status", help="show Harness² state")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_harness2, harness2_action="status")

    evolve = commands.add_parser("evolve", help="propose, probe, and compose harness changes")
    evolve.add_argument("--task", type=Path, required=True)
    evolve.add_argument("--domain", default="default")
    evolve.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    evolve.add_argument("--k", type=int)
    evolve.add_argument("--rounds", type=int)
    evolve.add_argument("--proposer", choices=("template", "llm"), default="template")
    evolve.add_argument("--probe", choices=("none", "live"), default="none")
    evolve.add_argument("--composer", choices=("none", "llm"), default="none")
    evolve.add_argument("--base", type=Path)
    evolve.add_argument("--seed", type=int, default=0)
    evolve.add_argument("--model")
    evolve.add_argument("--max-llm-calls", type=int, default=32)
    evolve.add_argument("--apply", action="store_true")
    evolve.add_argument("--dry-run", action="store_true")
    evolve.add_argument("--json", action="store_true")
    evolve.set_defaults(func=cmd_harness2, harness2_action="evolve")

    listing = commands.add_parser("library", help="list stored harnesses and lessons")
    listing.add_argument("--domain")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_harness2, harness2_action="library")

    lessons = commands.add_parser("lessons", help="list lessons for one known domain")
    lessons.add_argument("--domain", required=True)
    lessons.add_argument("--limit", type=int, default=20)
    lessons.add_argument("--json", action="store_true")
    lessons.set_defaults(func=cmd_harness2, harness2_action="lessons")

    rollback = commands.add_parser("rollback", help="restore the previous active overlay")
    rollback.add_argument("--json", action="store_true")
    rollback.set_defaults(func=cmd_harness2, harness2_action="rollback")
    return parser


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _human(rows) -> None:
    for key, value in rows:
        print(f"{key}: {value}")


def _error(message: str) -> None:
    print(f"harness2: {message}", file=sys.stderr)


def _make_llm_client(model: str | None):
    from agent.harness2.orchestrator import AuxLlmClient

    return AuxLlmClient(model=model)


def _active_overlay_state(home: Path, active):
    from agent.harness2 import store

    present = active is not None
    if not present:
        target = home / "harness2" / "active" / "overlay.md"
        try:
            store.confine_ancestors(target, anchor=home)
            os.lstat(target)
            present = True
        except FileNotFoundError:
            present = False
        except Exception:  # health: allow BLE001 -- unsafe active-state paths report present-but-invalid
            present = True
    return {"present": present, "chars": active.chars if active is not None else 0, "valid": active is not None}


def _status(home: Path, *, as_json: bool) -> int:
    from agent.harness2 import overlay, store
    from hermes_cli.config import load_config_readonly

    writer = store.Store(home)
    try:
        config = load_config_readonly()
    except Exception:  # health: allow BLE001 -- absent/unreadable config keeps optional injection disabled
        config = {}
    harness2_config = config.get("harness2") if isinstance(config, dict) else None
    inject = isinstance(harness2_config, dict) and bool(harness2_config.get("inject_overlay", False))
    active = overlay.load_active(home)
    run_ids = writer.list_run_ids()
    domains = writer.list_domains()
    last_run = None
    for run_id in run_ids:
        try:
            manifest = writer.read_json(f"runs/{run_id}/manifest.json")
        except (FileNotFoundError, ValueError, OSError):
            continue
        candidate = {"run_id": run_id, "decision": manifest.get("decision"), "created_at": manifest.get("created_at")}
        if last_run is None or str(candidate.get("created_at", "")) > str(last_run.get("created_at", "")):
            last_run = candidate
    payload = {
        "home": str(home),
        "inject_overlay": inject,
        "active_overlay": _active_overlay_state(home, active),
        "runs": len(run_ids),
        "domains": len(domains),
        "last_run": last_run,
    }
    if as_json:
        _json(payload)
    else:
        _human((("home", payload["home"]), ("inject_overlay", inject),
                ("active_overlay", payload["active_overlay"]), ("runs", len(run_ids)),
                ("domains", len(domains)), ("last_run", last_run)))
    return 0


def _library(home: Path, domain: str | None, *, as_json: bool) -> int:
    from agent.harness2 import library, store

    writer = store.Store(home)
    domains = writer.list_domains()
    if domain is not None and domain not in domains:
        if as_json:
            _json({"error": "E_NOT_FOUND", "domain": domain})
        _error(f"E_NOT_FOUND: unknown domain {domain}")
        return 9
    selected = [domain] if domain else domains
    rows = []
    for item in selected:
        rows.append({
            "domain": item,
            "harnesses": len(library._all_harness_ids(writer, item)),
            "lessons": len(library._all_lessons(writer, item)),
        })
    if as_json:
        _json({"domains": rows})
    else:
        if not rows:
            _human((("domains", 0),))
        for row in rows:
            _human((("domain", row["domain"]), ("harnesses", row["harnesses"]), ("lessons", row["lessons"])))
    return 0


def _lessons(home: Path, domain: str, limit: int, *, as_json: bool) -> int:
    from agent.harness2 import library, store

    writer = store.Store(home)
    if domain not in writer.list_domains():
        if as_json:
            _json({"error": "E_NOT_FOUND", "domain": domain})
        _error(f"E_NOT_FOUND: unknown domain {domain}")
        return 9
    if type(limit) is not int or not 0 <= limit <= 20:
        _error("E_USAGE: limit must be in 0..20")
        return 2
    entries = library.list_lessons(writer, domain, limit=limit)
    payload = {"domain": domain, "lessons": [item.to_dict() for item in entries], "total": len(entries)}
    if as_json:
        _json(payload)
    else:
        _human((("domain", domain), ("total", len(entries))))
        for item in entries:
            print(f"{item.lesson_id}: {item.cue} — {item.lesson}")
    return 0


def _rollback(home: Path, *, as_json: bool) -> int:
    from agent.harness2 import overlay, store

    writer = store.Store(home)
    try:
        restored = overlay.rollback_overlay(writer)
    except overlay.ApplyError:
        if as_json:
            _json({"restored": False})
        _error("E_APPLY: invalid previous overlay")
        return 10
    if as_json:
        _json({"restored": restored})
    else:
        _human((("restored", restored),))
    if not restored:
        _error("E_NOT_FOUND: no previous overlay")
        return 9
    return 0


def _evolve(args, home: Path, deps) -> int:
    from agent.harness2 import orchestrator

    options = orchestrator.EvolveOptions(
        task_path=args.task,
        domain=args.domain,
        mode=args.mode,
        k=args.k,
        rounds=args.rounds,
        proposer=args.proposer,
        probe=args.probe,
        composer=args.composer,
        base_dir=args.base,
        seed=args.seed,
        apply=args.apply,
        dry_run=args.dry_run,
        model=args.model,
        max_llm_calls=args.max_llm_calls,
    )
    resolved_deps = deps if deps is not None else orchestrator.Deps(llm_factory=_make_llm_client)
    result = orchestrator.run_evolve(options, home=home, deps=resolved_deps)
    payload = {
        "run_id": result.run_id,
        "decision": result.decision.value,
        "run_dir": result.run_dir,
        "exit_code": result.exit_code,
        "message": result.message,
    }
    if args.json:
        _json(payload)
    else:
        _human(payload.items())
    if result.exit_code != 0:
        _error(result.message)
    return result.exit_code


def cmd_harness2(args, *, deps=None) -> int:
    """Execute one Harness² command; dependencies are injectable for offline tests."""
    from hermes_constants import get_hermes_home
    from agent.harness2 import overlay, store

    home = Path(get_hermes_home())
    action = getattr(args, "harness2_action", "")
    as_json = bool(getattr(args, "json", False))
    try:
        if action == "status":
            return _status(home, as_json=as_json)
        if action == "evolve":
            return _evolve(args, home, deps)
        if action == "library":
            return _library(home, getattr(args, "domain", None), as_json=as_json)
        if action == "lessons":
            return _lessons(home, args.domain, args.limit, as_json=as_json)
        if action == "rollback":
            return _rollback(home, as_json=as_json)
        _error("E_USAGE: unknown command")
        return 2
    except store.SecretError:
        _error("E_SECRET: rejected content")
        return 6
    except (store.PathError, OSError, ValueError):
        _error("E_IO: confined Harness² storage error")
        return 8
