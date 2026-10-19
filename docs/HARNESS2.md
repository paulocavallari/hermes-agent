# Harness²

Harness² is an offline-first workflow for proposing, probing, composing, and reflecting on portable agent-harness changes. A harness is an always-on `overlay.md` plus optional on-demand `skills/<name>/...` files. Runs and the domain-scoped library live under `$HERMES_HOME/harness2/`.

The default is safe and offline: template proposer, no probe, no LLM composer. LLM proposer/composer and live probes are explicit opt-ins.

## Quickstart

Run from the Hermes Agent repository root. Set `HERMES_HOME` to the home you intend to use; the example below uses the normal user home.

```sh
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
printf '%s\n' 'Summarize the subject clearly.' > "$HOME/harness2-task.md"
env -u PYTHONPATH .venv/bin/python -m hermes_cli.main harness2 --help
env -u PYTHONPATH .venv/bin/python -m hermes_cli.main harness2 status --json
env -u PYTHONPATH .venv/bin/python -m hermes_cli.main harness2 evolve --task "$HOME/harness2-task.md" --domain demo --dry-run --json
env -u PYTHONPATH .venv/bin/python -m hermes_cli.main harness2 library --json
```

`--dry-run` writes a complete run record for inspection but does not change the active overlay or library. Inspect its `run_dir` from the JSON output. For an installed CLI, the corresponding command prefix is `hermes harness2`.

## Commands

```text
hermes harness2 status [--json]
hermes harness2 evolve --task PATH [--domain D] [--mode parallel|sequential]
  [--k N | --rounds N] [--proposer template|llm] [--probe none|live]
  [--composer none|llm] [--base DIR] [--seed N] [--model M]
  [--max-llm-calls N] [--dry-run] [--apply] [--json]
hermes harness2 library [--domain D] [--json]
hermes harness2 lessons --domain D [--limit N] [--json]
hermes harness2 rollback [--json]
```

Parallel mode uses `k` candidates (default 3); sequential mode uses `rounds` (default 2). Each is bounded to 1–5. `--k` is parallel-only and `--rounds` is sequential-only. The default base is the active overlay, with no skills; to reuse a complete library bundle, pass `--base "$HERMES_HOME/harness2/library/DOMAIN/harnesses/HARNESS_ID/harness"`.

## Prompt-injection gate and overlay

The active overlay is **not** injected into the system prompt by default. Injection requires `harness2.inject_overlay: true` in the user's Hermes configuration; the setting is absent/false by default. `harness2 status --json` reports `inject_overlay` and whether the active overlay is valid. With the gate off, the existing prompt is unchanged.

`evolve --apply` applies only an accepted candidate's `overlay.md` to `$HERMES_HOME/harness2/active/overlay.md`; it keeps one previous overlay at `overlay.prev`. `harness2 rollback` restores that previous overlay once. It does not install, copy, or activate skills. **Skills stay in the Harness² library for manual use and are never installed.**

`--dry-run --apply` is a simulation: `--dry-run` wins, records the requested apply in the run manifest, and leaves both the library and active files untouched. The run directory is still written.

## Model calls and costs

Harness² has no built-in USD price table; the ledger reports tokens and reports `cost_usd: null` unless a pricing function is supplied by the caller. Call counts below are the maximum planned successful-path calls; invalid model replies can be retried once, and `--max-llm-calls` defaults to 32 for budgeted proposal/composition requests.

- **Default:** `template` + `none` + `none` makes **0 model calls** and needs no network.
- **Parallel:** `k` candidates. `proposer=llm` uses `2k` proposal calls; `composer=llm` uses `k` votes/calls; `probe=live` uses `k+1` calls (base plus candidates). For default `k=3`, these are 6, 3, and 4 calls respectively when each stage is enabled.
- **Sequential:** for `R` rounds, `proposer=llm` uses `2R` calls; `composer=llm` uses `R`; `probe=live` uses `2R` (base and candidate each round). Default `R=2` gives 4, 2, and 4 calls respectively when enabled. Reflection is deterministic and uses 0 calls.
- **Live probe:** one text-only model call for each base/candidate observation. The same task is used for the compared bundles; no tools are passed. Live probes are counted in the ledger's probe stage.

`--model M` is forwarded to the configured model client for LLM work; it is not metadata-only. The ledger records the route actually reported by the client. Check provider pricing separately before opting into live or LLM stages.

## Limits and safety

- Probes are textual, single-call, and **never use tools or execute harness scripts**. There is no recorded/replay/cache mode.
- `overlay.md` is limited to 4000 characters; task and individual skill/support files are limited to 64 KiB; a bundle may contain at most 64 files and 1 MiB total; rendered system prompt is capped at 24,000 characters. Over-limit prompts are skipped, never silently truncated.
- Skills require valid frontmatter, an exact overlay trigger, safe relative paths, and existing support-file references. Symlinks, traversal, unsafe instructions, and secret-like content are rejected before probing or applying.
- The default path makes no model/network call. `--probe live`, `--proposer llm`, and `--composer llm` may send task and harness text to the configured model provider.
- Writes are confined to `$HERMES_HOME/harness2/`. Applying a candidate changes only `active/overlay.md` and its one-level backup; it never writes to `$HERMES_HOME/skills`, Hermes config, or the live checkout.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Completed; includes a valid `KEEP_BASE` decision. |
| 2 | Invalid usage/options or an out-of-range limit. |
| 3 | Task missing, unreadable, empty, or over the size limit. |
| 4 | Invalid/unreadable base bundle (including an unsafe base path). |
| 5 | Candidates were created, but all were invalid. |
| 6 | Secret-like content rejected. |
| 7 | LLM mode ended with no successful model call. |
| 8 | Confined storage or I/O error. |
| 9 | Requested domain/record is absent, or rollback has no previous overlay. |
| 10 | Apply/rollback validation error. |

## Troubleshooting

- **No model call / `KEEP_BASE`:** this is expected with defaults (`composer=none`, `probe=none`). Opt in to stages explicitly; inspect `selection.json` and `ledger.json` in the run directory.
- **No lesson:** reflection only records a lesson for an isolated, evidence-backed edit. Probe outputs must be usable and comparable, and `verify_evidence` requires the stored excerpts and real diff lines to match. Multi-path edits or inconclusive probes are classified as confounded/no-entry.
- **Exit 5:** inspect `rounds/round-NN/candidates/*.errors.json` for validation codes; fix the proposal or supply a valid `--base`.
- **Exit 6:** remove credentials/secrets from the task and base. Rejected content is not silently redacted into an accepted bundle.
- **Exit 7:** check provider/model configuration and the ledger; default offline mode is available without an LLM provider.
- **Exit 8:** check that `$HERMES_HOME` is an accessible, non-symlink home with writable storage; do not bypass path-safety errors.
- **Exit 9 on lessons/rollback:** choose a domain shown by `library --json`, or apply once before rollback so a previous overlay exists.
