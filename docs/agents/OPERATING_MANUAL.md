# Apollo Agent Operating Manual

## 1. Architecture Guide

- `apollo/__init__.py` is the compatibility shim and public namespace. Tests patch symbols on `import apollo`, so preserve that surface when moving logic.
- Command registration starts with `COMMANDS = []` and `@apollo.command(...)`. Built-ins live in `apollo/commands.py`; user extensions live in `custom_commands.py`.
- Audio entrypoints are `AudioListener` and `WhisperAudioListener` in `apollo/audio.py`.
- Deterministic matching and dispatch live in `apollo/routing.py`, including `match_command()`, `classify_route()`, and `route_command()`.
- Gemini router and workflow planning live in `apollo/planner.py`.
- Workflow execution and replanning live in `apollo/workflow.py`.
- Vision capture, AX targeting, coordinate resolution, and verification live in `apollo/vision.py`.
- AppleScript, Accessibility, and app-control helpers live in `apollo/macos.py`.
- Runtime startup and text mode live in `apollo/main.py` and `apollo/__main__.py`.
- Config, thresholds, log paths, and feature flags live in `apollo/config.py`.
- Schemas, enums, and validation helpers live in `apollo/types.py`.
- Core regression coverage lives in `tests/test_apollo_reliability.py`; broader scenario coverage lives in `tests/adversarial/`.

## 2. Rules For Safe Changes

- Preserve the routing priority chain: deterministic matching and deterministic workflow extraction should stay ahead of LLM interpretation.
- Treat config-backed thresholds and limits in `apollo/config.py` as behavior contracts. Do not tune them casually.
- Preserve the `import apollo` call-time access pattern for symbols that tests patch with `mock.patch.object(apollo, ...)`.
- Keep router, planner, and vision response schemas aligned with their validators in `apollo/types.py`, `apollo/planner.py`, and `apollo/gemini.py`.
- Preserve AX-first UI targeting and postcondition verification in `apollo/vision.py`.
- Do not hardcode local API keys, local paths, or machine-specific overrides.
- Keep logging code intact, but do not commit generated runtime artifacts such as `apollo_debug.jsonl` or `apollo_vision_debug/`.

## 3. Deterministic vs Prompt-Only Fixes

- Prefer deterministic fixes when the failure is repeatable: normalization, regex, keyword sets, extraction helpers, thresholds, or validator logic.
- Good deterministic targets in this repo include wake word detection, fuzzy command matching, negation handling, click/open/quit extractors, and workflow routing guards.
- Prefer prompt changes only when the request is genuinely ambiguous or requires open-ended planning after deterministic routing has done its job.
- Do not widen prompts to hide a deterministic routing bug.

## 4. Mandatory Testing Expectations

- Any routing or threshold change must update or extend `CLASSIFY_ROUTE_CASES` in `tests/test_apollo_reliability.py`.
- Any router/planner/schema change must add or update validator coverage.
- Any new command should get at least one direct match assertion and one route-classification case.
- Any workflow or vision change should get a happy-path test and a failure-path test.
- Any audio-capture or wake-word change should preserve threaded-behavior coverage using `ImmediateThread`.
- Default targeted verification: `pytest tests/test_apollo_reliability.py -v`
- Wider regression run for broader control-flow changes: `pytest tests -v`
- If you cannot run a required test, say exactly what was skipped and why.

## 5. Change Checklist

1. Identify the layer you are changing: audio, routing, planner, workflow, vision, or command registration.
2. Confirm whether the fix should be deterministic before touching prompts.
3. Check which config values, schemas, validators, or feature flags are affected.
4. Update tests before or alongside the code for risky routing behavior.
5. Preserve AX-first behavior, confidence gates, and postcondition verification for UI actions.
6. Verify that built-in commands, custom commands, and planner-visible function names still agree.
7. Run the narrowest relevant test file, then run the broader test suite when the change crosses module boundaries.
8. Remove any local debug artifacts before finishing.

## 6. Templates

- Use `docs/agents/templates/bugfix-prompt.md` for bug work.
- Use `docs/agents/templates/refactor-prompt.md` before moving logic across routing, workflow, or vision boundaries.
- Use `docs/agents/templates/risk-note.md` when routing order, thresholds, schemas, UI automation, or concurrency changed.
