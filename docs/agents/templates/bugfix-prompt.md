# Apollo Bugfix Prompt Template

## Goal
- Fix: `<bug summary>`

## Repro
- Transcript or action: `<input>`
- Current behavior: `<actual>`
- Expected behavior: `<expected>`

## Scope
- Likely layer: `<match | pre-router | router | planner | workflow | vision | audio>`
- Files to inspect first: `<paths>`

## Constraints
- Prefer deterministic logic if the failure is repeatable.
- Preserve routing order, schema validation, AX-first behavior, and confidence gates.
- Do not widen prompts to mask a deterministic bug.

## Required Tests
- Update: `CLASSIFY_ROUTE_CASES` if routing changes.
- Add: one regression test for the bug.
- Run: `pytest tests/test_apollo_reliability.py -v`
- Run if control flow changed broadly: `pytest tests/ -v`

## Done When
- Root cause is explained briefly.
- The regression test fails before the fix and passes after.
- Any remaining risk is called out explicitly.
