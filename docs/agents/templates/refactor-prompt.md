# Apollo Refactor Prompt Template

## Goal
- Refactor: `<area>`

## Why
- Current pain: `<duplication | drift | unsafe coupling | hard to test>`
- Non-goals: `<what must not change>`

## Guardrails
- Preserve external behavior unless tests are intentionally updated.
- Keep deterministic routing decisions ahead of LLM behavior.
- Keep structured-output schemas and validators in sync.
- Do not weaken vision safety checks or remove debug instrumentation without reason.

## Plan
1. Identify the exact functions and seams to move.
2. State the behavior contracts that must stay unchanged.
3. Add or update tests before moving risky logic.
4. Refactor in small steps and verify after each step.

## Required Verification
- Targeted tests: `<commands>`
- Full repo tests if routing/execution paths changed: `pytest tests/ -v`

## Done When
- Behavior is unchanged or intentionally documented.
- Tests pass.
- Remaining technical debt or follow-up work is noted briefly.
