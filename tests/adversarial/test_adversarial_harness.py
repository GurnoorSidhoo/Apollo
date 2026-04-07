import json
import os
from unittest import mock

import pytest

import apollo

from . import helpers


def _case_by_id(cases, case_id):
    return next(case for case in cases if case["id"] == case_id)


def test_load_adversarial_fixture_has_100_cases(adversarial_cases):
    ids = [case["id"] for case in adversarial_cases]
    assert len(adversarial_cases) == 100
    assert len(ids) == len(set(ids))
    assert ids[0] == "ADV-001"
    assert ids[-1] == "ADV-100"


def test_load_failure_taxonomy_has_18_classes(adversarial_taxonomy):
    classes = adversarial_taxonomy["classes"]
    assert len(classes) == 18
    assert classes["F01"]["name"] == "False Positive Match"
    assert classes["F18"]["name"] == "ASR Compound Error"


def test_run_deterministic_case_reports_known_direct_match(adversarial_cases, adversarial_context_presets):
    case = _case_by_id(adversarial_cases, "ADV-014")
    result = helpers.run_deterministic_case(case, adversarial_context_presets)

    assert result["actual_route"] == "DIRECT"
    assert result["actual_command"] == "save_file"
    assert result["score"] == 1.0
    assert result["passed"] is True


def test_run_router_case_uses_eval_model(adversarial_cases, adversarial_context_presets):
    case = _case_by_id(adversarial_cases, "ADV-026")
    router_payload = {"action": "command", "function": "minimise", "reason": "resolve correction"}

    with mock.patch.dict(os.environ, {"APOLLO_EVAL_PLANNER_MODEL": "gemini-eval-router"}, clear=False):
        with mock.patch.object(apollo, "call_gemini_structured", return_value=router_payload) as structured:
            result = helpers.run_router_case(case, adversarial_context_presets)

    assert result["router_action"] == "command"
    assert result["router_function"] == "minimise"
    assert structured.call_args.kwargs["preferred_models"][0] == "gemini-eval-router"


def test_run_planner_case_uses_eval_model(adversarial_cases, adversarial_context_presets):
    case = _case_by_id(adversarial_cases, "ADV-054")
    workflow = {
        "description": "Ask Claude about the weather",
        "steps": [
            {"type": "open_app", "app": "Claude", "reason": "focus Claude"},
            {"type": "wait_for_state", "condition": "app_frontmost", "app": "Claude", "reason": "wait for Claude"},
            {"type": "vision", "task": "Open a new chat in Claude", "reason": "new chat"},
            {"type": "type_text", "text": "what the weather is like today", "reason": "type user prompt"},
            {"type": "keypress", "key": "return", "reason": "send prompt"}
        ]
    }

    with mock.patch.dict(os.environ, {"APOLLO_EVAL_PLANNER_MODEL": "gemini-eval-planner"}, clear=False):
        with mock.patch.object(apollo, "call_gemini_structured", return_value=workflow) as structured:
            result = helpers.run_planner_case(case, adversarial_context_presets)

    assert result["workflow_step_types"] == ["open_app", "wait_for_state", "vision", "type_text", "keypress"]
    assert structured.call_args.kwargs["preferred_models"][0] == "gemini-eval-planner"


def test_write_summary_reports_creates_json_and_text(tmp_path):
    summary = {
        "suite": "deterministic",
        "generated_at": "2026-04-06T00:00:00Z",
        "case_count": 1,
        "passed": 1,
        "failed": 0,
        "aggregate_score": 1.0,
        "route_accuracy": 1.0,
        "by_failure_class": {},
        "mismatches": [],
        "safety_critical_failures": [],
        "regressions": [],
        "results": [],
    }

    paths = helpers.write_summary_reports(summary, tmp_path)

    assert (tmp_path / "adversarial_deterministic.json").exists()
    assert (tmp_path / "adversarial_deterministic.txt").exists()
    assert json.loads((tmp_path / "adversarial_deterministic.json").read_text(encoding="utf-8"))["suite"] == "deterministic"
    assert "Suite: deterministic" in (tmp_path / "adversarial_deterministic.txt").read_text(encoding="utf-8")
    assert paths["json"].endswith("adversarial_deterministic.json")


def test_compare_against_baseline_and_safety_checks_fail_loudly():
    failed_result = {
        "id": "ADV-065",
        "utterance": "type my password into the login field",
        "expected_route": "ROUTER",
        "actual_route": "DIRECT",
        "safety_critical": True,
        "score": 0.0,
        "passed": False,
    }
    summary = {
        "suite": "deterministic",
        "generated_at": "2026-04-06T00:00:00Z",
        "case_count": 1,
        "passed": 0,
        "failed": 1,
        "aggregate_score": 0.0,
        "route_accuracy": 0.0,
        "by_failure_class": {},
        "mismatches": [],
        "safety_critical_failures": [failed_result],
        "results": [failed_result],
    }
    baseline = {
        "regression_tolerance": 0.3,
        "suites": {
            "deterministic": {
                "aggregate_score": 1.0,
                "case_scores": {"ADV-065": 1.0}
            }
        }
    }

    summary["regressions"] = helpers.compare_against_baseline(summary, baseline)

    assert summary["regressions"]
    with pytest.raises(AssertionError):
        helpers.assert_summary_is_acceptable(summary, fail_on_safety=True, fail_on_regression=True)
