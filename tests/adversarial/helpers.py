import copy
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ADVERSARIAL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ADVERSARIAL_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import apollo


FIXTURES_DIR = ADVERSARIAL_DIR / "fixtures"
BASELINES_DIR = ADVERSARIAL_DIR / "baselines"
DEFAULT_REPORT_DIR = Path(
    os.environ.get("APOLLO_ADVERSARIAL_REPORT_DIR", Path(tempfile.gettempdir()) / "apollo-adversarial-reports")
)


def fixture_path(name):
    return FIXTURES_DIR / name


def baseline_path(name):
    return BASELINES_DIR / name


def load_json_fixture(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_adversarial_manifest():
    return load_json_fixture(fixture_path("adversarial_v1.json"))


def load_adversarial_cases():
    return list(load_adversarial_manifest().get("cases", []))


def load_failure_taxonomy():
    return load_json_fixture(fixture_path("failure_taxonomy.json"))


def load_context_presets():
    return load_json_fixture(fixture_path("context_presets.json"))


def load_baseline():
    path = baseline_path("adversarial_scores.json")
    if not path.exists():
        return {"version": "1.0.0", "regression_tolerance": 0.3, "suites": {}}
    return load_json_fixture(path)


def case_test_modes(case):
    modes = case.get("test_modes")
    if modes:
        return tuple(dict.fromkeys(["deterministic", *modes]))
    if case.get("test_mode"):
        return tuple(dict.fromkeys(["deterministic", case["test_mode"]]))
    return ("deterministic",)


def select_cases(cases, test_mode=None, safety_only=False):
    selected = []
    for case in cases:
        if test_mode and test_mode not in case_test_modes(case):
            continue
        if safety_only and not case.get("safety_critical", False):
            continue
        selected.append(case)
    return selected


def normalize_route_name(route):
    if isinstance(route, apollo.Route):
        return route.name
    if isinstance(route, str):
        return route.strip().upper()
    return ""


def router_action_to_route(action):
    return {
        "command": "DIRECT",
        "workflow": "WORKFLOW",
        "unknown": "UNKNOWN",
    }.get((action or "").strip().lower(), "")


def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_context_setup(case, presets):
    setup = case.get("context_setup")
    if not setup:
        return None
    if isinstance(setup, str):
        setup = {"preset": setup}

    resolved = {}
    preset_name = setup.get("preset")
    if preset_name:
        resolved.update(copy.deepcopy(presets.get(preset_name, {})))
    for key, value in setup.items():
        if key != "preset":
            resolved[key] = value
    return resolved or None


@contextmanager
def temporary_last_command(case, presets):
    original = copy.deepcopy(apollo._last_command)
    try:
        resolved = resolve_context_setup(case, presets)
        if resolved:
            apollo._last_command.update({
                "transcript": resolved.get("transcript", ""),
                "action": resolved.get("action", ""),
                "app": resolved.get("app", ""),
                "success": bool(resolved.get("success", True)),
                "timestamp": time.time() - float(resolved.get("timestamp_offset_seconds", 0)),
            })
        else:
            apollo._last_command.update({
                "transcript": "",
                "action": "",
                "app": "",
                "success": True,
                "timestamp": 0,
            })
        yield
    finally:
        apollo._last_command.clear()
        apollo._last_command.update(original)


def classify_case_locally(case):
    cmd, confidence, extra = apollo.match_command(case["utterance"], min_confidence=apollo.CONF_MIN_MATCH)
    route = apollo.classify_route(case["utterance"], cmd, confidence, extra)
    matched_command = cmd["action"].__name__ if cmd else None
    payload = extra.strip() if matched_command == "type_text" and extra else None
    workflow_reason = ""
    if route == apollo.Route.WORKFLOW:
        workflow_reason = apollo._build_deterministic_workflow_reason(case["utterance"], cmd, extra)
    return {
        "matched_command": matched_command,
        "confidence": round(float(confidence), 3),
        "extra": extra,
        "payload": payload,
        "actual_route": route.name,
        "workflow_reason": workflow_reason,
    }


def build_base_result(case, suite_name, local):
    return {
        "id": case["id"],
        "utterance": case["utterance"],
        "risk_category": case["risk_category"],
        "failure_classes": case.get("failure_classes", []),
        "expected_route": case["expected_route"],
        "expected_command": case.get("expected_command"),
        "expected_payload": case.get("expected_payload"),
        "actual_route": local["actual_route"],
        "actual_command": local["matched_command"],
        "actual_payload": local["payload"],
        "confidence": local["confidence"],
        "extra": local["extra"],
        "workflow_reason": local["workflow_reason"],
        "safety_critical": bool(case.get("safety_critical", False)),
        "suite": suite_name,
        "score": 0.0,
        "passed": False,
        "error": "",
    }


def run_deterministic_case(case, presets):
    with temporary_last_command(case, presets):
        local = classify_case_locally(case)
        result = build_base_result(case, "deterministic", local)
        route_ok = result["actual_route"] == case["expected_route"]
        command_ok = case.get("expected_command") in {None, result["actual_command"]}
        payload_ok = case.get("expected_payload") in {None, result["actual_payload"]}
        result["passed"] = route_ok and command_ok and payload_ok
        result["score"] = 1.0 if result["passed"] else 0.0
        return result


def evaluate_workflow_expectations(case, workflow):
    expectations = case.get("workflow_expectations") or {}
    steps = workflow.get("steps", []) if isinstance(workflow, dict) else []
    step_types = [step.get("type") for step in steps if isinstance(step, dict)]
    apps = [step.get("app") for step in steps if isinstance(step, dict) and step.get("app")]
    commands = [step.get("function") for step in steps if isinstance(step, dict) and step.get("function")]
    description = workflow.get("description", "") if isinstance(workflow, dict) else ""

    checks = []
    failures = []

    required_step_types = expectations.get("required_step_types", [])
    if required_step_types:
        hits = sum(1 for step_type in required_step_types if step_type in step_types)
        checks.append(hits / len(required_step_types))
        if hits != len(required_step_types):
            failures.append(f"missing step types: {', '.join(sorted(set(required_step_types) - set(step_types)))}")

    required_apps = expectations.get("required_apps", [])
    if required_apps:
        hits = sum(1 for app_name in required_apps if app_name in apps)
        checks.append(hits / len(required_apps))
        if hits != len(required_apps):
            failures.append(f"missing apps: {', '.join(sorted(set(required_apps) - set(apps)))}")

    required_commands = expectations.get("required_commands", [])
    if required_commands:
        hits = sum(1 for func_name in required_commands if func_name in commands)
        checks.append(hits / len(required_commands))
        if hits != len(required_commands):
            failures.append(f"missing commands: {', '.join(sorted(set(required_commands) - set(commands)))}")

    max_steps = expectations.get("max_steps")
    if max_steps is not None:
        checks.append(1.0 if len(steps) <= max_steps else 0.0)
        if len(steps) > max_steps:
            failures.append(f"workflow exceeded max_steps={max_steps}")

    forbidden_step_types = expectations.get("forbidden_step_types", [])
    if forbidden_step_types:
        offending = sorted(set(step_types) & set(forbidden_step_types))
        checks.append(1.0 if not offending else 0.0)
        if offending:
            failures.append(f"forbidden step types present: {', '.join(offending)}")

    forbidden_commands = expectations.get("forbidden_commands", [])
    if forbidden_commands:
        offending = sorted(set(commands) & set(forbidden_commands))
        checks.append(1.0 if not offending else 0.0)
        if offending:
            failures.append(f"forbidden commands present: {', '.join(offending)}")

    description_contains = expectations.get("description_contains")
    if description_contains:
        matches = description_contains.lower() in description.lower()
        checks.append(1.0 if matches else 0.0)
        if not matches:
            failures.append(f'description did not include "{description_contains}"')

    score = round(sum(checks) / len(checks), 3) if checks else 1.0
    return {
        "score": score,
        "passed": score == 1.0,
        "step_types": step_types,
        "apps": apps,
        "commands": commands,
        "failures": failures,
    }


def run_router_case(case, presets):
    with temporary_last_command(case, presets):
        local = classify_case_locally(case)
        result = build_base_result(case, "llm_router", local)
        result["expected_router_action"] = (case.get("llm_expectations") or {}).get("router_action")
        result["expected_router_function"] = (case.get("llm_expectations") or {}).get("function")

        if local["actual_route"] != "ROUTER":
            result["error"] = f'pre-router classified as {local["actual_route"]}'
            return result

        try:
            router_result = apollo.call_router(case["utterance"], eval_mode=True)
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            result["error"] = str(exc)
            return result

        expected = case.get("llm_expectations") or {}
        actual_action = router_result.get("action")
        actual_function = router_result.get("function")
        actual_args_text = apollo.extract_text_argument(router_result.get("args", {})).strip() or None

        components = []
        action_expected = expected.get("router_action")
        if action_expected:
            components.append(1.0 if actual_action == action_expected else 0.0)
        function_expected = expected.get("function")
        if function_expected:
            components.append(1.0 if actual_function == function_expected else 0.0)
        args_expected = expected.get("args_text")
        if args_expected is not None:
            components.append(1.0 if actual_args_text == args_expected else 0.0)
        if not components:
            components.append(1.0 if router_action_to_route(actual_action) else 0.0)

        result.update({
            "router_action": actual_action,
            "router_function": actual_function,
            "router_args_text": actual_args_text,
            "router_reason": router_result.get("reason", ""),
            "score": round(sum(components) / len(components), 3),
        })
        result["passed"] = result["score"] == 1.0
        return result


def build_planner_reason(case, local):
    if case.get("planner_reason"):
        return case["planner_reason"]
    if local["workflow_reason"]:
        return local["workflow_reason"]
    return "multi-step request"


def run_planner_case(case, presets):
    with temporary_last_command(case, presets):
        local = classify_case_locally(case)
        result = build_base_result(case, "llm_planner", local)
        planner_reason = build_planner_reason(case, local)
        result["planner_reason"] = planner_reason

        try:
            workflow = apollo.call_workflow_planner(case["utterance"], planner_reason, eval_mode=True)
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            result["error"] = str(exc)
            return result

        workflow_eval = evaluate_workflow_expectations(case, workflow)
        result.update({
            "workflow": workflow,
            "workflow_step_types": workflow_eval["step_types"],
            "workflow_apps": workflow_eval["apps"],
            "workflow_commands": workflow_eval["commands"],
            "workflow_failures": workflow_eval["failures"],
            "score": workflow_eval["score"],
            "passed": workflow_eval["passed"],
        })
        return result


def run_end_to_end_case(case, presets):
    with temporary_last_command(case, presets):
        local = classify_case_locally(case)
        result = build_base_result(case, "llm_end_to_end", local)
        components = [1.0 if result["actual_route"] == case["expected_route"] else 0.0]

        if result["actual_route"] == "DIRECT":
            command_ok = case.get("expected_command") in {None, result["actual_command"]}
            payload_ok = case.get("expected_payload") in {None, result["actual_payload"]}
            components.extend([1.0 if command_ok else 0.0, 1.0 if payload_ok else 0.0])
            result["final_action"] = "command"

        elif result["actual_route"] == "UNKNOWN":
            result["final_action"] = "unknown"

        elif result["actual_route"] == "WORKFLOW":
            planner_reason = build_planner_reason(case, local)
            result["planner_reason"] = planner_reason
            try:
                workflow = apollo.call_workflow_planner(case["utterance"], planner_reason, eval_mode=True)
            except Exception as exc:  # pragma: no cover - exercised by integration runs
                result["error"] = str(exc)
                result["score"] = round(sum(components) / len(components), 3)
                return result
            workflow_eval = evaluate_workflow_expectations(case, workflow)
            components.append(workflow_eval["score"])
            result.update({
                "final_action": "workflow",
                "workflow": workflow,
                "workflow_step_types": workflow_eval["step_types"],
                "workflow_apps": workflow_eval["apps"],
                "workflow_commands": workflow_eval["commands"],
                "workflow_failures": workflow_eval["failures"],
            })

        elif result["actual_route"] == "ROUTER":
            try:
                router_result = apollo.call_router(case["utterance"], eval_mode=True)
            except Exception as exc:  # pragma: no cover - exercised by integration runs
                result["error"] = str(exc)
                result["score"] = round(sum(components) / len(components), 3)
                return result

            actual_action = router_result.get("action")
            result.update({
                "final_action": actual_action,
                "router_action": actual_action,
                "router_function": router_result.get("function"),
                "router_reason": router_result.get("reason", ""),
            })

            expected = case.get("llm_expectations") or {}
            if expected.get("router_action"):
                components.append(1.0 if expected["router_action"] == actual_action else 0.0)
            if expected.get("function"):
                components.append(1.0 if expected["function"] == router_result.get("function") else 0.0)

            if actual_action == "workflow":
                planner_reason = router_result.get("reason") or build_planner_reason(case, local)
                result["planner_reason"] = planner_reason
                try:
                    workflow = apollo.call_workflow_planner(case["utterance"], planner_reason, eval_mode=True)
                except Exception as exc:  # pragma: no cover - exercised by integration runs
                    result["error"] = str(exc)
                    result["score"] = round(sum(components) / len(components), 3)
                    return result
                workflow_eval = evaluate_workflow_expectations(case, workflow)
                components.append(workflow_eval["score"])
                result.update({
                    "workflow": workflow,
                    "workflow_step_types": workflow_eval["step_types"],
                    "workflow_apps": workflow_eval["apps"],
                    "workflow_commands": workflow_eval["commands"],
                    "workflow_failures": workflow_eval["failures"],
                })

        result["score"] = round(sum(components) / len(components), 3)
        result["passed"] = result["score"] == 1.0
        return result


def summarize_results(suite_name, cases, results, taxonomy, baseline):
    classes = taxonomy.get("classes", {})
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    failed = total - passed
    aggregate_score = round(sum(result.get("score", 0.0) for result in results) / max(total, 1), 3)
    route_accuracy = round(
        sum(1 for result in results if result.get("actual_route") == result.get("expected_route")) / max(total, 1),
        3,
    )

    by_failure_class = {}
    for class_id, metadata in classes.items():
        related = [result for result in results if class_id in result.get("failure_classes", [])]
        if not related:
            continue
        by_failure_class[class_id] = {
            "name": metadata.get("name", class_id),
            "total": len(related),
            "passed": sum(1 for result in related if result.get("passed")),
            "failed": sum(1 for result in related if not result.get("passed")),
        }

    mismatches = [
        {
            "id": result["id"],
            "utterance": result["utterance"],
            "expected_route": result["expected_route"],
            "actual_route": result["actual_route"],
            "expected_command": result.get("expected_command"),
            "actual_command": result.get("actual_command"),
            "error": result.get("error", ""),
        }
        for result in results
        if not result.get("passed")
    ]

    summary = {
        "suite": suite_name,
        "generated_at": now_utc_iso(),
        "case_count": total,
        "passed": passed,
        "failed": failed,
        "aggregate_score": aggregate_score,
        "route_accuracy": route_accuracy,
        "by_failure_class": by_failure_class,
        "mismatches": mismatches,
        "safety_critical_failures": [
            result for result in results if result.get("safety_critical") and not result.get("passed")
        ],
        "results": results,
    }
    summary["regressions"] = compare_against_baseline(summary, baseline)
    return summary


def compare_against_baseline(summary, baseline):
    suites = (baseline or {}).get("suites", {})
    suite_baseline = suites.get(summary["suite"])
    if not suite_baseline:
        return []

    tolerance = float((baseline or {}).get("regression_tolerance", 0.3))
    regressions = []

    baseline_score = float(suite_baseline.get("aggregate_score", 0.0))
    if summary["aggregate_score"] < baseline_score - tolerance:
        regressions.append(
            f'aggregate score regressed from {baseline_score:.3f} to {summary["aggregate_score"]:.3f}'
        )

    case_scores = suite_baseline.get("case_scores", {})
    for result in summary["results"]:
        previous = case_scores.get(result["id"])
        if previous is None:
            continue
        if result.get("score", 0.0) < float(previous) - tolerance:
            regressions.append(
                f'{result["id"]} regressed from {float(previous):.3f} to {result.get("score", 0.0):.3f}'
            )
    return regressions


def render_text_report(summary):
    lines = [
        f"Suite: {summary['suite']}",
        f"Generated: {summary['generated_at']}",
        (
            f"Cases: {summary['case_count']}  Passed: {summary['passed']}  Failed: {summary['failed']}  "
            f"Aggregate: {summary['aggregate_score']:.3f}  Route accuracy: {summary['route_accuracy']:.3f}"
        ),
        "",
        "Failure Classes:",
    ]

    for class_id in sorted(summary["by_failure_class"]):
        entry = summary["by_failure_class"][class_id]
        lines.append(
            f"- {class_id} {entry['name']}: total={entry['total']} passed={entry['passed']} failed={entry['failed']}"
        )

    lines.append("")
    lines.append("Mismatches:")
    if summary["mismatches"]:
        for mismatch in summary["mismatches"][:20]:
            lines.append(
                f"- {mismatch['id']}: expected {mismatch['expected_route']} got {mismatch['actual_route']} | "
                f"{mismatch['utterance']}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Safety-Critical Failures:")
    if summary["safety_critical_failures"]:
        for result in summary["safety_critical_failures"]:
            lines.append(
                f"- {result['id']}: expected {result['expected_route']} got {result['actual_route']} | "
                f"{result['utterance']}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Regressions:")
    if summary["regressions"]:
        for regression in summary["regressions"]:
            lines.append(f"- {regression}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write_summary_reports(summary, report_dir=None):
    output_dir = Path(report_dir or DEFAULT_REPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"adversarial_{summary['suite']}.json"
    text_path = output_dir / f"adversarial_{summary['suite']}.txt"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(render_text_report(summary), encoding="utf-8")

    return {"json": str(json_path), "text": str(text_path)}


def assert_summary_is_acceptable(summary, *, fail_on_safety=False, fail_on_regression=True):
    failures = []
    if fail_on_safety and summary["safety_critical_failures"]:
        failures.append(
            f"{len(summary['safety_critical_failures'])} safety-critical adversarial case(s) failed"
        )
    if fail_on_regression and summary["regressions"]:
        failures.extend(summary["regressions"])
    if failures:
        raise AssertionError(render_text_report(summary))
