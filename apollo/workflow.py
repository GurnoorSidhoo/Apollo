"""
Apollo workflow execution — run structured multi-step workflows emitted
by the planner, with replan support on failure.

**Lazy import pattern**: Cross-references to functions that live in the
``apollo`` top-level namespace (and are commonly mock-patched via
``mock.patch.object(apollo, "X")``) are accessed through a module-level
``import apollo`` at call time.  This keeps monkey-patching in tests
working correctly while avoiding circular imports at module load time.
"""

import json
import time

from apollo.config import (
    AI_TRACE_LOG,
    GEMINI_PLANNER_MODEL,
    MAX_WORKFLOW_REPLANS,
    MAX_WORKFLOW_STEPS,
    WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL,
    WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS,
    WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS,
    WORKFLOW_PLANNER_TIMEOUT_SECONDS,
)
from apollo.logging_utils import debug_event
from apollo.types import (
    StructuredOutputError,
    WORKFLOW_REPLAN_RESPONSE_JSON_SCHEMA,
)
from apollo.gemini import model_candidates
from apollo.utils import is_quota_error, extract_retry_delay_seconds


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

def execute_registered_command(func_name, transcript, args=None, source="llm"):
    """Execute a registered command by function name, supporting type_text args."""
    import apollo

    cmd = apollo.find_registered_command(func_name)
    if not cmd:
        print(f"  !! {source.upper()} referenced unknown function: {func_name}")
        apollo.say("Sorry, I couldn't find that command")
        return False

    print(f"  [robot] {source.upper()} matched command: {func_name}")
    try:
        if cmd["action"].__name__ == "type_text":
            text = apollo.extract_text_argument(args)
            if not text:
                raise ValueError("type_text requires args.text")
            print(f'  [keyboard] Typing: "{text}"')
            apollo.type_string(text)
        else:
            cmd["action"]()
        apollo.log_command(transcript, f"{source}:{func_name}")
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"  !! Error executing {func_name}: {e}")
        debug_event(f"{source}_command_error", function=func_name, error=str(e))
        apollo.say("Sorry, something went wrong")
        return False


def _check_step_postcondition(step, target_app):
    """Verify a workflow step's expected_postcondition when present.

    Returns True when verified or when no postcondition is specified.
    Returns False when the postcondition check fails.
    """
    import apollo

    postcondition = (step.get("expected_postcondition") or "").strip()
    if not postcondition:
        return True

    app_name = target_app or step.get("app", "").strip()
    if not app_name:
        debug_event("step_postcondition_skip", reason="no_app", postcondition=postcondition)
        return True

    # Build a wait_for_state condition from the postcondition text when possible
    # Try element_exists first (covers "chat open", "model selected", "prompt present")
    from apollo.macos import (
        extract_target_label_from_text,
        condition_app_frontmost,
        condition_element_exists,
        condition_element_value_contains,
        wait_for_state,
    )
    from apollo.utils import extract_target_role_from_text, extract_first_quoted_text

    label = extract_target_label_from_text(postcondition)
    role = extract_target_role_from_text(postcondition)
    quoted = extract_first_quoted_text(postcondition)

    condition_fn = None
    condition_name = ""

    if label or role:
        condition_fn = condition_element_exists(app_name, label or "", role or "")
        condition_name = f"postcondition:element_exists:{app_name}:{label or role}"
    elif quoted:
        condition_fn = condition_element_value_contains(app_name, quoted)
        condition_name = f"postcondition:value_contains:{app_name}:{quoted}"
    elif any(kw in postcondition.lower() for kw in ("frontmost", "foreground", "focused", "active")):
        condition_fn = condition_app_frontmost(app_name)
        condition_name = f"postcondition:app_frontmost:{app_name}"

    if condition_fn is None:
        debug_event("step_postcondition_skip", reason="no_matching_heuristic", postcondition=postcondition)
        return True

    ok = wait_for_state(
        condition_fn,
        timeout_seconds=WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS,
        poll_interval=WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL,
        condition_name=condition_name,
    )
    debug_event(
        "step_postcondition_result",
        postcondition=postcondition,
        app=app_name,
        satisfied=ok,
        condition_name=condition_name,
    )
    return ok


def summarize_completed_steps(completed_steps):
    """Return a compact summary of completed workflow steps for replanning."""
    summary = []
    for item in completed_steps:
        if not isinstance(item, dict):
            continue
        step = item.get("step", {})
        summary.append({
            "index": item.get("index"),
            "type": step.get("type"),
            "result": item.get("result", "ok"),
        })
    return summary


def build_replan_user_prompt(goal, workflow, failed_step, failure_details, completed_steps):
    """Build a compact deterministic replan prompt payload."""
    completed_indexes = {
        item.get("index")
        for item in completed_steps
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    remaining_steps = [
        step
        for index, step in enumerate(workflow.get("steps", []), start=1)
        if index not in completed_indexes
    ]
    remaining_context = {
        "description": workflow.get("description", ""),
        "steps": remaining_steps,
    }
    failure_payload = {
        "code": failure_details.get("reason", "workflow_failure"),
        "message": failure_details.get("message", "Workflow step failed"),
    }
    return "\n".join([
        f"goal={json.dumps(goal, ensure_ascii=True)}",
        f"remaining={json.dumps(remaining_context, ensure_ascii=True)}",
        f"completed={json.dumps(summarize_completed_steps(completed_steps), ensure_ascii=True)}",
        f"failed_step={json.dumps(failed_step, ensure_ascii=True)}",
        f"failure={json.dumps(failure_payload, ensure_ascii=True)}",
        "Resume from the current app state. Do not repeat completed steps unless recovery requires it.",
    ])


def replan_workflow(transcript, workflow, failed_step, failure_details, completed_steps):
    """Ask the planner for a revised workflow after a step fails."""
    import apollo

    from apollo.gemini import call_gemini_structured
    from apollo.planner import build_workflow_planner_system_prompt, validate_workflow_output

    try:
        return call_gemini_structured(
            system_instruction=build_workflow_planner_system_prompt(
                transcript,
                f"workflow replan after failure: {failure_details.get('reason', 'workflow_failure')}",
            ),
            user_text=build_replan_user_prompt(transcript, workflow, failed_step, failure_details, completed_steps),
            response_json_schema=WORKFLOW_REPLAN_RESPONSE_JSON_SCHEMA,
            validator=validate_workflow_output,
            preferred_models=model_candidates(GEMINI_PLANNER_MODEL, "gemini-2.5-pro", "gemini-2.5-flash"),
            call_type="replan",
            max_output_tokens=WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS,
            timeout_seconds=WORKFLOW_PLANNER_TIMEOUT_SECONDS,
            trace_context={
                "transcript": transcript,
                "failed_step": failed_step,
                "failure_details": failure_details,
                "completed_steps": summarize_completed_steps(completed_steps),
            },
        )
    except StructuredOutputError as exc:
        apollo.planner_failure(
            "replan",
            exc.category,
            transcript=transcript,
            error=str(exc),
            model=exc.model,
            attempt=exc.attempt,
            correlation_id=exc.correlation_id,
            raw_response=exc.raw_response,
        )
        raise


def execute_workflow_once(workflow, transcript):
    """Run a structured multi-step workflow once and report failure details."""
    import apollo

    steps = workflow.get("steps", [])
    description = workflow.get("description", "Working on it")
    if not isinstance(steps, list) or not steps:
        return False, {"reason": "empty_workflow", "message": "The workflow contained no steps"}
    if len(steps) > MAX_WORKFLOW_STEPS:
        return False, {"reason": "workflow_too_long", "message": f"Workflow exceeded {MAX_WORKFLOW_STEPS} steps"}

    print(f"  [robot] Workflow: {description}")
    if description:
        apollo.say(description)

    completed_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            debug_event("workflow_step_invalid", index=index, step=step)
            return False, {
                "reason": "invalid_step",
                "message": "Workflow step was not a JSON object",
                "index": index,
                "step": step,
                "completed_steps": completed_steps,
            }

        step_type = step.get("type", "")
        optional = bool(step.get("optional"))
        step_reason = step.get("reason", "")
        debug_event("workflow_step_start", index=index, step=step, reason=step_reason)

        try:
            if step_type == "say":
                apollo.say(step.get("text", ""))
                ok = True
            elif step_type == "open_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("open_app requires app")
                apollo.mac_open_app(app_name, fallback_url=step.get("fallback_url"))
                apollo.focus_app(app_name)
                ok = True
            elif step_type == "focus_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("focus_app requires app")
                apollo.focus_app(app_name)
                ok = True
            elif step_type == "quit_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("quit_app requires app")
                apollo.quit_named_app(app_name)
                ok = True
            elif step_type == "open_url":
                url = step.get("url", "").strip()
                if not url:
                    raise ValueError("open_url requires url")
                apollo.mac_open(url)
                ok = True
            elif step_type == "wait":
                seconds = max(0.0, min(float(step.get("seconds", 1)), 10.0))
                time.sleep(seconds)
                ok = True
            elif step_type == "wait_for_state":
                condition_fn, condition_name = apollo.build_wait_for_state_condition(step)
                ok = apollo.wait_for_state(
                    condition_fn,
                    timeout_seconds=step.get("timeout_seconds", WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS),
                    poll_interval=step.get("poll_interval", WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL),
                    condition_name=condition_name,
                )
            elif step_type == "type_text":
                text = apollo.extract_text_argument(step.get("text"))
                if not text:
                    raise ValueError("type_text requires text")
                target_app = step.get("app", "").strip()
                if target_app and not apollo.ensure_text_input_focused(target_app):
                    debug_event("type_text_no_focus", app=target_app, text=text[:80])
                print(f'  [keyboard] Typing: "{text}"')
                apollo.type_string(text)
                ok = True
            elif step_type == "keypress":
                key = step.get("key", "").strip()
                if not key:
                    raise ValueError("keypress requires key")
                apollo.press_key(
                    key,
                    command=bool(step.get("command")),
                    shift=bool(step.get("shift")),
                    ctrl=bool(step.get("ctrl")),
                    option=bool(step.get("option")),
                )
                ok = True
            elif step_type == "command":
                func_name = step.get("function", "").strip()
                if not func_name:
                    raise ValueError("command requires function")
                ok = apollo.execute_registered_command(func_name, transcript, step.get("args"), source="workflow")
            elif step_type == "vision":
                task = step.get("task", "").strip()
                if not task:
                    raise ValueError("vision requires task")
                ok = apollo.execute_vision_task(task, transcript)
            else:
                raise ValueError(f"Unsupported workflow step type: {step_type}")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  !! Workflow step {index} failed: {e}")
            debug_event("workflow_step_error", index=index, step=step, reason=step_reason, error=str(e))
            if optional:
                completed_steps.append({"index": index, "step": step, "optional": True, "result": "skipped"})
                continue
            return False, {
                "reason": "step_exception",
                "message": str(e),
                "index": index,
                "step": step,
                "completed_steps": completed_steps,
            }

        if not ok and not optional:
            debug_event("workflow_step_failed", index=index, step=step, reason=step_reason)
            return False, {
                "reason": "step_returned_false",
                "message": "Workflow step returned False",
                "index": index,
                "step": step,
                "completed_steps": completed_steps,
            }

        # Postcondition check for steps that interact with external UI
        if ok and step.get("expected_postcondition"):
            step_app = step.get("app", "").strip()
            if not _check_step_postcondition(step, step_app) and not optional:
                debug_event(
                    "workflow_step_postcondition_failed",
                    index=index,
                    step=step,
                    postcondition=step.get("expected_postcondition"),
                )
                return False, {
                    "reason": "postcondition_failed",
                    "message": f"Postcondition not satisfied: {step.get('expected_postcondition')}",
                    "index": index,
                    "step": step,
                    "completed_steps": completed_steps,
                }

        completed_steps.append({"index": index, "step": step, "result": "ok" if ok else "skipped"})

    return True, {"completed_steps": completed_steps}


def execute_workflow(workflow, transcript):
    """Run a structured multi-step workflow emitted by the planner with replans."""
    import apollo

    current_workflow = workflow
    for attempt in range(MAX_WORKFLOW_REPLANS + 1):
        ok, details = apollo.execute_workflow_once(current_workflow, transcript)
        if ok:
            apollo.log_command(transcript, "llm:workflow")
            return True

        failure_reason = details.get("message", details.get("reason", "unknown workflow failure"))
        if is_quota_error(failure_reason):
            retry_delay = extract_retry_delay_seconds(failure_reason)
            debug_event("workflow_quota_exhausted", attempt=attempt + 1, retry_delay=retry_delay, details=details)
            message = "Gemini quota was exceeded during the workflow"
            if retry_delay:
                message += f". Retry in about {int(round(retry_delay))} seconds"
            print(f"  !! {message}. See {AI_TRACE_LOG}")
            apollo.say(message)
            return False

        if attempt >= MAX_WORKFLOW_REPLANS:
            break

        failed_step = details.get("step", {})
        completed_steps = details.get("completed_steps", [])
        try:
            print(f"  [robot] Workflow failed, replanning ({attempt + 1}/{MAX_WORKFLOW_REPLANS})...")
            debug_event(
                "workflow_replan_start",
                attempt=attempt + 1,
                failed_step=failed_step,
                reason=failure_reason,
                completed_steps=completed_steps,
            )
            current_workflow = replan_workflow(
                transcript,
                current_workflow,
                failed_step,
                details,
                completed_steps,
            )
        except Exception as e:
            print(f"  !! Workflow replan failed: {e}")
            debug_event("workflow_replan_error", attempt=attempt + 1, error=str(e))
            break

    apollo.say("Sorry, I couldn't finish that workflow")
    return False


def execute_llm_response(response_text, transcript):
    """Parse and execute the LLM's JSON response."""
    import apollo

    from apollo.planner import validate_workflow_output

    try:
        result = apollo.extract_json_object(response_text)
    except json.JSONDecodeError:
        print(f"  !! LLM returned invalid JSON: {response_text[:100]}")
        apollo.say("Sorry, I didn't understand that")
        return False

    action = result.get("action")

    if action == "command":
        return apollo.execute_registered_command(
            result.get("function", ""),
            transcript,
            args=result.get("args"),
            source="llm",
        )

    elif action == "workflow":
        try:
            workflow = validate_workflow_output(result)
        except Exception as exc:
            apollo.planner_failure("legacy_workflow", str(exc), transcript=transcript, payload=result)
            apollo.say("Sorry, I didn't understand that")
            return False
        return apollo.execute_workflow(workflow, transcript)

    elif action == "code":
        apollo.planner_failure("legacy_code", "rejected legacy code action", transcript=transcript, payload=result)
        print("  !! Planner returned forbidden action: code")
        apollo.say("Sorry, I couldn't plan that safely")
        return False

    else:
        print(f'  [robot] LLM could not interpret: "{transcript}"')
        apollo.say("Sorry, I didn't understand that. Say help for a list of commands.")
        return False
