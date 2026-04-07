"""
Apollo vision layer — screenshot capture, screen-aware clicking,
AX-first targeting, and postcondition verification.

Depends on: apollo.config, apollo.types, apollo.logging_utils, apollo.utils,
apollo.macos, apollo.gemini.

NOTE: Functions that tests mock-patch (capture_screenshot, click_at,
capture_vision_frame, request_vision_action, resolve_ui_target) are
called through ``import apollo`` at call time so that
``mock.patch.object(apollo, "X")`` works correctly.
"""

import os
import subprocess
import tempfile
import time

from apollo.config import (
    AX_QUERY_MAX_DEPTH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_VISION_MODEL,
    SAVE_VISION_DEBUG,
    VISION_CLICK_RETRY_LIMIT,
    VISION_CLICK_SETTLE_SECONDS,
    VISION_MIN_CLICK_CONFIDENCE,
    WORKFLOW_PLANNER_TIMEOUT_SECONDS,
    VISION_VERIFICATION_TIMEOUT_SECONDS,
)
from apollo.types import (
    VISION_ACTION_RESPONSE_JSON_SCHEMA,
    VISION_POSTCONDITION_RESPONSE_JSON_SCHEMA,
)
from apollo.gemini import (
    call_gemini_structured,
    model_candidates,
    validate_postcondition_verification_output,
    validate_vision_action_output,
)
from apollo.logging_utils import debug_event, save_debug_screenshot
from apollo.macos import (
    ax_get_focused_element_value,
    ax_get_window_count,
    extract_target_label_from_text,
    get_app_window_bounds,
    get_main_screen_bounds,
    infer_target_app_name,
    parse_sips_dimensions,
    query_ax_element,
    should_use_accessibility,
)
from apollo.types import StructuredOutputError
from apollo.utils import (
    extract_first_quoted_text,
    extract_target_role_from_text,
    strip_request_wrappers,
)


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

def capture_screenshot(region=None):
    """
    Capture the screen or a logical-coordinate region and return
    (png_bytes, metadata).
    """
    path = os.path.join(tempfile.gettempdir(), "apollo_screenshot.png")
    command = ["screencapture", "-x", "-t", "png"]
    logical_bounds = region or get_main_screen_bounds()

    if region:
        command.extend([
            "-R",
            f"{int(region['x'])},{int(region['y'])},{int(region['width'])},{int(region['height'])}",
        ])

    command.append(path)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"screencapture failed: {result.stderr}")

    pixel_width, pixel_height = parse_sips_dimensions(path)
    with open(path, "rb") as f:
        png_bytes = f.read()

    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

    scale_x = pixel_width / max(float(logical_bounds["width"]), 1.0)
    scale_y = pixel_height / max(float(logical_bounds["height"]), 1.0)
    metadata = {
        "region_x": float(logical_bounds["x"]),
        "region_y": float(logical_bounds["y"]),
        "logical_width": float(logical_bounds["width"]),
        "logical_height": float(logical_bounds["height"]),
        "pixel_width": int(pixel_width),
        "pixel_height": int(pixel_height),
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return png_bytes, metadata


def image_to_global_coordinates(x, y, metadata):
    """Translate image pixel coordinates back into global logical screen coordinates."""
    global_x = metadata["region_x"] + (float(x) / max(metadata["scale_x"], 1e-6))
    global_y = metadata["region_y"] + (float(y) / max(metadata["scale_y"], 1e-6))
    return int(round(global_x)), int(round(global_y))


def click_at(x, y):
    """Click at screen coordinates (x, y) using JXA (JavaScript for Automation)."""
    jxa_script = f"""
ObjC.import("CoreGraphics");
var point = $.CGPointMake({x}, {y});
var down = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseDown, point, $.kCGMouseButtonLeft);
$.CGEventPost($.kCGHIDEventTap, down);
var up = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseUp, point, $.kCGMouseButtonLeft);
$.CGEventPost($.kCGHIDEventTap, up);
"""
    subprocess.run(["osascript", "-l", "JavaScript", "-e", jxa_script], check=True)


# ---------------------------------------------------------------------------
# Vision frame capture
# ---------------------------------------------------------------------------

def capture_vision_frame(target_app, debug_prefix="vision"):
    """Capture a fresh screenshot for a vision step, preferring the target app window."""
    import apollo
    screenshot_region = None
    if target_app:
        try:
            screenshot_region = get_app_window_bounds(target_app)
            debug_event("vision_target_window", app=target_app, region=screenshot_region)
        except Exception as exc:
            debug_event("vision_target_window_failed", app=target_app, error=str(exc))

    image_bytes, metadata = apollo.capture_screenshot(region=screenshot_region)
    screenshot_path = None
    if SAVE_VISION_DEBUG:
        try:
            screenshot_path = save_debug_screenshot(image_bytes, debug_prefix)
        except Exception as exc:
            debug_event("vision_debug_save_failed", error=str(exc))

    metadata = dict(metadata)
    metadata["coordinate_space"] = "image"
    metadata["screenshot_region"] = screenshot_region
    metadata["screenshot_path"] = screenshot_path
    return image_bytes, metadata


# ---------------------------------------------------------------------------
# Vision action request
# ---------------------------------------------------------------------------

def request_vision_action(task, transcript, target_app, image_bytes, metadata):
    """Ask Gemini to resolve a vision task against a fresh screenshot."""
    prompt = (
        f'The user said: "{transcript}". Task: {task}\n'
        f'You are looking at a screenshot that is {metadata["pixel_width"]}x{metadata["pixel_height"]} pixels.\n'
        f'This screenshot represents the logical screen region x={metadata["region_x"]}, y={metadata["region_y"]}, '
        f'width={metadata["logical_width"]}, height={metadata["logical_height"]}.\n'
        f'The target app is: {target_app or "unknown"}.\n'
        "Return click coordinates relative to this screenshot image in image pixel space, not global screen space.\n"
        "For click actions, include target_label when possible, a confidence from 0.0 to 1.0, a short rationale, "
        "and an expected_postcondition when a successful click should change the UI.\n"
        "If confidence would be below 0.6, return not_found instead of an unsafe click."
    )
    return call_gemini_structured(
        system_instruction="""You are Biggie, a voice-controlled macOS assistant helping with a screen task.

Return ONLY the vision action object that matches the configured response schema.
Coordinates must be relative to the screenshot image itself, not global screen space.
Prefer targets inside the referenced app/window, not Terminal or editor chrome.
If multiple matches exist, choose the one most consistent with the target app and the user's request.
Use `noop` when the task is already satisfied and `not_found` when the target cannot be identified.
Only return `click` when the target is specific enough to act on safely.""",
        user_text=prompt,
        response_json_schema=VISION_ACTION_RESPONSE_JSON_SCHEMA,
        validator=validate_vision_action_output,
        preferred_models=model_candidates(GEMINI_VISION_MODEL, "gemini-2.5-flash", GEMINI_MODEL),
        call_type="vision",
        max_output_tokens=500,
        timeout_seconds=WORKFLOW_PLANNER_TIMEOUT_SECONDS,
        image_bytes=image_bytes,
        trace_context={
            "transcript": transcript,
            "task": task,
            "target_app": target_app,
            "metadata": metadata,
            "screenshot_path": metadata.get("screenshot_path"),
        },
    )


def resolve_click_coordinates(action, metadata):
    """Resolve action coordinates explicitly based on their declared coordinate space."""
    coordinate_space = metadata.get("coordinate_space", "image")
    if coordinate_space == "global":
        return int(action["x"]), int(action["y"])
    if coordinate_space == "image":
        return image_to_global_coordinates(int(action["x"]), int(action["y"]), metadata)
    raise ValueError(f"Unsupported coordinate space: {coordinate_space}")


# ---------------------------------------------------------------------------
# Postcondition verification
# ---------------------------------------------------------------------------

def _ax_verify_postcondition(postcondition, app_name, original_metadata):
    """Try to verify a click postcondition using accessibility data alone."""
    if not postcondition or not app_name:
        return {"status": "impossible", "reason": "No AX-verifiable postcondition available", "method": "ax"}

    target_label = (
        extract_target_label_from_text(postcondition)
        or original_metadata.get("target_label", "")
        or original_metadata.get("label", "")
    )
    target_role = extract_target_role_from_text(postcondition) or original_metadata.get("target_role", "")
    if target_label:
        matches = query_ax_element(app_name, target_label=target_label, target_role=target_role, max_results=1)
        if matches is None:
            return {"status": "unavailable", "reason": "AX element query unavailable", "method": "ax"}
        if matches:
            return {
                "status": "verified",
                "reason": f'AX element "{target_label}" is present',
                "method": "ax",
            }
        return {
            "status": "unverified",
            "reason": f'AX element "{target_label}" not found',
            "method": "ax",
        }

    normalized = strip_request_wrappers(postcondition)
    if any(word in normalized for word in ("window", "dialog", "sheet", "panel", "popover", "menu")):
        before = original_metadata.get("window_count_before")
        after = ax_get_window_count(app_name)
        if after is None:
            return {"status": "unavailable", "reason": "AX window count unavailable", "method": "ax"}
        if isinstance(before, int):
            if after > before:
                return {"status": "verified", "reason": "AX window count increased", "method": "ax"}
            if any(word in normalized for word in ("close", "dismiss")) and after < before:
                return {"status": "verified", "reason": "AX window count decreased", "method": "ax"}
            return {"status": "unverified", "reason": "AX window count did not change", "method": "ax"}

    quoted = extract_first_quoted_text(postcondition)
    if quoted and any(word in normalized for word in ("contain", "contains", "show", "shows", "value", "text")):
        focused_value = ax_get_focused_element_value(app_name)
        if focused_value is None:
            return {"status": "unavailable", "reason": "AX focused value unavailable", "method": "ax"}
        if quoted.lower() in focused_value.lower():
            return {"status": "verified", "reason": "Focused AX value contains expected text", "method": "ax"}
        return {"status": "unverified", "reason": "Focused AX value missing expected text", "method": "ax"}

    return {"status": "impossible", "reason": "No AX verification heuristic matched", "method": "ax"}


def verify_postcondition(postcondition, app_name, original_metadata):
    """Verify a click postcondition, preferring AX and falling back to one Gemini check."""
    import apollo
    ax_result = _ax_verify_postcondition(postcondition, app_name, original_metadata)
    if ax_result["status"] in {"verified", "unverified"}:
        ax_result["used_gemini"] = False
        return ax_result

    allow_gemini = bool(original_metadata.get("allow_gemini_verification", True))
    if not postcondition:
        return {
            "status": "unavailable",
            "reason": "No postcondition was provided",
            "method": "none",
            "used_gemini": False,
        }
    if not allow_gemini:
        return {
            "status": "unverified",
            "reason": "Gemini verification budget exhausted",
            "method": "none",
            "used_gemini": False,
        }
    if not apollo.GEMINI_API_KEY:
        return {
            "status": "unavailable",
            "reason": "Gemini verification unavailable",
            "method": "none",
            "used_gemini": False,
        }

    try:
        image_bytes, metadata = apollo.capture_vision_frame(app_name, debug_prefix="vision_verify")
    except Exception as exc:
        debug_event("screenshot_error", error=str(exc), phase="vision_verify")
        return {
            "status": "unavailable",
            "reason": f"Verification screenshot unavailable: {exc}",
            "method": "gemini",
            "used_gemini": False,
        }

    verification_prompt = (
        f'The user originally asked: "{original_metadata.get("transcript", "")}".\n'
        f'The executed task was: {original_metadata.get("task", "")}\n'
        f'The click description was: {original_metadata.get("description", "")}\n'
        f'The target app is: {app_name or "unknown"}.\n'
        f'Expected postcondition: {postcondition}\n'
        f'Visible target label before the click, if known: {original_metadata.get("target_label", "")}\n'
        f'The screenshot is {metadata["pixel_width"]}x{metadata["pixel_height"]} pixels representing '
        f'x={metadata["region_x"]}, y={metadata["region_y"]}, width={metadata["logical_width"]}, '
        f'height={metadata["logical_height"]}.'
    )

    try:
        result = call_gemini_structured(
            system_instruction="""You are verifying whether a UI postcondition is satisfied after a click.

Return ONLY the verification object matching the configured response schema.
Mark satisfied=true only when the screenshot clearly shows the requested postcondition.""",
            user_text=verification_prompt,
            response_json_schema=VISION_POSTCONDITION_RESPONSE_JSON_SCHEMA,
            validator=validate_postcondition_verification_output,
            preferred_models=model_candidates(GEMINI_VISION_MODEL, "gemini-2.5-flash", GEMINI_MODEL),
            call_type="vision_verify",
            max_output_tokens=120,
            timeout_seconds=VISION_VERIFICATION_TIMEOUT_SECONDS,
            image_bytes=image_bytes,
            trace_context={
                "transcript": original_metadata.get("transcript", ""),
                "task": original_metadata.get("task", ""),
                "target_app": app_name,
                "postcondition": postcondition,
                "screenshot_path": metadata.get("screenshot_path"),
            },
        )
    except Exception as exc:
        debug_event("vision_verify_unavailable", error=str(exc), target_app=app_name, postcondition=postcondition)
        return {
            "status": "unavailable",
            "reason": f"Gemini verification unavailable: {exc}",
            "method": "gemini",
            "used_gemini": True,
        }

    return {
        "status": "verified" if result["satisfied"] else "unverified",
        "reason": result["reason"],
        "method": "gemini",
        "used_gemini": True,
    }


# ---------------------------------------------------------------------------
# UI target resolution (AX-first, then vision fallback)
# ---------------------------------------------------------------------------

def resolve_ui_target(app_name, target_description, transcript):
    """Resolve a UI target with AX first when appropriate, then fall back to screenshot vision."""
    import apollo
    target_app = apollo.resolve_generic_app_name(app_name) if app_name else ""
    target_label = extract_target_label_from_text(target_description) or extract_target_label_from_text(transcript)
    target_role = extract_target_role_from_text(target_description)
    original_metadata = {
        "target_app": target_app,
        "task": target_description,
        "transcript": transcript,
        "target_label": target_label,
        "target_role": target_role,
        "window_count_before": apollo.ax_get_window_count(target_app) if target_app else None,
        "focused_value_before": apollo.ax_get_focused_element_value(target_app) if target_app else None,
    }

    if apollo.should_use_accessibility(target_app, target_description) and (target_label or target_role):
        ax_matches = apollo.query_ax_element(
            target_app,
            target_label=target_label,
            target_role=target_role,
            max_depth=AX_QUERY_MAX_DEPTH,
            max_results=1,
        )
        if ax_matches:
            first_match = ax_matches[0]
            if "center_x" in first_match and "center_y" in first_match:
                action = {
                    "action": "click",
                    "x": int(first_match["center_x"]),
                    "y": int(first_match["center_y"]),
                    "description": f'Click {first_match.get("label") or target_label or "target"}',
                    "target_label": first_match.get("label") or target_label or "",
                    "confidence": 1.0,
                    "rationale": "Resolved via macOS accessibility",
                }
                debug_event(
                    "vision_ax_resolved",
                    target_app=target_app,
                    target_label=target_label,
                    target_role=target_role,
                    match=first_match,
                )
                return {
                    "source": "ax",
                    "target_app": target_app,
                    "action": action,
                    "metadata": {
                        **original_metadata,
                        "coordinate_space": "global",
                        "ax_match": first_match,
                    },
                }
        debug_event("vision_ax_miss", target_app=target_app, target_label=target_label, target_role=target_role)

    image_bytes, metadata = apollo.capture_vision_frame(target_app, debug_prefix="vision")
    metadata = {
        **original_metadata,
        **metadata,
    }
    result = apollo.request_vision_action(target_description, transcript, target_app, image_bytes, metadata)
    debug_event("vision_response", response=result, metadata=metadata, target_app=target_app)
    return {
        "source": "vision",
        "target_app": target_app,
        "action": result,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Vision step execution
# ---------------------------------------------------------------------------

def execute_vision_steps_action(result, metadata, transcript):
    """Execute a legacy multi-step vision action while enforcing click confidence gating."""
    import apollo
    for index, step in enumerate(result.get("steps", []), start=1):
        if step["type"] == "click":
            confidence = step.get("confidence")
            if confidence is not None and float(confidence) < VISION_MIN_CLICK_CONFIDENCE:
                debug_event(
                    "vision_click_low_confidence",
                    confidence=float(confidence),
                    threshold=VISION_MIN_CLICK_CONFIDENCE,
                    step_index=index,
                    action=step,
                )
                apollo.say("I couldn't safely identify where to click")
                return False
            x, y = resolve_click_coordinates(step, metadata)
            print(f"  [eye] Vision step: click at ({x}, {y})")
            apollo.say(step.get("description", "Clicking"))
            apollo.click_at(x, y)
        elif step["type"] == "wait":
            time.sleep(step.get("seconds", 1))
    apollo.log_command(transcript, "vision:steps")
    return True


def execute_vision_task(task, transcript):
    """Resolve and execute a screen task with AX-first targeting and post-click verification."""
    import apollo
    apollo.say("Let me look at the screen")

    target_app = infer_target_app_name(f"{task} {transcript}")
    gemini_verification_used = False

    for attempt in range(1, VISION_CLICK_RETRY_LIMIT + 1):
        try:
            resolution = apollo.resolve_ui_target(target_app, task, transcript)
        except StructuredOutputError as exc:
            apollo.planner_failure(
                "vision",
                exc.category,
                transcript=transcript,
                task=task,
                error=str(exc),
                model=exc.model,
                attempt=exc.attempt,
                correlation_id=exc.correlation_id,
                raw_response=exc.raw_response,
            )
            print(f"  !! Vision returned invalid structured output: {exc}")
            apollo.say("I couldn't figure out where to click")
            return False
        except Exception as exc:
            error_text = str(exc)
            if any(token in error_text.lower() for token in ("screencapture", "capture", "screen")):
                print(f"  !! Screenshot failed: {exc}")
                debug_event("screenshot_error", error=error_text)
                apollo.say("I couldn't capture the screen")
            else:
                print(f"  !! Vision step failed: {exc}")
                debug_event("vision_error", task=task, transcript=transcript, error=error_text)
                apollo.say("I couldn't figure out where to click")
            return False

        result = resolution["action"]
        metadata = resolution["metadata"]
        target_app = resolution.get("target_app", target_app)
        action = result.get("action")

        if action == "click":
            confidence = result.get("confidence")
            if confidence is not None and float(confidence) < VISION_MIN_CLICK_CONFIDENCE:
                debug_event(
                    "vision_click_low_confidence",
                    confidence=float(confidence),
                    threshold=VISION_MIN_CLICK_CONFIDENCE,
                    action=result,
                    target_app=target_app,
                )
                apollo.say("I couldn't safely identify where to click")
                return False

            x, y = resolve_click_coordinates(result, metadata)
            desc = result.get("description", f"Clicking at ({x}, {y})")
            print(f"  [eye] Vision: {desc}")
            apollo.say(desc)
            apollo.click_at(x, y)
            time.sleep(VISION_CLICK_SETTLE_SECONDS)

            verification = apollo.verify_postcondition(
                result.get("expected_postcondition", ""),
                target_app,
                {
                    **metadata,
                    "task": task,
                    "transcript": transcript,
                    "description": desc,
                    "target_label": result.get("target_label") or metadata.get("target_label", ""),
                    "allow_gemini_verification": not gemini_verification_used,
                },
            )
            gemini_verification_used = gemini_verification_used or verification.get("used_gemini", False)

            if verification["status"] in {"verified", "unavailable"}:
                debug_event(
                    "vision_click_verified",
                    target_app=target_app,
                    attempt=attempt,
                    verification_status=verification["status"],
                    reason=verification["reason"],
                )
                apollo.log_command(transcript, f"vision:click:{x},{y}")
                return True

            if attempt < VISION_CLICK_RETRY_LIMIT:
                debug_event(
                    "vision_click_retry",
                    target_app=target_app,
                    attempt=attempt,
                    max_attempts=VISION_CLICK_RETRY_LIMIT,
                    reason=verification["reason"],
                )
                continue

            debug_event(
                "vision_click_unverified",
                target_app=target_app,
                attempt=attempt,
                reason=verification["reason"],
            )
            apollo.say("I couldn't confirm the click worked")
            return False

        if action == "steps":
            return apollo.execute_vision_steps_action(result, metadata, transcript)

        if action == "not_found":
            desc = result.get("description", "Could not find the target")
            print(f"  [eye] Vision: {desc}")
            apollo.say(desc)
            return False

        if action == "noop":
            desc = result.get("description", "Nothing else to do")
            print(f"  [eye] Vision: {desc}")
            return True

    return False
