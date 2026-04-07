"""
Command matching and routing functions for Apollo.

This module uses a *lazy import* pattern for cross-references back into the
``apollo`` package namespace.  Any symbol that tests mock-patch via
``mock.patch.object(apollo, "X")`` is accessed at **call time** through
``import apollo; apollo.X`` so that the mock is visible.  Only symbols that
are *never* patched are imported directly from their canonical submodule.
"""

import json
import os
import re
import threading
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Direct imports — only things that are NEVER mock-patched by tests
# ---------------------------------------------------------------------------
from apollo.config import (
    AI_TRACE_LOG,
    COMMAND_FAIL_SOUND,
    COMMAND_LOG,
    COMMAND_SUCCESS_SOUND,
    CONF_DIRECT_HIGH,
    CONF_DIRECT_MID,
    CONF_MIN_MATCH,
    LLM_DEFER_CONFIDENCE,
    MIN_COMMAND_CONFIDENCE,
    MULTI_STEP_HINT_WORDS,
    NEGATION_PHRASES,
    ROUTER_BYPASS_CONFIDENCE,
    UNKNOWN_MAX_WORDS,
    VISION_HINT_WORDS,
)
from apollo.logging_utils import (
    debug_event,
    log_transcript,
    redact_secret_like_text,
)
from apollo.types import Route
from apollo.utils import (
    build_match_candidates,
    extract_retry_delay_seconds,
    is_quota_error,
    starts_with_soft_request_prefix,
    strip_request_wrappers,
)

# ---------------------------------------------------------------------------
# Module-level lock shared across routing functions
# ---------------------------------------------------------------------------
_route_lock = threading.Lock()


# ===========================================================================
# COMMAND MATCHING
# ===========================================================================


def match_command(transcript, min_confidence=MIN_COMMAND_CONFIDENCE):
    """
    Find the best matching command for a transcript.
    Returns (command_dict, confidence, extra_text) or (None, 0, "")
    """
    import apollo

    best_match = None
    best_score = 0
    extra_text = ""

    for text in build_match_candidates(transcript):
        for cmd in apollo.COMMANDS:
            for phrase in cmd["phrases"]:
                # Exact match
                if text == phrase:
                    return cmd, 1.0, ""

                # Starts with the phrase (e.g. "type hello world" matches "type")
                if text.startswith(phrase):
                    score = min(1.0, len(phrase.split()) / max(len(text.split()), 1) + 0.5)
                    remainder = text[len(phrase):].strip()
                    if score > best_score:
                        best_score = score
                        best_match = cmd
                        extra_text = remainder

                # Phrase appears contiguously inside the request
                phrase_with_spaces = f" {phrase} "
                padded_text = f" {text} "
                if phrase_with_spaces in padded_text:
                    score = min(0.78, len(phrase.split()) / max(len(text.split()), 1) + 0.25)
                    if score > best_score:
                        best_score = score
                        best_match = cmd
                        extra_text = ""

                # All words of the phrase appear in the transcript
                phrase_words = set(phrase.split())
                text_words = set(text.split())
                if phrase_words.issubset(text_words):
                    score = len(phrase_words) / max(len(text_words), 1)
                    if score > best_score:
                        best_score = score
                        best_match = cmd
                        extra_text = ""

    if best_score >= min_confidence:
        return best_match, best_score, extra_text

    return None, 0, ""


# ===========================================================================
# DEFERRAL / BYPASS HELPERS
# ===========================================================================


def should_defer_match_to_llm(transcript, cmd, confidence, extra=""):
    """Decide whether to defer to the LLM planner or execute a local match directly.

    Fast-path: high-confidence single-action matches execute immediately without
    an LLM round-trip, saving 1-3 seconds of latency. Multi-step and ambiguous
    requests still go through the planner for smart handling.
    """
    import apollo

    if not apollo.LLM_FALLBACK_ENABLED:
        return False
    normalized = strip_request_wrappers(transcript)

    # --- Always fast-path these (no LLM needed) ---
    from apollo.commands import show_help, stop_listening, type_text

    if cmd and cmd["action"] in {show_help, stop_listening} and confidence >= 0.9 and not extra:
        return False
    if cmd and cmd["action"] == type_text and extra:
        return False
    # High-confidence single-action match: execute directly (biggest speed win)
    if cmd and confidence >= 0.85 and not apollo.looks_like_multi_step_request(transcript, extra):
        return False

    # --- Always defer these (LLM adds value) ---
    if apollo.looks_like_multi_step_request(transcript, extra):
        return True
    words = set(normalized.split())
    if words & VISION_HINT_WORDS:
        return True
    if any(phrase in normalized for phrase in NEGATION_PHRASES):
        return True
    if normalized.count("open") > 1 or normalized.count("click") > 1:
        return True

    # --- No local match at all: LLM is the only option ---
    if not cmd:
        return True

    # --- Low-confidence match: let LLM disambiguate ---
    if confidence < LLM_DEFER_CONFIDENCE:
        return True

    return False


# ===========================================================================
# EXECUTION HELPERS
# ===========================================================================


def execute_named_app_quit_request(app_name, transcript):
    """Run a normalized named-app quit request."""
    import apollo

    apollo.quit_named_app(app_name)
    print(f'  [robot] Quitting app: "{app_name}"')
    apollo.say(f"Closing {app_name}")
    log_command(transcript, f"quit app:{app_name}")
    return True


def execute_click_target_request(target, transcript):
    """Run a single-step vision workflow for a named on-screen click target."""
    import apollo

    if not apollo.GEMINI_API_KEY:
        print("  !! Vision click requested but Gemini is not configured")
        apollo.say("Screen-aware clicking needs Gemini vision enabled")
        return False
    workflow = apollo.build_click_target_workflow(target)
    return apollo.execute_workflow(workflow, transcript)


def emit_route_latency(route_start):
    """Log end-to-end route latency in milliseconds."""
    debug_event("route_latency_ms", ms=int((time.time() - route_start) * 1000))


def execute_matched_command(cmd, transcript, confidence, extra, route_start):
    """Execute a locally matched command and handle logging/state updates."""
    import apollo
    from apollo.commands import type_text
    from apollo.macos import play_sound, type_string

    print(f"  [check] Matched: \"{cmd['phrases'][0]}\" (confidence: {confidence:.0%})")

    if cmd["action"] == type_text and extra:
        print(f"  [keyboard] Typing: \"{extra}\"")
        type_string(extra)
        play_sound(COMMAND_SUCCESS_SOUND)
        apollo.update_command_state(transcript, "type_text", True)
        emit_route_latency(route_start)
        return True

    try:
        cmd["action"]()
        log_command(transcript, cmd["phrases"][0])
        play_sound(COMMAND_SUCCESS_SOUND)
        apollo.update_command_state(transcript, cmd["phrases"][0], True)
        emit_route_latency(route_start)
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"  !! Error: {e}")
        debug_event("command_error", transcript=transcript, matched=cmd["phrases"][0], error=str(e))
        apollo.say("Sorry, something went wrong")
        play_sound(COMMAND_FAIL_SOUND)
        apollo.update_command_state(transcript, cmd["phrases"][0], False)
        return False


# ===========================================================================
# ROUTING CLASSIFICATION
# ===========================================================================


def has_router_workflow_signal(transcript, extra="", cmd=None):
    """Conservatively detect requests that should not bypass the Stage-1 router."""
    normalized = strip_request_wrappers(transcript)
    trailing = strip_request_wrappers(extra)
    combined = " ".join(part for part in [normalized, trailing] if part).strip()
    if not combined:
        return False

    exact_command_phrase = bool(cmd and normalized in cmd["phrases"] and not trailing)
    words = combined.split()
    action_words = {
        "open", "launch", "start", "click", "press", "type", "write", "dictate",
        "save", "run", "scroll", "ask", "tell", "send", "quit", "close", "focus",
        "switch", "ensure", "select", "choose", "paste", "copy", "undo", "redo",
    }
    action_count = sum(1 for word in words if word in action_words)

    if any(phrase in normalized for phrase in NEGATION_PHRASES):
        return True
    if " and then " in f" {combined} " or re.search(r"\bthen\b", combined):
        return True
    if any(token in normalized for token in ("make sure", "if not already")) or normalized.startswith("ensure "):
        return True
    if re.match(r"^click(?:\s+on)?\s+.+$", normalized) and not exact_command_phrase:
        return True
    if re.search(r"\b(ask|tell|message|send)\s+claude\b", normalized):
        return True
    if "claude" in normalized and any(token in normalized for token in ("sonnet", "model", "new chat")):
        return True
    if re.match(r"^(?:close|quit|exit)\s+.+$", normalized) and not exact_command_phrase:
        return True
    if re.match(r"^(?:open|launch|start)\s+.+$", normalized) and not exact_command_phrase:
        return True
    if trailing and set(trailing.split()) & MULTI_STEP_HINT_WORDS:
        return True
    if normalized.count(" and ") >= 1 and action_count >= 2:
        return True
    if action_count >= 2 and len(words) >= 5:
        return True
    return False


def should_bypass_router(transcript, cmd, confidence, extra=""):
    """Return True when a high-confidence local match should skip Stage 1."""
    from apollo.commands import show_help, stop_listening, type_text

    if not cmd:
        return False
    if cmd["action"] == type_text and extra:
        return True
    if cmd["action"] in {show_help, stop_listening} and confidence >= ROUTER_BYPASS_CONFIDENCE:
        return True
    if confidence < ROUTER_BYPASS_CONFIDENCE:
        return False
    return not has_router_workflow_signal(transcript, extra, cmd)


def can_fallback_to_local_match(transcript, cmd, confidence, extra="", min_confidence=MIN_COMMAND_CONFIDENCE):
    """Return True when it is safe to fall back to the fuzzy matcher."""
    from apollo.commands import type_text

    if not cmd or confidence < min_confidence:
        return False
    if cmd["action"] == type_text and extra:
        return True
    return not has_router_workflow_signal(transcript, extra, cmd)


def classify_route(transcript, cmd, confidence, extra=""):
    """Deterministic pre-router for the new two-stage planner path."""
    import apollo
    from apollo.commands import show_help, stop_listening, type_text

    normalized = strip_request_wrappers(transcript)
    trailing = strip_request_wrappers(extra)
    word_count = len(normalized.split())
    has_soft_prefix = starts_with_soft_request_prefix(transcript)

    if word_count == 0:
        return Route.UNKNOWN

    if cmd and cmd["action"] == type_text and extra and not apollo.looks_like_multi_step_request(transcript, extra):
        return Route.DIRECT

    if cmd and cmd["action"] in {show_help, stop_listening} and confidence >= CONF_DIRECT_HIGH:
        return Route.DIRECT

    if any(phrase in normalized for phrase in NEGATION_PHRASES):
        return Route.ROUTER

    if apollo.looks_like_multi_step_request(transcript, extra):
        return Route.WORKFLOW

    if apollo.extract_click_target_request(transcript):
        return Route.WORKFLOW

    if apollo.extract_quit_app_request(transcript):
        return Route.WORKFLOW

    if cmd is None and apollo.is_simple_open_request(transcript):
        return Route.WORKFLOW

    if (
        cmd and confidence >= CONF_DIRECT_HIGH and not has_soft_prefix
        and not has_router_workflow_signal(transcript, extra, cmd)
    ):
        return Route.DIRECT

    exact_phrase = bool(cmd and normalized in cmd["phrases"] and not trailing)
    if cmd and confidence >= CONF_DIRECT_MID and exact_phrase and not has_soft_prefix:
        return Route.DIRECT

    if cmd and confidence >= CONF_MIN_MATCH:
        return Route.ROUTER

    if cmd is None and word_count <= UNKNOWN_MAX_WORDS:
        return Route.UNKNOWN

    return Route.ROUTER


def _build_deterministic_workflow_reason(transcript, cmd, extra):
    """Build a planner reason when Stage 1 routing is skipped."""
    import apollo

    click_target = apollo.extract_click_target_request(transcript)
    if click_target:
        return f"click target: {click_target}"

    quit_app = apollo.extract_quit_app_request(transcript)
    if quit_app:
        return f"quit app: {quit_app}"

    if cmd is None and apollo.is_simple_open_request(transcript):
        app_name = apollo.extract_open_app_request(transcript)
        return f"open app: {app_name}" if app_name else "generic app launch"

    return "multi-step request"


def announce_quota_issue(context, error):
    """Speak a concise quota error message with retry guidance when available."""
    import apollo

    retry_delay = extract_retry_delay_seconds(error)
    message = f"Gemini quota was exceeded while {context}"
    if retry_delay:
        message += f". Retry in about {int(round(retry_delay))} seconds"
    print(f"  !! {message}. See {AI_TRACE_LOG}")
    apollo.say(message)


# ===========================================================================
# MAIN ROUTING ENTRY POINTS
# ===========================================================================


def route_command_two_stage(transcript):
    """Run the strict two-stage router + planner flow."""
    import apollo
    from apollo.macos import play_sound
    from apollo.planner import planner_failure

    route_start = time.time()
    cmd, confidence, extra = apollo.match_command(transcript, min_confidence=CONF_MIN_MATCH)
    route = classify_route(transcript, cmd, confidence, extra)
    debug_event(
        "route_command_two_stage",
        transcript=transcript,
        confidence=round(confidence, 3),
        matched=cmd["phrases"][0] if cmd else None,
        extra=extra,
    )
    debug_event(
        "classify_route",
        route=route.value,
        confidence=round(confidence, 3),
        matched=cmd["phrases"][0] if cmd else None,
        extra=extra,
    )

    if route == Route.DIRECT:
        return apollo.execute_matched_command(cmd, transcript, confidence, extra, route_start)

    if route == Route.UNKNOWN:
        apollo.say("Sorry, I didn't understand that")
        play_sound(COMMAND_FAIL_SOUND)
        apollo.update_command_state(transcript, "unknown", False)
        return False

    if not apollo.LLM_FALLBACK_ENABLED:
        if cmd and confidence >= CONF_MIN_MATCH:
            return apollo.execute_matched_command(cmd, transcript, confidence, extra, route_start)
        print(f"  ? Didn't understand: \"{transcript}\"")
        log_transcript("unmatched_command", transcript)
        apollo.say("Sorry, I didn't understand that. Say help for a list of commands.")
        play_sound(COMMAND_FAIL_SOUND)
        apollo.update_command_state(transcript, "unmatched", False)
        return False

    if not _route_lock.acquire(blocking=False):
        print("  ... LLM already processing another command...")
        play_sound(COMMAND_FAIL_SOUND)
        return False

    try:
        if route == Route.WORKFLOW:
            workflow_reason = _build_deterministic_workflow_reason(transcript, cmd, extra)
        else:
            try:
                router_result = apollo.call_router(transcript)
                debug_event("router_response", transcript=transcript, result=router_result)
            except TimeoutError as exc:
                planner_failure("router", "router timed out", transcript=transcript, error=str(exc))
                debug_event("router_timeout", transcript=transcript, error=str(exc))
                router_result = None
            except Exception as exc:
                debug_event("router_error", transcript=transcript, error=str(exc))
                router_result = None

            if router_result is None:
                if can_fallback_to_local_match(transcript, cmd, confidence, extra, min_confidence=CONF_MIN_MATCH):
                    debug_event("router_local_fallback", transcript=transcript, confidence=round(confidence, 3))
                    return apollo.execute_matched_command(cmd, transcript, confidence, extra, route_start)
                apollo.say("Sorry, something went wrong")
                play_sound(COMMAND_FAIL_SOUND)
                apollo.update_command_state(transcript, "router_error", False)
                return False

            if router_result["action"] == "unknown":
                print(f"  [robot] Router returned unknown for: \"{transcript}\"")
                apollo.say("Sorry, I didn't understand that")
                play_sound(COMMAND_FAIL_SOUND)
                apollo.update_command_state(transcript, "unknown", False)
                emit_route_latency(route_start)
                return False

            if router_result["action"] == "command":
                handled = apollo.execute_registered_command(
                    router_result["function"],
                    transcript,
                    args=router_result.get("args"),
                    source="router",
                )
                if handled:
                    play_sound(COMMAND_SUCCESS_SOUND)
                    apollo.update_command_state(transcript, f'router:{router_result["function"]}', True)
                    emit_route_latency(route_start)
                    return True
                play_sound(COMMAND_FAIL_SOUND)
                apollo.update_command_state(transcript, f'router:{router_result["function"]}', False)
                return False

            workflow_reason = router_result["reason"]

        try:
            workflow = apollo.call_workflow_planner(transcript, workflow_reason)
            debug_event("workflow_planner_response", transcript=transcript, workflow=workflow)
        except TimeoutError as exc:
            planner_failure("workflow", "workflow planner timed out", transcript=transcript, error=str(exc))
            apollo.say("I understood but couldn't plan the steps")
            play_sound(COMMAND_FAIL_SOUND)
            apollo.update_command_state(transcript, "workflow_timeout", False)
            return False
        except Exception as exc:
            debug_event("workflow_planner_error", transcript=transcript, error=str(exc))
            if is_quota_error(exc):
                announce_quota_issue("planning the steps", exc)
            elif can_fallback_to_local_match(transcript, cmd, confidence, extra, min_confidence=CONF_MIN_MATCH):
                debug_event("workflow_local_fallback", transcript=transcript, confidence=round(confidence, 3))
                return apollo.execute_matched_command(cmd, transcript, confidence, extra, route_start)
            else:
                apollo.say("Sorry, something went wrong")
            play_sound(COMMAND_FAIL_SOUND)
            apollo.update_command_state(transcript, "workflow_error", False)
            return False

        handled = apollo.execute_workflow(workflow, transcript)
        if handled:
            play_sound(COMMAND_SUCCESS_SOUND)
            apollo.update_command_state(transcript, "workflow", True)
            emit_route_latency(route_start)
            return True
        play_sound(COMMAND_FAIL_SOUND)
        apollo.update_command_state(transcript, "workflow", False)
        return False
    finally:
        _route_lock.release()


def route_command_legacy(transcript):
    """Match a transcript to a command and execute it using the legacy planner."""
    import apollo
    from apollo.macos import play_sound

    route_start = time.time()
    cmd, confidence, extra = apollo.match_command(transcript)
    quit_app_name = apollo.extract_quit_app_request(transcript) if cmd is None else ""
    click_target = apollo.extract_click_target_request(transcript) if cmd is None else ""
    is_multi_step = apollo.looks_like_multi_step_request(transcript, extra)
    force_llm_first = should_defer_match_to_llm(transcript, cmd, confidence, extra)
    if force_llm_first and not quit_app_name and not click_target:
        debug_event("route_command_deferred_to_llm", transcript=transcript,
                    confidence=round(confidence, 3),
                    matched=cmd["phrases"][0] if cmd else None)
        cmd = None
        confidence = 0
        extra = ""
    debug_event("route_command", transcript=transcript,
                confidence=round(confidence, 3),
                matched=cmd["phrases"][0] if cmd else None)

    if cmd is None:
        if quit_app_name:
            try:
                handled = execute_named_app_quit_request(quit_app_name, transcript)
                if handled:
                    play_sound(COMMAND_SUCCESS_SOUND)
                    apollo.update_command_state(transcript, f"quit:{quit_app_name}", True, app=quit_app_name)
                    emit_route_latency(route_start)
                    return True
            except Exception as e:
                debug_event("quit_named_app_failed", transcript=transcript, app=quit_app_name, error=str(e))

        if click_target:
            handled = execute_click_target_request(click_target, transcript)
            if handled:
                play_sound(COMMAND_SUCCESS_SOUND)
                apollo.update_command_state(transcript, "vision:click_target", True)
                emit_route_latency(route_start)
                return True
            play_sound(COMMAND_FAIL_SOUND)
            apollo.update_command_state(transcript, "vision:click_target", False)
            return False

        llm_attempted = False
        if apollo.LLM_FALLBACK_ENABLED:
            if not _route_lock.acquire(blocking=False):
                print("  ... LLM already processing another command...")
                play_sound(COMMAND_FAIL_SOUND)
                return False
            try:
                print(f"  [robot] No exact match. Asking AI...")
                debug_event("llm_fallback_start", transcript=transcript)
                llm_response = apollo.llm_interpret_command(transcript, complex=is_multi_step)
                debug_event("llm_fallback_response", response=llm_response[:200])
                llm_attempted = True
                handled = apollo.execute_llm_response(llm_response, transcript)
                if handled:
                    play_sound(COMMAND_SUCCESS_SOUND)
                    apollo.update_command_state(transcript, "llm", True)
                    emit_route_latency(route_start)
                    return True
            except Exception as e:
                print(f"  !! LLM fallback error: {e}")
                debug_event("llm_fallback_error", error=str(e))
            finally:
                _route_lock.release()

        # Try generic "open X" app pattern only after the planner had its shot.
        app_name = apollo.extract_open_app_request(transcript)
        if app_name:
            try:
                apollo.mac_open_app(app_name)
                print(f"  [check] Opened app: \"{app_name}\"")
                apollo.say(f"Opening {app_name}")
                log_command(transcript, f"open app:{app_name}")
                play_sound(COMMAND_SUCCESS_SOUND)
                apollo.update_command_state(transcript, f"open:{app_name}", True, app=app_name)
                return True
            except Exception as e:
                debug_event("generic_app_open_failed", transcript=transcript,
                            app=app_name, error=str(e))

        if llm_attempted:
            play_sound(COMMAND_FAIL_SOUND)
            apollo.update_command_state(transcript, "llm", False)
            return False

        print(f"  ? Didn't understand: \"{transcript}\"")
        log_transcript("unmatched_command", transcript)
        apollo.say("Sorry, I didn't understand that. Say help for a list of commands.")
        play_sound(COMMAND_FAIL_SOUND)
        apollo.update_command_state(transcript, "unmatched", False)
        return False

    return apollo.execute_matched_command(cmd, transcript, confidence, extra, route_start)


def route_command(transcript):
    """Match a transcript to a command and execute it."""
    import apollo
    if apollo.APOLLO_2STAGE_PLANNER:
        return route_command_two_stage(transcript)
    return route_command_legacy(transcript)


# ===========================================================================
# LOGGING
# ===========================================================================


def log_command(transcript, matched_phrase):
    """Log commands to history file for debugging."""
    entry = {
        "time": datetime.now().isoformat(),
        "said": redact_secret_like_text(transcript),
        "matched": redact_secret_like_text(matched_phrase),
    }
    try:
        history = []
        if os.path.exists(COMMAND_LOG):
            with open(COMMAND_LOG, "r") as f:
                history = json.load(f)
        history.append(entry)
        history = history[-500:]
        with open(COMMAND_LOG, "w") as f:
            json.dump(history, f, indent=2)
        log_transcript("command", transcript, matched=matched_phrase)
    except Exception:
        pass
