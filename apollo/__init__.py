"""
Apollo package — voice-controlled macOS assistant.

This file is the compatibility shim: it re-exports every public symbol from
the extracted submodules so that ``import apollo; apollo.X`` continues to
work and ``mock.patch.object(apollo, "X")`` targets the correct namespace.
"""

import subprocess  # noqa: F401 — tests reference apollo.subprocess
import threading   # noqa: F401 — tests reference apollo.threading
import time

# ------------------------------------------------------------------
# Extracted submodules (canonical source of truth for these symbols)
# ------------------------------------------------------------------
from apollo.config import *                          # noqa: F401,F403
from apollo.types import *                           # noqa: F401,F403
from apollo.logging_utils import *                   # noqa: F401,F403

COMMANDS = []


# Persistent state across commands — enables follow-up context
_last_command = {
    "transcript": "",
    "action": "",
    "app": "",
    "success": False,
    "timestamp": 0.0,
}


def update_command_state(transcript, action, success, app=""):
    """Update the last-command state for follow-up context."""
    _last_command["transcript"] = transcript
    _last_command["action"] = action
    _last_command["app"] = app or infer_target_app_name(transcript)
    _last_command["success"] = success
    _last_command["timestamp"] = time.time()


def get_command_context():
    """Return a context string describing the last command for the planner."""
    if not _last_command["transcript"] or (time.time() - _last_command["timestamp"]) > 120:
        return ""
    status = "succeeded" if _last_command["success"] else "failed"
    app_info = f" (app: {_last_command['app']})" if _last_command["app"] else ""
    return (
        f'The user\'s previous command was: "{_last_command["transcript"]}" '
        f'which {status}{app_info}. '
        f"Use this context to interpret follow-up requests like "
        f'"now type hello", "do that again", "click the other one", etc.'
    )


def command(phrases, description=""):
    """Decorator to register a voice command."""
    def decorator(func):
        COMMANDS.append({
            "phrases": [p.lower() for p in phrases],
            "action": func,
            "description": description,
        })
        return func
    return decorator


# ---------------------------------------------------------------------------
# Utility functions — imported from apollo.utils
# ---------------------------------------------------------------------------
from apollo.utils import (                           # noqa: F401,F403
    is_quota_error,
    extract_retry_delay_seconds,
    has_local_whisper,
    normalize_text,
    canonicalize_text,
    strip_request_wrappers,
    starts_with_soft_request_prefix,
    collapse_repeated_phrase,
    build_match_candidates,
    detect_wake_word,
    extract_first_quoted_text,
    extract_target_role_from_text,
)

from apollo.utils import describe_websocket_error    # noqa: F401
from apollo.utils import has_lip_reading_deps         # noqa: F401
from apollo.utils import has_mediapipe_face_mesh      # noqa: F401



# ===========================================================================
# COMMANDS — built-in commands are loaded from apollo.commands at the
# bottom of this file (after all helper functions are defined).
# ===========================================================================

# ===========================================================================
# macOS CONTROL FUNCTIONS — imported from apollo.macos
# ===========================================================================
from apollo.macos import (                           # noqa: F401,F403
    play_sound,
    say,
    mac_open,
    mac_open_app,
    applescript,
    run_applescript,
    focus_app,
    quit_named_app,
    spotify_command,
    press_key,
    hotkey,
    type_string,
    get_input_device,
    load_custom_commands,
    parse_sips_dimensions,
    get_main_screen_bounds,
    get_app_window_bounds,
    infer_target_app_name,
    _run_ax_query_json,
    extract_target_label_from_text,
    query_ax_element,
    ax_check_app_frontmost,
    ax_get_window_count,
    ax_get_focused_element_value,
    wait_for_state,
    condition_app_frontmost,
    condition_window_exists,
    condition_element_exists,
    condition_element_value_contains,
    build_wait_for_state_condition,
    should_use_accessibility,
    set_clipboard_and_paste,
    ensure_text_input_focused,
)

# ===========================================================================
# GEMINI / LLM CLIENT — imported from apollo.gemini
# ===========================================================================
from apollo.gemini import (                             # noqa: F401,F403
    get_gemini_transport,
    get_gemini_client,
    extract_json_object,
    extract_gemini_text,
    model_candidates,
    looks_like_truncated_json,
    parse_structured_json_response,
    _normalize_optional_vision_fields,
    validate_postcondition_verification_output,
    validate_vision_action_output,
    normalize_structured_error,
    gemini_generate_structured_candidate,
    call_gemini_structured,
    gemini_generate_json,
)

# ===========================================================================
# VISION — imported from apollo.vision
# ===========================================================================
from apollo.vision import (                             # noqa: F401,F403
    capture_screenshot,
    image_to_global_coordinates,
    click_at,
    capture_vision_frame,
    request_vision_action,
    resolve_click_coordinates,
    _ax_verify_postcondition,
    verify_postcondition,
    resolve_ui_target,
    execute_vision_steps_action,
    execute_vision_task,
)


# ===========================================================================
# PLANNER — imported from apollo.planner
# ===========================================================================
from apollo.planner import (                            # noqa: F401,F403
    build_command_context,
    build_router_system_prompt,
    build_workflow_planner_system_prompt,
    call_router,
    call_workflow_planner,
    resolve_generic_app_name,
    is_simple_open_request,
    extract_open_app_request,
    extract_quit_app_request,
    extract_click_target_request,
    build_click_target_workflow,
    looks_like_multi_step_request,
    build_workflow_capabilities_context,
    build_planner_system_prompt,
    resolve_eval_planner_model,
    eval_structured_model_candidates,
    planner_model_candidates,
    llm_interpret_command,
    extract_text_argument,
    registered_command_function_names,
    find_registered_command,
    edit_distance,
    maybe_correct_function_name,
    planner_warning,
    planner_failure,
    normalize_reason,
    normalize_workflow_step_for_validation,
    validate_router_output,
    validate_workflow_output,
)

# run_with_timeout — imported from apollo.utils
from apollo.utils import run_with_timeout              # noqa: F401



# ===========================================================================
# WORKFLOW EXECUTION — imported from apollo.workflow
# ===========================================================================
from apollo.workflow import (                           # noqa: F401,F403
    execute_registered_command,
    summarize_completed_steps,
    build_replan_user_prompt,
    replan_workflow,
    execute_workflow_once,
    execute_workflow,
    execute_llm_response,
)

# ===========================================================================
# COMMAND MATCHING & ROUTING — imported from apollo.routing
# ===========================================================================
from apollo.routing import (                            # noqa: F401,F403
    match_command,
    should_defer_match_to_llm,
    execute_named_app_quit_request,
    execute_click_target_request,
    emit_route_latency,
    execute_matched_command,
    has_router_workflow_signal,
    should_bypass_router,
    can_fallback_to_local_match,
    classify_route,
    _build_deterministic_workflow_reason,
    announce_quota_issue,
    route_command_two_stage,
    route_command_legacy,
    route_command,
    log_command,
)


# ===========================================================================
# AUDIO LISTENER — imported from apollo.audio
# ===========================================================================
from apollo.audio import (                              # noqa: F401,F403
    AudioListener,
    DeepgramUnavailableError,
    WhisperAudioListener,
)

# ===========================================================================
# MAIN — imported from apollo.main
# ===========================================================================
from apollo.main import (                               # noqa: F401,F403
    print_banner,
    check_dependencies,
    run_text_mode,
    main,
)


# ===========================================================================
# LIP READING / LIP SYNC — optional features
# ===========================================================================
from apollo.config import LIP_READING_ENABLED, LIP_SYNC_ENABLED  # noqa: F401

if LIP_READING_ENABLED:
    try:
        from apollo.lip_reading import LipReader       # noqa: F401
    except ImportError:
        pass

if LIP_SYNC_ENABLED:
    try:
        from apollo.lip_sync import (                  # noqa: F401
            LipSyncAnimator,
            TerminalLipSyncAnimator,
            get_lip_sync_animator,
        )
    except ImportError:
        pass


# ------------------------------------------------------------------
# Deferred import: register built-in commands now that all helpers
# (say, hotkey, mac_open_app, etc.) are defined.
# ------------------------------------------------------------------
from apollo.commands import *                        # noqa: F401,F403


if __name__ == "__main__":
    main()
