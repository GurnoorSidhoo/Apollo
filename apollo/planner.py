"""
Planner, router, and validation functions for the Apollo voice assistant.

Many functions use a lazy ``import apollo`` pattern at call time so that
``mock.patch.object(apollo, "X")`` in tests intercepts the correct reference.
Functions that access the mutable ``COMMANDS`` list or ``_last_command`` dict
also use this pattern because those objects live in ``apollo.__init__``.
"""

import os
import re

from apollo.config import (
    GEMINI_MODEL,
    GEMINI_ROUTER_MODEL,
    GEMINI_PLANNER_MODEL,
    DEFAULT_GEMINI_EVAL_PLANNER_MODEL,
    ROUTER_MAX_OUTPUT_TOKENS,
    ROUTER_TIMEOUT_SECONDS,
    WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS,
    WORKFLOW_PLANNER_TIMEOUT_SECONDS,
    MAX_WORKFLOW_STEPS,
    MAX_WORKFLOW_WAIT_SECONDS,
    MAX_WORKFLOW_VISION_STEPS,
    NEGATION_PHRASES,
    GENERIC_APP_ALIASES,
    OPEN_REQUEST_BLOCKERS,
    CLICK_TARGET_KEYWORDS,
    MULTI_STEP_HINT_WORDS,
    WAIT_FOR_STATE_CONDITIONS,
    WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS,
    WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL,
    WAIT_FOR_STATE_MIN_POLL_INTERVAL,
    WAIT_FOR_STATE_MAX_POLL_INTERVAL,
)
from apollo.types import (
    PlannerValidationError,
    StructuredOutputError,
    ROUTER_RESPONSE_JSON_SCHEMA,
    ROUTER_OUTPUT_SCHEMA,
    WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
    WORKFLOW_PLANNER_OUTPUT_SCHEMA,
    is_valid_uri,
    validate_json_schema,
)
from apollo.logging_utils import debug_event, ai_trace_event
from apollo.utils import strip_request_wrappers, build_match_candidates
from apollo.gemini import model_candidates


def build_command_context():
    """Generate a context string listing all registered commands for the LLM."""
    import apollo
    lines = []
    for cmd in apollo.COMMANDS:
        func_name = cmd["action"].__name__
        phrases = ", ".join(cmd["phrases"][:3])
        desc = cmd["description"]
        lines.append(f"- {func_name}(): triggers=[{phrases}] -- {desc}")
    return "\n".join(lines)


def build_router_system_prompt():
    """Return the strict Stage-1 router prompt."""
    import apollo
    context_block = apollo.get_command_context()
    return f"""You are the command router for Biggie, a voice-controlled macOS assistant.
Your ONLY job is to classify the user's spoken request into exactly one action type.

## Available Commands
{build_command_context()}

## Previous Context
{context_block}

## Response Contract
Return ONLY the router object that matches the configured response schema.
The "reason" field is a short observability tag, not an explanation.
Only include "args" for type_text.

## Classification Rules (in priority order)

1. SINGLE REGISTERED COMMAND: If the request maps unambiguously to exactly one
   registered command, return action="command" with that function name. This
   includes requests wrapped in conversational filler ("hey can you please save",
   "um just paste it").

2. TYPE_TEXT SPECIAL CASE: If the request starts with "type", "write", or
   "dictate" followed by content, return action="command" with function="type_text"
   and args={{"text": "<the content after the trigger word>"}}. Do NOT include the
   trigger word in the text.

3. MULTI-STEP SIGNALS -- return action="workflow" when ANY of these are true:
   - Contains "and then", "then", or multiple action verbs
   - Names a specific UI element to click ("click on the settings button")
   - Involves interacting with a chat interface (typing a message and sending it)
   - Opens an app AND does something inside it
   - Uses "ensure", "make sure", "if not already"
   - Involves quitting/closing a specific named app ("close chrome", "quit spotify")
   - Involves opening an app with no dedicated registered command
   - Contains negation/correction ("not chrome, safari" / "don't close it, minimize it")

4. FOLLOW-UP CONTEXT: If the request is a follow-up ("now type hello", "do that
   again") and previous context resolves it to a single command, return
   action="command". If ambiguous or multi-step, return action="workflow".

5. UNKNOWN: If the request is conversational filler with no actionable intent,
   gibberish, or asks for something outside Biggie's capabilities, return
   action="unknown".

## Constraints
- NEVER invent function names. Only use names from Available Commands.
- The "args" field is ONLY used for type_text. Omit it for all other commands.
- The "reason" field is a SHORT observability tag (3-8 words), NOT chain-of-thought.
- When in doubt between command and workflow, prefer command if a single registered
  command clearly handles the full intent."""


def build_workflow_planner_system_prompt(utterance, router_reason):
    """Return the strict Stage-2 workflow planner prompt."""
    import apollo
    context_block = apollo.get_command_context()
    return f"""Only invoked when the router returns action="workflow".
You are the workflow planner for Biggie, a voice-controlled macOS assistant.
The router has determined this request requires a multi-step workflow. Produce a structured plan.

## User Request
"{utterance}"

## Router Classification
{router_reason}

## Previous Context
{context_block}

## Available Step Types

1. `open_app`: launch or bring an app to the foreground.
2. `focus_app`: bring an already-running app to the foreground.
3. `quit_app`: quit a named application.
4. `open_url`: open a URL in the default browser.
5. `command`: execute a registered Biggie command by function name.
6. `keypress`: simulate a keyboard shortcut.
7. `type_text`: type a string at the current cursor position.
8. `wait`: pause between steps for UI to settle.
9. `wait_for_state`: poll for a specific UI state before continuing.
10. `vision`: screenshot + AI vision to find/click a UI element.
11. `say`: speak text aloud.

## Available Registered Commands (for "command" steps)
{build_command_context()}

## Response Contract
Return ONLY the workflow object that matches the configured response schema.

## Planning Rules

1. DETERMINISTIC OVER VISION: Prefer open_app, keypress, type_text, command
   over vision. Use vision ONLY when no keyboard shortcut or deterministic path
   exists.

2. VISION TASK DESCRIPTIONS: Write precise, self-contained instructions. Include
   idempotency: "If X is already Y, do nothing."

3. MAX 12 STEPS. If more are needed, truncate and add a say step explaining.

4. IDEMPOTENT STEPS: For "ensure" requests, use "If already in state X, do
   nothing" in vision tasks.

5. CHAT-APP PATTERN ("ask Claude about X"):
   open_app -> wait_for_state(app_frontmost) -> vision(new chat) -> vision(ensure model) -> type_text -> keypress(return)

6. QUIT PATTERN ("close chrome"): Single quit_app with normalized macOS name.

7. CLICK PATTERN ("click on X"): Single vision step.

8. STATEFUL WAITING: Prefer wait_for_state after open_app and focus_app when
   you can name the condition (frontmost app, window exists, element exists,
   focused value contains text). Use plain wait only for short settle pauses.

9. APP NAME NORMALIZATION: "chrome" -> "Google Chrome", "vs code" -> "Visual
   Studio Code", "claude" -> "Claude", etc.

10. NO CODE GENERATION. Only use listed step types."""


def call_router(transcript, eval_mode=False):
    """Call the Stage-1 router and return a validated router payload."""
    import apollo
    try:
        return apollo.call_gemini_structured(
            system_instruction=build_router_system_prompt(),
            user_text=transcript,
            response_json_schema=ROUTER_RESPONSE_JSON_SCHEMA,
            validator=validate_router_output,
            preferred_models=eval_structured_model_candidates(
                GEMINI_ROUTER_MODEL,
                GEMINI_MODEL,
                "gemini-2.5-flash",
                call_type="router",
                eval_mode=eval_mode,
            ),
            call_type="router",
            max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS,
            timeout_seconds=ROUTER_TIMEOUT_SECONDS,
            trace_context={"transcript": transcript, "eval_mode": eval_mode},
        )
    except StructuredOutputError as exc:
        planner_failure(
            "router",
            exc.category,
            transcript=transcript,
            error=str(exc),
            model=exc.model,
            attempt=exc.attempt,
            correlation_id=exc.correlation_id,
            raw_response=exc.raw_response,
        )
        raise


def call_workflow_planner(transcript, router_reason, eval_mode=False):
    """Call the Stage-2 planner and return a validated workflow payload."""
    import apollo
    try:
        return apollo.call_gemini_structured(
            system_instruction=build_workflow_planner_system_prompt(transcript, router_reason),
            user_text="Return the workflow object.",
            response_json_schema=WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
            validator=validate_workflow_output,
            preferred_models=eval_structured_model_candidates(
                GEMINI_PLANNER_MODEL,
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                call_type="workflow_planner",
                eval_mode=eval_mode,
            ),
            call_type="workflow_planner",
            max_output_tokens=WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS,
            timeout_seconds=WORKFLOW_PLANNER_TIMEOUT_SECONDS,
            trace_context={"transcript": transcript, "router_reason": router_reason, "eval_mode": eval_mode},
        )
    except StructuredOutputError as exc:
        planner_failure(
            "workflow",
            exc.category,
            transcript=transcript,
            error=str(exc),
            model=exc.model,
            attempt=exc.attempt,
            correlation_id=exc.correlation_id,
            raw_response=exc.raw_response,
        )
        raise


def resolve_generic_app_name(requested_name):
    """Map a spoken app name to a concrete macOS app name when possible."""
    normalized = " ".join(strip_request_wrappers(requested_name).split())
    normalized = re.sub(r"^(?:the|my)\s+", "", normalized).strip()
    normalized = re.sub(r"\b(?:app|application)$", "", normalized).strip()
    if not normalized:
        return ""
    return GENERIC_APP_ALIASES.get(normalized, " ".join(word.capitalize() for word in normalized.split()))


def is_simple_open_request(transcript):
    """Only treat short single-intent 'open X' requests as generic app launches."""
    normalized = strip_request_wrappers(transcript)
    if not normalized or any(phrase in normalized for phrase in NEGATION_PHRASES):
        return False
    if " and " in normalized or " then " in normalized:
        return False
    match = re.match(r"^(open|launch|start)\s+(.+)$", normalized)
    if not match:
        return False
    target = re.sub(r"^(?:the|my)\s+", "", match.group(2)).strip()
    target = re.sub(r"\b(?:app|application)$", "", target).strip()
    if not target or len(target.split()) > 4:
        return False
    if set(target.split()) & OPEN_REQUEST_BLOCKERS:
        return False
    return True


def extract_open_app_request(transcript):
    """Extract a generic 'open X' style app request from natural language."""
    if not is_simple_open_request(transcript):
        return ""
    for candidate in build_match_candidates(transcript):
        match = re.match(r"^(open|launch|start)\s+(.+)$", candidate)
        if match:
            target = re.sub(r"^(?:the|my)\s+", "", match.group(2)).strip()
            target = re.sub(r"\b(?:app|application)$", "", target).strip()
            if target:
                return resolve_generic_app_name(target)
    return ""


def extract_quit_app_request(transcript):
    """Extract a simple 'close/quit X' app request and normalize the app name."""
    normalized = strip_request_wrappers(transcript)
    if not normalized or " and " in normalized or " then " in normalized:
        return ""

    for candidate in build_match_candidates(transcript):
        match = re.match(r"^(?:close|quit|exit)\s+(.+)$", candidate)
        if not match:
            continue
        target = re.sub(r"^(?:the|my)\s+", "", match.group(1)).strip()
        target = re.sub(r"\b(?:app|application)$", "", target).strip()
        if target in {"window", "this", "it", "that", "here"}:
            return ""
        if not target or len(target.split()) > 4:
            return ""
        return resolve_generic_app_name(target)
    return ""


def extract_click_target_request(transcript):
    """Extract a simple on-screen target from 'click on X' / 'click X'."""
    normalized = strip_request_wrappers(transcript)
    if not normalized or " and " in normalized or " then " in normalized:
        return ""

    for candidate in build_match_candidates(transcript):
        match = re.match(r"^click(?:\s+on)?\s+(.+)$", candidate)
        if not match:
            continue
        target = match.group(1).strip()
        if not target or target in {"it", "this", "that", "here", "there"}:
            return ""
        if target in CLICK_TARGET_KEYWORDS:
            return ""
        return target
    return ""


def build_click_target_workflow(target):
    """Build a single-step vision workflow for clicking a named UI element."""
    return {
        "action": "workflow",
        "description": f"Clicking {target}",
        "steps": [{
            "type": "vision",
            "task": f'Click on the on-screen UI element or content labeled "{target}"',
            "reason": "locate and click target",
        }],
    }


def looks_like_multi_step_request(transcript, extra=""):
    """Detect prompts that should go through the planner instead of a direct match."""
    normalized = strip_request_wrappers(transcript)
    trailing = strip_request_wrappers(extra)
    combined = " ".join(part for part in [normalized, trailing] if part).strip()
    if not combined:
        return False

    words = combined.split()
    actions = {"open", "launch", "start", "click", "press", "type", "write", "select", "choose",
               "switch", "ask", "tell", "send", "focus", "ensure"}
    action_count = sum(1 for word in words if word in actions)

    if any(phrase in normalized for phrase in NEGATION_PHRASES):
        return True
    if any(phrase in normalized for phrase in (
        "new chat", "chat section", "chat tab", "chat feature", "model picker",
        "switch model", "use sonnet", "ask claude", "tell claude", "send to claude",
    )):
        return True
    if normalized.startswith("ensure ") and "claude" in normalized and "sonnet" in normalized:
        return True
    if trailing and set(trailing.split()) & MULTI_STEP_HINT_WORDS:
        return True
    if " then " in normalized:
        return True
    if normalized.count(" and ") >= 1 and action_count >= 2:
        return True
    if action_count >= 2 and len(words) >= 5:
        return True
    return False


def build_workflow_capabilities_context():
    """Describe the structured workflow steps the planner is allowed to emit."""
    return """Allowed workflow step types:
- {"type": "open_app", "app": "Claude", "fallback_url": "https://claude.ai"}
- {"type": "focus_app", "app": "Claude"}
- {"type": "quit_app", "app": "Google Chrome"}
- {"type": "open_url", "url": "https://example.com"}
- {"type": "command", "function": "open_claude", "args": {"text": "optional only for type_text"}}
- {"type": "keypress", "key": "return", "command": false, "shift": false, "ctrl": false, "option": false}
- {"type": "type_text", "text": "what's the weather like today?"}
- {"type": "wait", "seconds": 1.0}
- {"type": "wait_for_state", "condition": "app_frontmost", "app": "Claude", "timeout_seconds": 2.0}
- {"type": "vision", "task": "Open a new chat in Claude if one is not already open"}
- {"type": "say", "text": "Working on it"}

Use "workflow" for multi-step UI tasks, chat-app requests, model selection, or anything that involves more than one action.
Use idempotent vision tasks for "ensure" style requests, for example: "If Claude is not using Sonnet, switch it to Sonnet. If it is already Sonnet, do nothing."
Prefer wait_for_state when the next step depends on app focus, a window appearing, or an element becoming available.
For prompts like "ask Claude Sonnet what's the weather like today", prefer:
open_app -> wait_for_state(app_frontmost) -> vision(new chat) -> vision(ensure Sonnet) -> type_text -> keypress(return)."""


def build_planner_system_prompt():
    """Return the planner instructions for the main LLM routing pass."""
    import apollo
    context_block = apollo.get_command_context()
    context_section = f"\n\nPrevious command context:\n{context_block}" if context_block else ""

    return f"""You are Biggie, a voice-controlled macOS assistant. Your job is to convert the user's request into a robust executable plan.

Available commands and their functions:
{build_command_context()}

{build_workflow_capabilities_context()}{context_section}

Respond with EXACTLY one JSON object in one of these formats:

1. Existing command:
{{"action": "command", "function": "function_name_here", "args": {{"text": "optional only for type_text"}}}}

2. Workflow plan:
{{"action": "workflow", "description": "Short present-tense summary", "steps": [
  {{"type": "open_app", "app": "Claude", "fallback_url": "https://claude.ai"}},
  {{"type": "wait_for_state", "condition": "app_frontmost", "app": "Claude", "timeout_seconds": 2.0}},
  {{"type": "vision", "task": "Open a new chat in Claude if one is not already open"}},
  {{"type": "vision", "task": "If Claude is not using Sonnet, switch it to Sonnet. If it is already Sonnet, do nothing."}},
  {{"type": "type_text", "text": "what's the weather like today?"}},
  {{"type": "keypress", "key": "return"}}
]}}

3. Unknown:
{{"action": "unknown"}}

Rules:
- Prefer "workflow" for almost every non-trivial request.
- Do not emit a top-level "vision" action.
- Do not emit "code".
- Prefer idempotent steps that can be safely retried.
- Prefer "wait_for_state" over blind "wait" when you can name the expected UI state.
- Map "scroll up" and "scroll down" to the existing scroll_up / scroll_down commands, not "unknown".
- Map "click on X" or "click X" to a workflow with a single vision step that clicks the named on-screen target.
- Map "close chrome", "quit chrome", "close spotify", and similar requests to a quit_app workflow step with the normalized macOS app name.
- For chat-app requests, always include steps to focus/open the app, ensure the right chat/model state, type the user prompt, and send it.
- Keep workflows under {MAX_WORKFLOW_STEPS} steps.
- Do not return markdown. Do not explain your reasoning.
- If the user's request seems like a follow-up to their previous command, use the context to resolve ambiguity.

Examples:
User: "scroll down"
{{"action": "command", "function": "scroll_down"}}

User: "click on the video your mama so stupid"
{{"action": "workflow", "description": "Clicking the requested on-screen target", "steps": [{{"type": "vision", "task": "Click on the on-screen UI element or content labeled \\"the video your mama so stupid\\""}}]}}

User: "close chrome"
{{"action": "workflow", "description": "Quitting Google Chrome", "steps": [{{"type": "quit_app", "app": "Google Chrome"}}]}}

Respond with ONLY the JSON."""


def resolve_eval_planner_model():
    """Return the eval planner model, falling back to production when unset."""
    raw_value = os.environ.get("APOLLO_EVAL_PLANNER_MODEL")
    if raw_value is None:
        return GEMINI_PLANNER_MODEL
    return raw_value.strip() or DEFAULT_GEMINI_EVAL_PLANNER_MODEL


def eval_structured_model_candidates(primary_model, *fallback_models, call_type="planner", eval_mode=False):
    """Return structured-call model candidates, swapping in the eval model when requested."""
    if not eval_mode:
        return model_candidates(primary_model, *fallback_models)

    selected_model = resolve_eval_planner_model()
    print(f"  [eval] Planner model: {selected_model}")
    debug_event("eval_planner_model_selected", model=selected_model, call_type=call_type)
    return model_candidates(selected_model, primary_model, *fallback_models)


def planner_model_candidates(complex=False, eval_mode=False):
    """Select planner candidates without affecting the production planner path."""
    if eval_mode:
        selected_model = resolve_eval_planner_model()
        print(f"  [eval] Planner model: {selected_model}")
        debug_event("eval_planner_model_selected", model=selected_model, complex=complex)
        return model_candidates(selected_model, GEMINI_PLANNER_MODEL, GEMINI_MODEL)
    if complex:
        return model_candidates(GEMINI_PLANNER_MODEL, GEMINI_MODEL, "gemini-2.5-flash")
    return model_candidates("gemini-2.5-flash", GEMINI_PLANNER_MODEL, GEMINI_MODEL)


def llm_interpret_command(transcript, complex=False, eval_mode=False):
    """Ask Gemini to interpret a voice command that didn't match any predefined phrase.

    When complex=True (multi-step workflows), use Pro for better planning quality.
    For simple disambiguation, use Flash for speed (~2-3x faster).
    """
    import apollo
    return apollo.gemini_generate_json(
        system_instruction=build_planner_system_prompt(),
        user_text=f'The user said: "{transcript}"',
        max_output_tokens=700,
        preferred_models=planner_model_candidates(complex=complex, eval_mode=eval_mode),
        trace_label="planner",
        trace_context={"transcript": transcript, "complex": complex, "eval_mode": eval_mode},
    )


def extract_text_argument(value):
    """Coerce a flexible JSON field into a plain text string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, str) else ""
    return ""


def registered_command_function_names():
    """Return all registered command function names."""
    import apollo
    return [cmd["action"].__name__ for cmd in apollo.COMMANDS]


def find_registered_command(func_name):
    """Return the registered command dict for a function name, if any."""
    import apollo
    for cmd in apollo.COMMANDS:
        if cmd["action"].__name__ == func_name:
            return cmd
    return None


def edit_distance(left, right):
    """Return the Levenshtein edit distance between two short strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def maybe_correct_function_name(func_name):
    """Auto-correct an obviously hallucinated function name when the match is unique."""
    if not isinstance(func_name, str):
        return "", ""
    normalized = func_name.strip()
    if not normalized:
        return "", ""
    if find_registered_command(normalized):
        return normalized, ""

    close_matches = []
    for candidate in registered_command_function_names():
        if edit_distance(normalized, candidate) <= 2:
            close_matches.append(candidate)

    if len(close_matches) == 1:
        corrected = close_matches[0]
        return corrected, f'corrected function "{normalized}" -> "{corrected}"'
    return normalized, ""


def planner_warning(stage, message, **fields):
    """Log a non-fatal planner warning to debug and AI trace logs."""
    debug_event("planner_validation_warning", stage=stage, message=message, **fields)
    ai_trace_event("planner_validation_warning", stage=stage, message=message, **fields)


def planner_failure(stage, message, **fields):
    """Log a planner validation or parsing failure to debug and AI trace logs."""
    debug_event("planner_validation_failure", stage=stage, message=message, **fields)
    ai_trace_event("planner_validation_failure", stage=stage, message=message, **fields)


def normalize_reason(value, fallback):
    """Return a short observability reason, filling or trimming when needed."""
    if isinstance(value, str):
        trimmed = value.strip()
        if len(trimmed) >= 3:
            if len(trimmed) > 60:
                return trimmed[:60].rstrip(), True
            return trimmed, False
    return fallback, True


def normalize_workflow_step_for_validation(step, index):
    """Apply safe normalizations to obviously near-valid planner steps."""
    if not isinstance(step, dict):
        raise PlannerValidationError(f"workflow: step {index} must be an object")

    normalized = dict(step)
    warnings = []
    step_type = normalized.get("type")
    if isinstance(step_type, str):
        lowered = step_type.strip().lower()
        if lowered != step_type:
            normalized["type"] = lowered
            warnings.append(f"step {index}: normalized type casing")

    normalized["reason"], reason_was_filled = normalize_reason(
        normalized.get("reason"),
        "step reason missing",
    )
    if reason_was_filled:
        warnings.append(f"step {index}: filled missing reason")

    if normalized.get("type") == "wait" and "seconds" in normalized and isinstance(normalized["seconds"], str):
        try:
            normalized["seconds"] = float(normalized["seconds"].strip())
            warnings.append(f"step {index}: coerced wait seconds to number")
        except ValueError as exc:
            raise PlannerValidationError(f"workflow: step {index} has invalid wait seconds") from exc

    if normalized.get("type") == "wait_for_state":
        condition = normalized.get("condition")
        if isinstance(condition, str):
            lowered = condition.strip().lower()
            if lowered != condition:
                normalized["condition"] = lowered
                warnings.append(f"step {index}: normalized wait_for_state condition casing")
        for field_name in ("timeout_seconds", "poll_interval"):
            if field_name in normalized and isinstance(normalized[field_name], str):
                try:
                    normalized[field_name] = float(normalized[field_name].strip())
                    warnings.append(f"step {index}: coerced {field_name} to number")
                except ValueError as exc:
                    raise PlannerValidationError(
                        f"workflow: step {index} has invalid {field_name}"
                    ) from exc

    if normalized.get("type") == "command":
        corrected, correction_warning = maybe_correct_function_name(normalized.get("function", ""))
        if correction_warning:
            normalized["function"] = corrected
            warnings.append(f"step {index}: {correction_warning}")

    return normalized, warnings


def validate_router_output(result):
    """Validate and normalize a Stage-1 router response."""
    if not isinstance(result, dict):
        raise PlannerValidationError("router: payload must be a JSON object")

    normalized = dict(result)
    warnings = []
    action = normalized.get("action")
    if isinstance(action, str):
        lowered = action.strip().lower()
        if lowered != action:
            normalized["action"] = lowered
            warnings.append("normalized router action casing")

    normalized["reason"], reason_was_filled = normalize_reason(
        normalized.get("reason"),
        "router reason missing",
    )
    if reason_was_filled:
        warnings.append("filled missing router reason")

    if normalized.get("action") == "command":
        function_name = normalized.get("function", "")
        if not isinstance(function_name, str) or not function_name.strip():
            warnings.append("empty router function treated as unknown")
            normalized = {"action": "unknown", "reason": normalized["reason"]}
        else:
            normalized["function"] = function_name.strip()
            corrected, correction_warning = maybe_correct_function_name(normalized["function"])
            if correction_warning:
                normalized["function"] = corrected
                warnings.append(correction_warning)
            if not find_registered_command(normalized["function"]):
                warnings.append(f'unknown router function "{normalized["function"]}" treated as unknown')
                normalized = {"action": "unknown", "reason": normalized["reason"]}

    validate_json_schema(normalized, ROUTER_OUTPUT_SCHEMA, "router")

    if normalized["action"] == "command":
        if normalized["function"] == "type_text":
            text = extract_text_argument(normalized.get("args", {})).strip()
            if not text:
                raise PlannerValidationError("router: type_text requires non-empty args.text")
            normalized["args"] = {"text": text}
        elif "args" in normalized:
            normalized.pop("args", None)
            warnings.append("dropped unexpected router args")

    for warning in warnings:
        planner_warning("router", warning, payload=normalized)
    return normalized


def validate_workflow_output(result):
    """Validate and normalize a Stage-2 workflow planner response."""
    import apollo

    if not isinstance(result, dict):
        raise PlannerValidationError("workflow: payload must be a JSON object")

    normalized = dict(result)
    warnings = []

    if normalized.get("action") == "workflow" and "description" in normalized and "steps" in normalized:
        normalized.pop("action", None)
        warnings.append("dropped legacy workflow action field")

    description = normalized.get("description", "")
    if not isinstance(description, str) or not description.strip():
        raise PlannerValidationError("workflow: description is required")
    normalized["description"] = description.strip()

    if len(normalized["description"]) > 200:
        normalized["description"] = normalized["description"][:200].rstrip()
        warnings.append("trimmed workflow description")

    steps = normalized.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlannerValidationError("workflow: steps must be a non-empty array")

    normalized_steps = []
    for index, step in enumerate(steps, start=1):
        step_payload, step_warnings = normalize_workflow_step_for_validation(step, index)
        normalized_steps.append(step_payload)
        warnings.extend(step_warnings)

    if len(normalized_steps) > MAX_WORKFLOW_STEPS:
        normalized_steps = normalized_steps[:MAX_WORKFLOW_STEPS - 1] + [{
            "type": "say",
            "text": "I planned only the first few steps.",
            "reason": "explain truncation limit",
        }]
        warnings.append(f"truncated workflow to {MAX_WORKFLOW_STEPS} steps")

    normalized["steps"] = normalized_steps
    validate_json_schema(normalized, WORKFLOW_PLANNER_OUTPUT_SCHEMA, "workflow")

    vision_count = 0
    active_app_context = bool(apollo._last_command.get("app"))
    seen_quit_apps = set()

    for index, step in enumerate(normalized["steps"], start=1):
        step_type = step["type"]

        if step_type == "open_app":
            app_name = step["app"].strip()
            if not app_name:
                raise PlannerValidationError(f"workflow: step {index} has empty app")
            step["app"] = app_name
            if "fallback_url" in step and not is_valid_uri(step["fallback_url"]):
                raise PlannerValidationError(f"workflow: step {index} has invalid fallback_url")
            active_app_context = True

        elif step_type == "focus_app":
            app_name = step["app"].strip()
            if not app_name:
                raise PlannerValidationError(f"workflow: step {index} has empty app")
            step["app"] = app_name
            if app_name in seen_quit_apps:
                warnings.append(f"step {index}: focus_app follows quit_app for {app_name}")
            active_app_context = True

        elif step_type == "quit_app":
            app_name = step["app"].strip()
            if not app_name:
                raise PlannerValidationError(f"workflow: step {index} has empty app")
            step["app"] = app_name
            seen_quit_apps.add(app_name)

        elif step_type == "open_url":
            url = step["url"].strip()
            if not url:
                raise PlannerValidationError(f"workflow: step {index} has empty url")
            step["url"] = url
            active_app_context = True

        elif step_type == "command":
            func_name = step["function"].strip()
            if not func_name:
                raise PlannerValidationError(f"workflow: step {index} has empty function")
            corrected, correction_warning = maybe_correct_function_name(func_name)
            if correction_warning:
                step["function"] = corrected
                warnings.append(f"step {index}: {correction_warning}")
            if not find_registered_command(step["function"]):
                raise PlannerValidationError(f'workflow: step {index} references unknown function "{step["function"]}"')
            if step["function"] == "type_text":
                text = extract_text_argument(step.get("args", {})).strip()
                if not text:
                    raise PlannerValidationError(f"workflow: step {index} type_text requires args.text")
                step["args"] = {"text": text}
            elif "args" in step:
                step.pop("args", None)
                warnings.append(f"step {index}: dropped unexpected command args")
            active_app_context = True

        elif step_type == "keypress":
            key = step["key"].strip()
            if not key:
                raise PlannerValidationError(f"workflow: step {index} has empty key")
            step["key"] = key
            for modifier in ("command", "shift", "ctrl", "option"):
                if modifier in step and not isinstance(step[modifier], bool):
                    raise PlannerValidationError(f"workflow: step {index} has invalid {modifier} modifier")
                step[modifier] = bool(step.get(modifier, False))
            active_app_context = True

        elif step_type == "type_text":
            text = step["text"].strip()
            if not text:
                raise PlannerValidationError(f"workflow: step {index} has empty text")
            step["text"] = text
            if not active_app_context:
                warnings.append(f"step {index}: type_text without prior app focus context")
            active_app_context = True

        elif step_type == "wait":
            seconds = float(step["seconds"])
            if seconds < 0:
                raise PlannerValidationError(f"workflow: step {index} has negative wait")
            if seconds == 0:
                step["seconds"] = 0.1
                warnings.append(f"step {index}: clamped zero wait to 0.1s")
            elif seconds > MAX_WORKFLOW_WAIT_SECONDS:
                step["seconds"] = MAX_WORKFLOW_WAIT_SECONDS
                warnings.append(f"step {index}: clamped wait to {MAX_WORKFLOW_WAIT_SECONDS:.1f}s")

        elif step_type == "wait_for_state":
            condition = step["condition"].strip().lower()
            if condition not in WAIT_FOR_STATE_CONDITIONS:
                raise PlannerValidationError(
                    f'workflow: step {index} has invalid wait_for_state condition "{condition}"'
                )
            step["condition"] = condition

            timeout_seconds = float(step.get("timeout_seconds", WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS))
            if timeout_seconds <= 0:
                raise PlannerValidationError(f"workflow: step {index} has non-positive timeout_seconds")
            if timeout_seconds > MAX_WORKFLOW_WAIT_SECONDS:
                step["timeout_seconds"] = MAX_WORKFLOW_WAIT_SECONDS
                warnings.append(
                    f"step {index}: clamped wait_for_state timeout to {MAX_WORKFLOW_WAIT_SECONDS:.1f}s"
                )
            else:
                step["timeout_seconds"] = timeout_seconds

            poll_interval = float(step.get("poll_interval", WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL))
            if poll_interval <= 0:
                raise PlannerValidationError(f"workflow: step {index} has non-positive poll_interval")
            clamped_poll_interval = min(
                WAIT_FOR_STATE_MAX_POLL_INTERVAL,
                max(WAIT_FOR_STATE_MIN_POLL_INTERVAL, poll_interval),
            )
            if clamped_poll_interval != poll_interval:
                warnings.append(f"step {index}: clamped wait_for_state poll_interval")
            step["poll_interval"] = clamped_poll_interval

            app_name = step.get("app", "").strip()
            if condition in {"app_frontmost", "window_exists", "element_exists", "element_value_contains"}:
                if not app_name:
                    raise PlannerValidationError(f"workflow: step {index} wait_for_state requires app")
                step["app"] = app_name
                active_app_context = True

            if condition == "element_exists":
                label = step.get("label", "").strip()
                if not label:
                    raise PlannerValidationError(f"workflow: step {index} element_exists requires label")
                step["label"] = label
                if "role" in step:
                    step["role"] = step["role"].strip()

            elif condition == "element_value_contains":
                substring = step.get("substring", "").strip()
                if not substring:
                    raise PlannerValidationError(
                        f"workflow: step {index} element_value_contains requires substring"
                    )
                step["substring"] = substring

        elif step_type == "vision":
            task = step["task"].strip()
            if not task:
                raise PlannerValidationError(f"workflow: step {index} has empty task")
            step["task"] = task
            vision_count += 1
            active_app_context = True

        elif step_type == "say":
            text = step["text"].strip()
            if not text:
                raise PlannerValidationError(f"workflow: step {index} has empty text")
            step["text"] = text

    if vision_count > MAX_WORKFLOW_VISION_STEPS:
        raise PlannerValidationError(f"workflow: too many vision steps ({vision_count})")
    if vision_count > 3:
        warnings.append(f"workflow uses {vision_count} vision steps")

    for warning in warnings:
        planner_warning("workflow", warning, payload=normalized)
    return normalized
