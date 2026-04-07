"""
Apollo Gemini / LLM client — transport detection, structured generation,
retry logic, and fallback model support.

Depends on: apollo.config, apollo.types, apollo.logging_utils, apollo.utils.

NOTE: Functions that tests mock-patch (GEMINI_API_KEY, run_with_timeout,
gemini_generate_structured_candidate, GEMINI_MODEL, etc.) access those
names through ``import apollo`` at call time so that
``mock.patch.object(apollo, "X")`` works correctly.
"""

import base64
import json
import re
import time
import urllib.error
import urllib.request
import uuid

from apollo.logging_utils import ai_trace_event, debug_event
from apollo.types import (
    PlannerValidationError,
    StructuredOutputError,
    VISION_ACTION_RESPONSE_JSON_SCHEMA,
    VISION_POSTCONDITION_RESPONSE_JSON_SCHEMA,
    validate_json_schema,
)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_gemini_client = None
_gemini_transport = None


# ---------------------------------------------------------------------------
# Transport detection and client caching
# ---------------------------------------------------------------------------

def get_gemini_transport():
    """Prefer the SDK when installed, otherwise use Gemini's REST API directly."""
    import apollo
    global _gemini_transport
    if _gemini_transport is not None:
        return _gemini_transport
    if not apollo.GEMINI_API_KEY:
        _gemini_transport = None
        return None
    try:
        from google import genai  # noqa: F401
        from google.genai import types  # noqa: F401
        _gemini_transport = "sdk"
    except Exception:
        _gemini_transport = "rest"
    return _gemini_transport


def get_gemini_client():
    """Return a cached Gemini SDK client to avoid per-call connection overhead."""
    import apollo
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    from google import genai
    _gemini_client = genai.Client(api_key=apollo.GEMINI_API_KEY)
    return _gemini_client


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_object(text):
    """Parse a JSON object, tolerating fenced output from LLMs."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def extract_gemini_text(payload):
    """Extract the text content from a Gemini API response payload."""
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        text = text.strip()
        if text:
            return text
    feedback = payload.get("promptFeedback")
    if feedback:
        raise RuntimeError(f"Gemini prompt rejected: {feedback}")
    raise RuntimeError("Gemini returned no text")


def model_candidates(*model_names):
    """Return unique non-empty model names in priority order."""
    seen = set()
    candidates = []
    for name in model_names:
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    return candidates


def looks_like_truncated_json(text):
    """Return True when a response looks like an incomplete JSON value."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] not in "}]" and stripped.startswith(("{", "[")):
        return True

    stack = []
    in_string = False
    escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if not stack or stack.pop() != "{":
                return False
        elif ch == "]":
            if not stack or stack.pop() != "[":
                return False
    return in_string or bool(stack)


def parse_structured_json_response(text):
    """Parse a structured JSON response without broad best-effort repair."""
    stripped = (text or "").strip()
    if not stripped:
        raise StructuredOutputError(
            "empty_response",
            "Gemini returned an empty structured response",
            call_type="unknown",
            raw_response=text or "",
            parse_result="empty_response",
        )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        category = "truncated_output" if looks_like_truncated_json(stripped) else "malformed_json"
        raise StructuredOutputError(
            category,
            f"Gemini returned {category.replace('_', ' ')}",
            call_type="unknown",
            raw_response=text,
            parse_result=category,
        ) from exc


# ---------------------------------------------------------------------------
# Vision output validation
# ---------------------------------------------------------------------------

def _normalize_optional_vision_fields(payload):
    """Normalize optional metadata added to vision click payloads."""
    normalized = {}
    target_label = payload.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        normalized["target_label"] = target_label.strip()
    if "confidence" in payload and payload.get("confidence") is not None:
        normalized["confidence"] = float(payload["confidence"])
    rationale = payload.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        normalized["rationale"] = rationale.strip()
    expected_postcondition = payload.get("expected_postcondition")
    if isinstance(expected_postcondition, str) and expected_postcondition.strip():
        normalized["expected_postcondition"] = expected_postcondition.strip()
    return normalized


def validate_postcondition_verification_output(result):
    """Validate the optional post-click verification response."""
    if not isinstance(result, dict):
        raise PlannerValidationError("vision_verify: payload must be a JSON object")
    validate_json_schema(result, VISION_POSTCONDITION_RESPONSE_JSON_SCHEMA, "vision_verify")
    return {
        "satisfied": bool(result["satisfied"]),
        "reason": result["reason"].strip(),
    }


def validate_vision_action_output(result):
    """Validate and normalize a structured vision action payload."""
    if not isinstance(result, dict):
        raise PlannerValidationError("vision: payload must be a JSON object")

    validate_json_schema(result, VISION_ACTION_RESPONSE_JSON_SCHEMA, "vision")
    action = result["action"]

    if action == "click":
        if "x" not in result or "y" not in result:
            raise PlannerValidationError("vision: click requires x and y")
        normalized = {
            "action": "click",
            "x": int(result["x"]),
            "y": int(result["y"]),
            "description": result["description"].strip(),
        }
        normalized.update(_normalize_optional_vision_fields(result))
        return normalized

    if action == "steps":
        steps = result.get("steps")
        if not isinstance(steps, list) or not steps:
            raise PlannerValidationError("vision: steps action requires a non-empty steps array")
        normalized_steps = []
        for index, step in enumerate(steps, start=1):
            step_type = step.get("type")
            if step_type == "click":
                if "x" not in step or "y" not in step:
                    raise PlannerValidationError(f"vision: step {index} click requires x and y")
                normalized_step = {
                    "type": "click",
                    "x": int(step["x"]),
                    "y": int(step["y"]),
                    "description": step["description"].strip(),
                }
                normalized_step.update(_normalize_optional_vision_fields(step))
                normalized_steps.append(normalized_step)
            elif step_type == "wait":
                if "seconds" not in step:
                    raise PlannerValidationError(f"vision: step {index} wait requires seconds")
                normalized_steps.append({
                    "type": "wait",
                    "seconds": float(step["seconds"]),
                    "description": step["description"].strip(),
                })
            else:
                raise PlannerValidationError(f'vision: step {index} has invalid type "{step_type}"')
        normalized = {
            "action": "steps",
            "steps": normalized_steps,
            "description": result["description"].strip(),
        }
        normalized.update(_normalize_optional_vision_fields(result))
        return normalized

    if action in {"not_found", "noop"}:
        normalized = {
            "action": action,
            "description": result["description"].strip(),
        }
        normalized.update(_normalize_optional_vision_fields(result))
        return normalized

    raise PlannerValidationError(f'vision: invalid action "{action}"')


# ---------------------------------------------------------------------------
# Error normalization
# ---------------------------------------------------------------------------

def normalize_structured_error(error, *, call_type, model="", attempt=0, correlation_id="", fallback_used=False, raw_response=""):
    """Coerce parsing, timeout, provider, and validation issues into structured categories."""
    if isinstance(error, StructuredOutputError):
        error.call_type = call_type
        error.model = error.model or model
        error.attempt = error.attempt or attempt
        error.correlation_id = error.correlation_id or correlation_id
        error.fallback_used = error.fallback_used or fallback_used
        if raw_response and not error.raw_response:
            error.raw_response = raw_response
        return error
    if isinstance(error, TimeoutError):
        return StructuredOutputError(
            "timeout",
            f"{call_type} timed out",
            call_type=call_type,
            model=model,
            attempt=attempt,
            correlation_id=correlation_id,
            fallback_used=fallback_used,
        )
    if isinstance(error, PlannerValidationError):
        return StructuredOutputError(
            "schema_mismatch",
            str(error),
            call_type=call_type,
            model=model,
            attempt=attempt,
            correlation_id=correlation_id,
            raw_response=raw_response,
            parse_result="ok",
            validation_result="schema_mismatch",
            fallback_used=fallback_used,
        )
    message = str(error)
    category = "safety_block" if "prompt rejected" in message.lower() or "safety" in message.lower() else "model_error"
    return StructuredOutputError(
        category,
        message,
        call_type=call_type,
        model=model,
        attempt=attempt,
        correlation_id=correlation_id,
        raw_response=raw_response,
        fallback_used=fallback_used,
    )


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def gemini_generate_structured_candidate(
    *,
    system_instruction,
    user_text,
    model_name,
    response_json_schema,
    image_bytes=None,
    max_output_tokens=300,
):
    """Generate one schema-constrained Gemini response for a single model."""
    import apollo
    if not apollo.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    transport = get_gemini_transport()
    if transport == "sdk":
        from google.genai import types

        parts = []
        if image_bytes is not None:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        parts.append(user_text)

        client = get_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
            config={
                "system_instruction": system_instruction,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": response_json_schema,
            },
        )
        return (response.text or "").strip(), transport

    parts = []
    if image_bytes is not None:
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
            }
        })
    parts.append({"text": user_text})

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_json_schema,
        },
    }

    request = urllib.request.Request(
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apollo.GEMINI_API_KEY}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
        response_text = extract_gemini_text(json.loads(body))
        return response_text.strip(), transport
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc


def call_gemini_structured(
    *,
    system_instruction,
    user_text,
    response_json_schema,
    validator,
    preferred_models,
    call_type,
    max_output_tokens,
    timeout_seconds,
    image_bytes=None,
    trace_context=None,
):
    """Call Gemini with schema-constrained output, validation, retries, and fallback."""
    import apollo
    if not apollo.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    correlation_id = uuid.uuid4().hex[:12]
    models = preferred_models or [apollo.GEMINI_MODEL]
    if not models:
        raise RuntimeError("No Gemini models configured")

    primary_model = models[0]
    fallback_model = models[1] if len(models) > 1 else ""
    attempts = [
        {"model": primary_model, "repair": False, "fallback_used": False},
    ]
    if primary_model:
        attempts.append({"model": primary_model, "repair": True, "fallback_used": False})
    if fallback_model:
        attempts.append({"model": fallback_model, "repair": False, "fallback_used": True})

    ai_trace_event(
        "gemini_structured_request",
        call_type=call_type,
        correlation_id=correlation_id,
        candidate_models=models,
        response_schema_used=True,
        trace_context=trace_context or {},
    )

    final_error = None
    for attempt_index, attempt_spec in enumerate(attempts, start=1):
        model_name = attempt_spec["model"]
        fallback_used = attempt_spec["fallback_used"]
        repair_attempt = attempt_spec["repair"]
        prompt_text = user_text
        if repair_attempt:
            if not isinstance(final_error, StructuredOutputError) or final_error.category not in {
                "empty_response",
                "malformed_json",
                "schema_mismatch",
                "truncated_output",
            }:
                continue
            prompt_text = f"{user_text}\n\n{apollo.GEMINI_STRUCTURED_REPAIR_INSTRUCTION}"

        start_time = time.time()
        parse_result = "not_attempted"
        validation_result = "not_attempted"
        disposition = "failed"
        raw_response = ""

        debug_event(
            "gemini_structured_attempt",
            call_type=call_type,
            model=model_name,
            latency_ms=0,
            attempt=attempt_index,
            repair_attempt=repair_attempt,
            response_schema_used=True,
            correlation_id=correlation_id,
            fallback_used=fallback_used,
        )
        try:
            raw_response, transport = apollo.run_with_timeout(
                timeout_seconds,
                apollo.gemini_generate_structured_candidate,
                system_instruction=system_instruction,
                user_text=prompt_text,
                model_name=model_name,
                response_json_schema=response_json_schema,
                image_bytes=image_bytes,
                max_output_tokens=max_output_tokens,
            )
            parsed = parse_structured_json_response(raw_response)
            parse_result = "ok"
            result = validator(parsed)
            validation_result = "ok"
            disposition = "succeeded"
            latency_ms = int((time.time() - start_time) * 1000)
            debug_event(
                "gemini_structured_result",
                call_type=call_type,
                model=model_name,
                latency_ms=latency_ms,
                attempt=attempt_index,
                response_schema_used=True,
                parse_result=parse_result,
                validation_result=validation_result,
                fallback_used=fallback_used,
                disposition=disposition,
                correlation_id=correlation_id,
            )
            ai_trace_event(
                "gemini_structured_result",
                call_type=call_type,
                model=model_name,
                transport=transport,
                attempt=attempt_index,
                response_schema_used=True,
                parse_result=parse_result,
                validation_result=validation_result,
                fallback_used=fallback_used,
                disposition=disposition,
                correlation_id=correlation_id,
            )
            if model_name != apollo.GEMINI_MODEL:
                debug_event("gemini_model_used", transport=transport, model=model_name)
            return result
        except Exception as exc:
            structured_error = normalize_structured_error(
                exc,
                call_type=call_type,
                model=model_name,
                attempt=attempt_index,
                correlation_id=correlation_id,
                fallback_used=fallback_used,
                raw_response=raw_response,
            )
            parse_result = structured_error.parse_result if structured_error.parse_result != "not_attempted" else parse_result
            validation_result = structured_error.validation_result if structured_error.validation_result != "not_attempted" else validation_result
            final_error = structured_error
            latency_ms = int((time.time() - start_time) * 1000)
            debug_event(
                "gemini_structured_result",
                call_type=call_type,
                model=model_name,
                latency_ms=latency_ms,
                attempt=attempt_index,
                response_schema_used=True,
                parse_result=parse_result,
                validation_result=validation_result,
                fallback_used=fallback_used,
                disposition=structured_error.category,
                correlation_id=correlation_id,
                error=str(structured_error),
            )
            ai_trace_event(
                "gemini_structured_result",
                call_type=call_type,
                model=model_name,
                attempt=attempt_index,
                response_schema_used=True,
                parse_result=parse_result,
                validation_result=validation_result,
                fallback_used=fallback_used,
                disposition=structured_error.category,
                correlation_id=correlation_id,
                error=str(structured_error),
                raw_response=raw_response,
            )
            if fallback_used or not fallback_model and not repair_attempt:
                continue
            if structured_error.category in {"timeout", "model_error", "safety_block"}:
                continue

    if final_error is None:
        final_error = StructuredOutputError(
            "model_error",
            f"{call_type} failed before any Gemini attempt completed",
            call_type=call_type,
            correlation_id=correlation_id,
        )
    raise final_error


def gemini_generate_json(system_instruction, user_text, image_bytes=None, max_output_tokens=300, preferred_models=None, trace_label="gemini", trace_context=None):
    """Generate a JSON response from Gemini using either the SDK or REST."""
    import apollo
    if not apollo.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    candidate_models = preferred_models or [apollo.GEMINI_MODEL]
    transport = get_gemini_transport()
    last_error = None
    ai_trace_event(
        "gemini_request",
        trace_label=trace_label,
        transport=transport,
        candidate_models=candidate_models,
        system_instruction=system_instruction,
        user_text=user_text,
        has_image=bool(image_bytes),
        trace_context=trace_context or {},
    )

    for model_name in candidate_models:
        ai_trace_event("gemini_attempt", trace_label=trace_label, model=model_name, transport=transport)
        if transport == "sdk":
            try:
                from google import genai
                from google.genai import types

                parts = []
                if image_bytes is not None:
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
                parts.append(user_text)

                client = get_gemini_client()
                response = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
                if model_name != apollo.GEMINI_MODEL:
                    debug_event("gemini_model_used", transport=transport, model=model_name)
                ai_trace_event("gemini_response", trace_label=trace_label, model=model_name, transport=transport, response=response.text)
                return response.text
            except Exception as exc:
                last_error = exc
                debug_event("gemini_sdk_failed", model=model_name, error=str(exc))
                ai_trace_event("gemini_error", trace_label=trace_label, model=model_name, transport=transport, error=str(exc))

        parts = []
        if image_bytes is not None:
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                }
            })
        parts.append({"text": user_text})

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        request = urllib.request.Request(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apollo.GEMINI_API_KEY}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
            if model_name != apollo.GEMINI_MODEL:
                debug_event("gemini_model_used", transport="rest", model=model_name)
            response_text = extract_gemini_text(json.loads(body))
            ai_trace_event("gemini_response", trace_label=trace_label, model=model_name, transport="rest", response=response_text)
            return response_text
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini HTTP {exc.code}: {details}")
            debug_event("gemini_rest_failed", model=model_name, error=str(last_error))
            ai_trace_event("gemini_error", trace_label=trace_label, model=model_name, transport="rest", error=str(last_error))
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Gemini request failed: {exc.reason}")
            debug_event("gemini_rest_failed", model=model_name, error=str(last_error))
            ai_trace_event("gemini_error", trace_label=trace_label, model=model_name, transport="rest", error=str(last_error))

    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed before any model could be tried")
