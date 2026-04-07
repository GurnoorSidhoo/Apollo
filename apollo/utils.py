"""
Apollo text-processing and miscellaneous utility functions.

Depends on: apollo.config, apollo.logging_utils (no circular deps).
"""

import difflib
import importlib
import queue
import re
import threading

from apollo.config import (
    AX_ROLE_KEYWORDS,
    COMMON_TRANSCRIPT_REPLACEMENTS,
    REQUEST_PREFIX_PATTERNS,
    REQUEST_SUFFIX_PATTERNS,
    WAKE_WORD_ALIASES,
)


def is_quota_error(error):
    """Return True when an error message indicates provider quota exhaustion."""
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message or "Gemini HTTP 429" in message


def extract_retry_delay_seconds(error):
    """Best-effort parse of retry delay hints from provider errors."""
    message = str(error)
    patterns = (
        r"retryDelay\"\s*:\s*\"(\d+)s\"",
        r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def has_local_whisper():
    """Return True when local Whisper packages appear to be installed."""
    try:
        required_modules = ("whisper", "torch", "numpy", "sounddevice")
        return all(importlib.util.find_spec(module_name) is not None for module_name in required_modules)
    except Exception:
        return False


def has_lip_reading_deps():
    """Return True if opencv-python and MediaPipe Face Mesh are importable."""
    try:
        return (
            importlib.util.find_spec("cv2") is not None
            and has_mediapipe_face_mesh()
        )
    except Exception:
        return False


def has_mediapipe_face_mesh():
    """Return True when the installed mediapipe package exposes Face Mesh."""
    try:
        return (
            importlib.util.find_spec("mediapipe.solutions.face_mesh") is not None
            or importlib.util.find_spec("mediapipe.python.solutions.face_mesh") is not None
        )
    except Exception:
        return False


def describe_websocket_error(error):
    """Return a more useful websocket error string when the server rejects the handshake."""
    parts = [str(error)]

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        reason = getattr(response, "reason_phrase", None)
        body = getattr(response, "body", b"")

        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", errors="replace")

        body = (body or "").strip()
        if status_code is not None and reason:
            parts.append(f"status={status_code} {reason}")
        if body:
            parts.append(f"body={body[:500]}")

    api_status = getattr(error, "status_code", None)
    api_body = getattr(error, "body", None)
    if api_status is not None and api_body:
        parts.append(f"api_status={api_status}")
        parts.append(f"api_body={str(api_body)[:500]}")

    seen = set()
    deduped_parts = []
    for part in parts:
        if part and part not in seen:
            deduped_parts.append(part)
            seen.add(part)
    return " | ".join(deduped_parts)


def normalize_text(text):
    """Lowercase and collapse punctuation so matching is more forgiving."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def canonicalize_text(text):
    """Normalize common STT mistakes before wake-word and command matching."""
    cleaned = normalize_text(text)
    for pattern, replacement in COMMON_TRANSCRIPT_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return " ".join(cleaned.split())


def strip_request_wrappers(text):
    """Remove polite filler around otherwise imperative commands."""
    stripped = canonicalize_text(text)
    changed = True
    while changed and stripped:
        changed = False
        for pattern in REQUEST_PREFIX_PATTERNS:
            updated = re.sub(pattern, "", stripped).strip()
            if updated != stripped:
                stripped = updated
                changed = True
        for pattern in REQUEST_SUFFIX_PATTERNS:
            updated = re.sub(pattern, "", stripped).strip()
            if updated != stripped:
                stripped = updated
                changed = True
    return stripped


def starts_with_soft_request_prefix(text):
    """Return True for modal question wrappers that should stay conservative."""
    normalized = canonicalize_text(text)
    return bool(re.match(r"^(?:can|could|would|will)\s+you\s+", normalized))


def collapse_repeated_phrase(text):
    """Collapse exact repeated phrase halves such as 'save save'."""
    words = text.split()
    if len(words) >= 2 and len(words) % 2 == 0:
        half = len(words) // 2
        if words[:half] == words[half:]:
            return " ".join(words[:half])
    return text


def build_match_candidates(text):
    """Generate a small set of normalized variants for fuzzy command matching."""
    seen = set()
    candidates = []

    def add(candidate):
        candidate = collapse_repeated_phrase(" ".join(candidate.split()))
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    base = canonicalize_text(text)
    add(base)

    trimmed = strip_request_wrappers(base)
    add(trimmed)

    detected, trailing = detect_wake_word(base)
    if detected:
        add(trailing)
        add(strip_request_wrappers(trailing))

    detected_trimmed, trailing_trimmed = detect_wake_word(trimmed)
    if detected_trimmed:
        add(trailing_trimmed)
        add(strip_request_wrappers(trailing_trimmed))

    return candidates or [base]


def detect_wake_word(text):
    """Return (detected, trailing_text) for a wake word at utterance start."""
    normalized = canonicalize_text(text)
    if not normalized:
        return False, ""

    words = normalized.split()
    alias_variants = sorted(
        (" ".join(alias.split()), tuple(alias.split())) for alias in WAKE_WORD_ALIASES
    )
    for _, alias_words in sorted(alias_variants, key=lambda item: len(item[1]), reverse=True):
        if words[:len(alias_words)] == list(alias_words):
            trailing = " ".join(words[len(alias_words):]).strip()
            return True, trailing

    first_word = words[0]
    if any(
        len(alias.split()) == 1 and difflib.SequenceMatcher(None, first_word, alias).ratio() >= 0.88
        for alias in WAKE_WORD_ALIASES
    ):
        trailing = " ".join(words[1:]).strip()
        return True, trailing

    return False, ""


def extract_first_quoted_text(text):
    """Return the first quoted span in a task description, if present."""
    if not isinstance(text, str):
        return ""
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        match = re.search(pattern, text)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return ""


def extract_target_role_from_text(text):
    """Infer a likely accessibility role hint from the task wording."""
    normalized = strip_request_wrappers(text or "")
    if not normalized:
        return ""
    for keyword, role in AX_ROLE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return role
    return ""


def run_with_timeout(timeout_seconds, func, *args, **kwargs):
    """Run a function in a helper thread and raise TimeoutError if it stalls."""
    result_queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_queue.put(("ok", func(*args, **kwargs)))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"timed out after {timeout_seconds:.0f}s") from exc

    if status == "error":
        raise payload
    return payload
