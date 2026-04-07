"""
Apollo logging utilities — debug events, transcript logs, AI traces.

Depends only on apollo.config.
"""

import json
import os
import re
from datetime import datetime

from apollo.config import (
    AI_TRACE_LOG,
    DEBUG_LOG,
    SECRET_REPLACEMENT,
    TRANSCRIPT_LOG,
    VERBOSE_AI_TRACE,
    VISION_DEBUG_DIR,
)


def append_jsonl(path, payload):
    """Append one JSON object per line for lightweight debugging."""
    try:
        with open(path, "a") as f:
            json.dump(payload, f)
            f.write("\n")
    except Exception:
        pass


def debug_event(event, **fields):
    """Write a timestamped debug event to the Biggie debug log."""
    append_jsonl(DEBUG_LOG, {
        "time": datetime.now().isoformat(),
        "event": event,
        **sanitize_for_logging(fields),
    })


def log_transcript(stage, text, **fields):
    """Persist intermediate transcripts for debugging weak recognition."""
    append_jsonl(TRANSCRIPT_LOG, {
        "time": datetime.now().isoformat(),
        "stage": stage,
        "text": redact_secret_like_text(text),
        **sanitize_for_logging(fields),
    })


def redact_secret_like_text(text):
    """Best-effort redaction for secret-like transcripts before logging them."""
    if not isinstance(text, str) or not text:
        return text

    redacted = text
    patterns = (
        (r"(?i)\b(export\s+[a-z0-9_]*(?:api|token|secret|password)[a-z0-9_]*\s*(?:=|\s+))([^\s].*)", r"\1" + SECRET_REPLACEMENT),
        (r"(?i)\b((?:api key|token|secret|password)\s+)([a-z0-9_\-]{8,}(?:\s+[a-z0-9_\-]{4,})*)", r"\1" + SECRET_REPLACEMENT),
        (r"\bsk[-_a-z0-9]{12,}\b", SECRET_REPLACEMENT),
        (r"\bAIza[0-9A-Za-z\-_]{12,}\b", SECRET_REPLACEMENT),
        (r"\b[a-f0-9]{32,}\b", SECRET_REPLACEMENT),
    )
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def sanitize_for_logging(value):
    """Recursively redact secret-like strings before writing debug payloads."""
    if isinstance(value, str):
        return redact_secret_like_text(value)
    if isinstance(value, dict):
        return {key: sanitize_for_logging(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_logging(item) for item in value]
    return value


def ai_trace_event(kind, **fields):
    """Record detailed LLM/screenshot inputs and outputs for debugging."""
    payload = {
        "time": datetime.now().isoformat(),
        "kind": kind,
        **sanitize_for_logging(fields),
    }
    append_jsonl(AI_TRACE_LOG, payload)
    if VERBOSE_AI_TRACE:
        print(f"  [ai-trace] {kind}: {json.dumps(payload, ensure_ascii=True)[:1200]}")


def save_debug_screenshot(png_bytes, prefix):
    """Persist a screenshot used for vision debugging and return its path."""
    os.makedirs(VISION_DEBUG_DIR, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = os.path.join(VISION_DEBUG_DIR, filename)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path
