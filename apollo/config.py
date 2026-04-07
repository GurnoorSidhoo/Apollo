"""
Apollo configuration — env vars, constants, model selection.

Every other apollo.* module imports from here. This module must have
zero intra-package dependencies.
"""

import os
from typing import Set


# ---------------------------------------------------------------------------
# Env-var helpers (used in this module only)
# ---------------------------------------------------------------------------

def _read_env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to the supplied default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _read_env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to the supplied default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------
WAKE_WORD = "biggie"
SAMPLE_RATE = 16000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMAND_LOG = os.path.join(BASE_DIR, "apollo_history.json")
DEBUG_LOG = os.path.join(BASE_DIR, "apollo_debug.jsonl")
TRANSCRIPT_LOG = os.path.join(BASE_DIR, "apollo_transcripts.jsonl")
AI_TRACE_LOG = os.path.join(BASE_DIR, "apollo_ai_trace.jsonl")
VISION_DEBUG_DIR = os.path.join(BASE_DIR, "apollo_vision_debug")
DEBUG_AUDIO = os.environ.get("APOLLO_DEBUG_AUDIO", "").lower() in {"1", "true", "yes"}
VERBOSE_AI_TRACE = os.environ.get("APOLLO_VERBOSE_AI", "").lower() in {"1", "true", "yes"}
SAVE_VISION_DEBUG = os.environ.get("APOLLO_SAVE_VISION_DEBUG", "").lower() in {"1", "true", "yes"}
VOICE_FEEDBACK = False
AUDIO_FEEDBACK = True
WAKE_SOUND = "/System/Library/Sounds/Tink.aiff"
COMMAND_SUCCESS_SOUND = "/System/Library/Sounds/Glass.aiff"
COMMAND_FAIL_SOUND = "/System/Library/Sounds/Basso.aiff"
COMMAND_TIMEOUT = 8.0
WAKE_WORD_ALIASES: Set[str] = {"biggie", "biggy", "bigie", "big e", "piggy", "biddy", "picky"}

# ---------------------------------------------------------------------------
# Deepgram / Whisper
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = "".join(ch for ch in os.environ.get("DEEPGRAM_API_KEY", "") if ch.isprintable()).strip()
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_KEYTERM = os.environ.get("APOLLO_DEEPGRAM_KEYTERM", "Biggie").strip() or "Biggie"
DEEPGRAM_UTTERANCE_END_MS = "1500"
DEEPGRAM_ENDPOINTING_MS = 300
DEEPGRAM_PREFER_RICH_VARIANTS = os.environ.get("APOLLO_DEEPGRAM_PREFER_RICH", "").lower() in {"1", "true", "yes"}
DEEPGRAM_RECONNECT_DELAY = 2.0
DEEPGRAM_MAX_INIT_FAILURES = 2
WHISPER_MODEL = os.environ.get("APOLLO_WHISPER_MODEL", "tiny").strip() or "tiny"
WHISPER_CHUNK_SECONDS = float(os.environ.get("APOLLO_WHISPER_CHUNK_SECONDS", "2.5"))
WHISPER_SILENCE_THRESHOLD = float(os.environ.get("APOLLO_WHISPER_SILENCE_THRESHOLD", "0.008"))

# ---------------------------------------------------------------------------
# Lip Reading / Lip Sync (optional — requires opencv-python, mediapipe)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Push-to-talk (default: enabled; set APOLLO_PTT=0 to disable)
# ---------------------------------------------------------------------------
PTT_ENABLED = os.environ.get("APOLLO_PTT", "1").lower() not in {"0", "false", "no"}
PTT_BUTTON = os.environ.get("APOLLO_PTT_BUTTON", "f9").strip().lower()

LIP_READING_ENABLED = os.environ.get("APOLLO_LIP_READING", "").lower() in {"1", "true", "yes"}
LIP_SYNC_ENABLED = os.environ.get("APOLLO_LIP_SYNC", "").lower() in {"1", "true", "yes"}
WEBCAM_DEVICE_INDEX = _read_env_int("APOLLO_WEBCAM_DEVICE", 0)
LIP_READING_FPS = _read_env_int("APOLLO_LIP_READING_FPS", 15)
LIP_OPEN_THRESHOLD = _read_env_float("APOLLO_LIP_OPEN_THRESHOLD", 0.25)
LIP_MOVING_WINDOW_SECONDS = _read_env_float("APOLLO_LIP_MOVING_WINDOW", 0.5)
LIP_EXTEND_CAPTURE_SECONDS = _read_env_float("APOLLO_LIP_EXTEND_CAPTURE", 2.0)
LIP_SYNC_WINDOW_SIZE = _read_env_int("APOLLO_LIP_SYNC_WINDOW_SIZE", 120)
LIP_SYNC_POSITION = os.environ.get("APOLLO_LIP_SYNC_POSITION", "bottom-right").strip()

# ---------------------------------------------------------------------------
# Gemini / LLM
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
LLM_FALLBACK_ENABLED = bool(GEMINI_API_KEY)
APOLLO_2STAGE_PLANNER = os.environ.get("APOLLO_2STAGE_PLANNER", "").lower() in {"1", "true", "yes"}
GEMINI_MODEL = os.environ.get("APOLLO_GEMINI_MODEL", "gemini-3-pro-preview").strip() or "gemini-3-pro-preview"
GEMINI_ROUTER_MODEL = os.environ.get("APOLLO_GEMINI_ROUTER_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_PLANNER_MODEL = os.environ.get("APOLLO_GEMINI_PLANNER_MODEL", "gemini-3-pro-preview").strip() or "gemini-3-pro-preview"
GEMINI_VISION_MODEL = os.environ.get("APOLLO_GEMINI_VISION_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
DEFAULT_GEMINI_EVAL_PLANNER_MODEL = "gemini-2.5-flash"
SCREEN_SCALE = 2

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
MIN_COMMAND_CONFIDENCE = 0.35
ROUTER_BYPASS_CONFIDENCE = 0.90
LLM_DEFER_CONFIDENCE = 0.6
CONF_DIRECT_HIGH = _read_env_float("APOLLO_CONF_DIRECT_HIGH", ROUTER_BYPASS_CONFIDENCE)
CONF_DIRECT_MID = _read_env_float("APOLLO_CONF_DIRECT_MID", LLM_DEFER_CONFIDENCE)
CONF_MIN_MATCH = _read_env_float("APOLLO_CONF_MIN_MATCH", MIN_COMMAND_CONFIDENCE)
UNKNOWN_MAX_WORDS = _read_env_int("APOLLO_UNKNOWN_MAX_WORDS", 2)

# ---------------------------------------------------------------------------
# Workflow limits
# ---------------------------------------------------------------------------
MAX_WORKFLOW_STEPS = 12
MAX_WORKFLOW_REPLANS = 2
MAX_WORKFLOW_VISION_STEPS = 4
MAX_WORKFLOW_WAIT_SECONDS = 5.0
WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS = _read_env_float("APOLLO_WAIT_FOR_STATE_TIMEOUT_SECONDS", 5.0)
WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL = _read_env_float("APOLLO_WAIT_FOR_STATE_POLL_INTERVAL", 0.3)
WAIT_FOR_STATE_MIN_POLL_INTERVAL = _read_env_float("APOLLO_WAIT_FOR_STATE_MIN_POLL_INTERVAL", 0.1)
WAIT_FOR_STATE_MAX_POLL_INTERVAL = _read_env_float("APOLLO_WAIT_FOR_STATE_MAX_POLL_INTERVAL", 1.0)
ROUTER_MAX_OUTPUT_TOKENS = 200
WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS = 700
ROUTER_TIMEOUT_SECONDS = 30.0
WORKFLOW_PLANNER_TIMEOUT_SECONDS = 60.0
GEMINI_STRUCTURED_REPAIR_INSTRUCTION = "Return ONLY a valid JSON object matching the response schema."

# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------
AX_QUERY_TIMEOUT_SECONDS = _read_env_float("APOLLO_AX_QUERY_TIMEOUT_SECONDS", 1.5)
AX_QUERY_MAX_DEPTH = _read_env_int("APOLLO_AX_QUERY_MAX_DEPTH", 5)
AX_QUERY_MAX_RESULTS = _read_env_int("APOLLO_AX_QUERY_MAX_RESULTS", 10)
AX_QUERY_MAX_CHILDREN = _read_env_int("APOLLO_AX_QUERY_MAX_CHILDREN", 40)

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
CLIPBOARD_PASTE_THRESHOLD = _read_env_int("APOLLO_CLIPBOARD_PASTE_THRESHOLD", 80)

VISION_MIN_CLICK_CONFIDENCE = _read_env_float("APOLLO_VISION_MIN_CLICK_CONFIDENCE", 0.6)
VISION_CLICK_SETTLE_SECONDS = _read_env_float("APOLLO_VISION_CLICK_SETTLE_SECONDS", 0.4)
VISION_CLICK_RETRY_LIMIT = max(1, _read_env_int("APOLLO_VISION_CLICK_RETRY_LIMIT", 2))
VISION_VERIFICATION_TIMEOUT_SECONDS = _read_env_float("APOLLO_VISION_VERIFICATION_TIMEOUT_SECONDS", 10.0)

# ---------------------------------------------------------------------------
# Lookup tables / sets
# ---------------------------------------------------------------------------
AX_PREFERRED_APPS: Set[str] = {
    "Finder",
    "Mail",
    "Messages",
    "Notes",
    "Preview",
    "System Settings",
    "TextEdit",
}
AX_AVOID_APPS: Set[str] = {
    "Claude",
    "Discord",
    "Google Chrome",
    "Slack",
    "Spotify",
    "Visual Studio Code",
}
WAIT_FOR_STATE_CONDITIONS: Set[str] = {
    "app_frontmost",
    "window_exists",
    "element_exists",
    "element_value_contains",
}
AX_ROLE_KEYWORDS = {
    "button": "button",
    "tab": "tab",
    "checkbox": "checkbox",
    "check box": "checkbox",
    "radio": "radio",
    "menu": "menu",
    "dialog": "window",
    "window": "window",
    "field": "text field",
    "text field": "text field",
    "input": "text field",
    "search": "text field",
    "sidebar": "group",
}
VISION_HINT_WORDS: Set[str] = {
    "click", "tap", "press", "button", "tab", "section", "feature",
    "chat", "sidebar", "panel", "screen", "onscreen", "on", "window",
}
MULTI_STEP_HINT_WORDS: Set[str] = {
    "and", "then", "after", "before", "click", "press", "select", "choose",
    "chat", "model", "new", "type", "write", "enter", "submit", "send",
    "switch", "ensure", "ask", "tell",
}
OPEN_REQUEST_BLOCKERS: Set[str] = {
    "click", "press", "type", "write", "ask", "tell", "chat", "section",
    "feature", "model", "today", "weather", "ensure", "select", "choose",
    "button", "screen", "onscreen", "right", "left",
}
NEGATION_PHRASES = ("don t", "dont", "do not", "instead of", "rather than")
SECRET_REPLACEMENT = "[REDACTED]"

COMMON_TRANSCRIPT_REPLACEMENTS = (
    (r"\bpiggy\b", "biggie"),
    (r"\bbig e\b", "biggie"),
    (r"\bbiddy\b", "biggie"),
    (r"\bbiggy\b", "biggie"),
    (r"\bpicky\b", "biggie"),
    (r"\bcloud\b", "claude"),
    (r"\bclawed\b", "claude"),
    (r"\bclod\b", "claude"),
    (r"\bconclude\b", "claude"),
    (r"\bvscode\b", "vs code"),
)
REQUEST_PREFIX_PATTERNS = (
    r"^(?:hey|hi|okay|ok|so|well|um|uh)\s+",
    r"^(?:can|could|would|will)\s+you\s+",
    r"^please\s+",
    r"^go\s+ahead\s+and\s+",
    r"^i\s+need\s+you\s+to\s+",
    r"^i\s+want\s+you\s+to\s+",
    r"^let\s+s\s+",
    r"^just\s+",
)
REQUEST_SUFFIX_PATTERNS = (
    r"\s+please$",
    r"\s+for\s+me$",
    r"\s+right\s+now$",
)
GENERIC_APP_ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "browser": "Google Chrome",
    "my browser": "Google Chrome",
    "web browser": "Google Chrome",
    "chrome browser": "Google Chrome",
    "safari": "Safari",
    "spotify": "Spotify",
    "terminal": "Terminal",
    "finder": "Finder",
    "messages": "Messages",
    "imessage": "Messages",
    "text messages": "Messages",
    "claude": "Claude",
    "claude code": "Claude",
    "claude app": "Claude",
    "code": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "v s code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
}
CLICK_TARGET_KEYWORDS: Set[str] = {
    "enter", "return", "tab", "escape", "esc", "delete", "backspace",
    "space", "up", "down", "left", "right",
}
CAPTURE_CANCEL_PHRASES: Set[str] = {"cancel", "never mind", "nevermind", "forget it"}
