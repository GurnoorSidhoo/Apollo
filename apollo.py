#!/usr/bin/env python3
"""
Biggie - Voice-controlled macOS assistant for coding
=====================================================
A simple Python program you run in Terminal. It listens for "Biggie",
then executes voice commands to control your Mac.

Usage:
    python3 apollo.py

Requirements (install once):
    pip3 install sounddevice numpy deepgram-sdk google-genai

    Set your Deepgram API key:
    export DEEPGRAM_API_KEY="your-key-here"

    Optionally set Gemini for AI fallback + vision:
    export GEMINI_API_KEY="your-key-here"

Permissions needed (macOS will prompt you):
    - Microphone access (for listening)
    - Accessibility access (for controlling apps) — enable in:
      System Settings → Privacy & Security → Accessibility → add Terminal
"""

import subprocess
import json
import time
import threading
import os
import queue
import re
import sys
import difflib
import importlib
import tempfile
import base64
import urllib.request
import urllib.error
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union
from urllib.parse import urlparse


def _read_env_float(name, default):
    """Parse a float env var, falling back to the supplied default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _read_env_int(name, default):
    """Parse an int env var, falling back to the supplied default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WAKE_WORD = "biggie"
SAMPLE_RATE = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMAND_LOG = os.path.join(BASE_DIR, "apollo_history.json")
DEBUG_LOG = os.path.join(BASE_DIR, "apollo_debug.jsonl")
TRANSCRIPT_LOG = os.path.join(BASE_DIR, "apollo_transcripts.jsonl")
AI_TRACE_LOG = os.path.join(BASE_DIR, "apollo_ai_trace.jsonl")
VISION_DEBUG_DIR = os.path.join(BASE_DIR, "apollo_vision_debug")
DEBUG_AUDIO = os.environ.get("APOLLO_DEBUG_AUDIO", "").lower() in {"1", "true", "yes"}
VERBOSE_AI_TRACE = os.environ.get("APOLLO_VERBOSE_AI", "").lower() in {"1", "true", "yes"}
SAVE_VISION_DEBUG = os.environ.get("APOLLO_SAVE_VISION_DEBUG", "").lower() in {"1", "true", "yes"}
VOICE_FEEDBACK = False       # Keep TTS off by default so Biggie doesn't hear itself
AUDIO_FEEDBACK = True        # Play system sounds for wake word, success, failure
WAKE_SOUND = "/System/Library/Sounds/Tink.aiff"
COMMAND_SUCCESS_SOUND = "/System/Library/Sounds/Glass.aiff"
COMMAND_FAIL_SOUND = "/System/Library/Sounds/Basso.aiff"
COMMAND_TIMEOUT = 4.0
WAKE_WORD_ALIASES = {"biggie", "biggy", "bigie", "big e", "piggy", "biddy", "picky"}
DEEPGRAM_API_KEY = "".join(ch for ch in os.environ.get("DEEPGRAM_API_KEY", "") if ch.isprintable()).strip()
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_KEYTERM = os.environ.get("APOLLO_DEEPGRAM_KEYTERM", "Biggie").strip() or "Biggie"
DEEPGRAM_UTTERANCE_END_MS = "1000"
DEEPGRAM_ENDPOINTING_MS = 300
DEEPGRAM_RECONNECT_DELAY = 2.0
DEEPGRAM_MAX_INIT_FAILURES = 2
WHISPER_MODEL = os.environ.get("APOLLO_WHISPER_MODEL", "tiny").strip() or "tiny"
WHISPER_CHUNK_SECONDS = float(os.environ.get("APOLLO_WHISPER_CHUNK_SECONDS", "2.5"))
WHISPER_SILENCE_THRESHOLD = float(os.environ.get("APOLLO_WHISPER_SILENCE_THRESHOLD", "0.008"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
LLM_FALLBACK_ENABLED = bool(GEMINI_API_KEY)
APOLLO_2STAGE_PLANNER = os.environ.get("APOLLO_2STAGE_PLANNER", "").lower() in {"1", "true", "yes"}
GEMINI_MODEL = os.environ.get("APOLLO_GEMINI_MODEL", "gemini-3-pro-preview").strip() or "gemini-3-pro-preview"
GEMINI_ROUTER_MODEL = os.environ.get("APOLLO_GEMINI_ROUTER_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_PLANNER_MODEL = os.environ.get("APOLLO_GEMINI_PLANNER_MODEL", "gemini-3-pro-preview").strip() or "gemini-3-pro-preview"
GEMINI_VISION_MODEL = os.environ.get("APOLLO_GEMINI_VISION_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
SCREEN_SCALE = 2  # Retina display scale factor
MIN_COMMAND_CONFIDENCE = 0.35
ROUTER_BYPASS_CONFIDENCE = 0.90
LLM_DEFER_CONFIDENCE = 0.6
CONF_DIRECT_HIGH = _read_env_float("APOLLO_CONF_DIRECT_HIGH", ROUTER_BYPASS_CONFIDENCE)
CONF_DIRECT_MID = _read_env_float("APOLLO_CONF_DIRECT_MID", LLM_DEFER_CONFIDENCE)
CONF_MIN_MATCH = _read_env_float("APOLLO_CONF_MIN_MATCH", MIN_COMMAND_CONFIDENCE)
UNKNOWN_MAX_WORDS = _read_env_int("APOLLO_UNKNOWN_MAX_WORDS", 2)
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
AX_QUERY_TIMEOUT_SECONDS = _read_env_float("APOLLO_AX_QUERY_TIMEOUT_SECONDS", 1.5)
AX_QUERY_MAX_DEPTH = _read_env_int("APOLLO_AX_QUERY_MAX_DEPTH", 5)
AX_QUERY_MAX_RESULTS = _read_env_int("APOLLO_AX_QUERY_MAX_RESULTS", 10)
AX_QUERY_MAX_CHILDREN = _read_env_int("APOLLO_AX_QUERY_MAX_CHILDREN", 40)
VISION_MIN_CLICK_CONFIDENCE = _read_env_float("APOLLO_VISION_MIN_CLICK_CONFIDENCE", 0.6)
VISION_CLICK_SETTLE_SECONDS = _read_env_float("APOLLO_VISION_CLICK_SETTLE_SECONDS", 0.4)
VISION_CLICK_RETRY_LIMIT = max(1, _read_env_int("APOLLO_VISION_CLICK_RETRY_LIMIT", 2))
VISION_VERIFICATION_TIMEOUT_SECONDS = _read_env_float("APOLLO_VISION_VERIFICATION_TIMEOUT_SECONDS", 10.0)
AX_PREFERRED_APPS = {
    "Finder",
    "Mail",
    "Messages",
    "Notes",
    "Preview",
    "System Settings",
    "TextEdit",
}
AX_AVOID_APPS = {
    "Claude",
    "Discord",
    "Google Chrome",
    "Slack",
    "Spotify",
    "Visual Studio Code",
}
WAIT_FOR_STATE_CONDITIONS = {
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
VISION_HINT_WORDS = {
    "click", "tap", "press", "button", "tab", "section", "feature",
    "chat", "sidebar", "panel", "screen", "onscreen", "on", "window",
}
MULTI_STEP_HINT_WORDS = {
    "and", "then", "after", "before", "click", "press", "select", "choose",
    "chat", "model", "new", "type", "write", "enter", "submit", "send",
    "switch", "ensure", "ask", "tell",
}
OPEN_REQUEST_BLOCKERS = {
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
CLICK_TARGET_KEYWORDS = {
    "enter", "return", "tab", "escape", "esc", "delete", "backspace",
    "space", "up", "down", "left", "right",
}
CAPTURE_CANCEL_PHRASES = {"cancel", "never mind", "nevermind", "forget it"}

# ---------------------------------------------------------------------------
# Command Registry
# ---------------------------------------------------------------------------
COMMANDS = []


class CommandArgs(TypedDict):
    text: str


class _RouterCommandRequired(TypedDict):
    action: Literal["command"]
    function: str
    reason: str


class RouterCommandOutput(_RouterCommandRequired, total=False):
    args: CommandArgs


class RouterWorkflowOutput(TypedDict):
    action: Literal["workflow"]
    reason: str


class RouterUnknownOutput(TypedDict):
    action: Literal["unknown"]
    reason: str


RouterOutput = Union[RouterCommandOutput, RouterWorkflowOutput, RouterUnknownOutput]


class Route(Enum):
    DIRECT = "direct"
    WORKFLOW = "workflow"
    ROUTER = "router"
    UNKNOWN = "unknown"


class _OpenAppStepRequired(TypedDict):
    type: Literal["open_app"]
    app: str
    reason: str


class OpenAppStep(_OpenAppStepRequired, total=False):
    fallback_url: str


class FocusAppStep(TypedDict):
    type: Literal["focus_app"]
    app: str
    reason: str


class QuitAppStep(TypedDict):
    type: Literal["quit_app"]
    app: str
    reason: str


class OpenUrlStep(TypedDict):
    type: Literal["open_url"]
    url: str
    reason: str


class _CommandStepRequired(TypedDict):
    type: Literal["command"]
    function: str
    reason: str


class CommandStep(_CommandStepRequired, total=False):
    args: CommandArgs


class _KeypressStepRequired(TypedDict):
    type: Literal["keypress"]
    key: str
    reason: str


class KeypressStep(_KeypressStepRequired, total=False):
    command: bool
    shift: bool
    ctrl: bool
    option: bool


class TypeTextStep(TypedDict):
    type: Literal["type_text"]
    text: str
    reason: str


class WaitStep(TypedDict):
    type: Literal["wait"]
    seconds: float
    reason: str


class _WaitForStateStepRequired(TypedDict):
    type: Literal["wait_for_state"]
    condition: Literal["app_frontmost", "window_exists", "element_exists", "element_value_contains"]
    reason: str


class WaitForStateStep(_WaitForStateStepRequired, total=False):
    app: str
    label: str
    role: str
    substring: str
    timeout_seconds: float
    poll_interval: float


class VisionStep(TypedDict):
    type: Literal["vision"]
    task: str
    reason: str


class SayStep(TypedDict):
    type: Literal["say"]
    text: str
    reason: str


WorkflowStep = Union[
    OpenAppStep,
    FocusAppStep,
    QuitAppStep,
    OpenUrlStep,
    CommandStep,
    KeypressStep,
    TypeTextStep,
    WaitStep,
    WaitForStateStep,
    VisionStep,
    SayStep,
]


class WorkflowPlannerOutput(TypedDict):
    description: str
    steps: List[WorkflowStep]


WorkflowReplanOutput = WorkflowPlannerOutput


class VisionActionMetadata(TypedDict, total=False):
    target_label: str
    confidence: float
    rationale: str
    expected_postcondition: str


class _VisionClickActionRequired(TypedDict):
    action: Literal["click"]
    x: int
    y: int
    description: str


class VisionClickAction(_VisionClickActionRequired, VisionActionMetadata):
    pass


class VisionWaitAction(TypedDict):
    type: Literal["wait"]
    seconds: float
    description: str


class _VisionClickStepRequired(TypedDict):
    type: Literal["click"]
    x: int
    y: int
    description: str


class VisionClickStep(_VisionClickStepRequired, VisionActionMetadata):
    pass


class _VisionStepsActionRequired(TypedDict):
    action: Literal["steps"]
    steps: List[Union[VisionClickStep, VisionWaitAction]]
    description: str


class VisionStepsAction(_VisionStepsActionRequired, VisionActionMetadata):
    pass


class _VisionNotFoundActionRequired(TypedDict):
    action: Literal["not_found"]
    description: str


class VisionNotFoundAction(_VisionNotFoundActionRequired, VisionActionMetadata):
    pass


class _VisionNoopActionRequired(TypedDict):
    action: Literal["noop"]
    description: str


class VisionNoopAction(_VisionNoopActionRequired, VisionActionMetadata):
    pass


VisionActionOutput = Union[
    VisionClickAction,
    VisionStepsAction,
    VisionNotFoundAction,
    VisionNoopAction,
]


ROUTER_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["command", "workflow", "unknown"],
            "description": "High-level routing decision.",
        },
        "function": {
            "type": "string",
            "minLength": 1,
            "description": "Registered command name when action is command.",
        },
        "args": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
            "description": "Arguments only used for type_text.",
        },
        "reason": {
            "type": "string",
            "minLength": 3,
            "maxLength": 60,
            "description": "Short observability reason.",
        },
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


WORKFLOW_STEP_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "open_app",
                "focus_app",
                "quit_app",
                "open_url",
                "command",
                "keypress",
                "type_text",
                "wait",
                "wait_for_state",
                "vision",
                "say",
            ],
        },
        "reason": {"type": "string", "minLength": 3, "maxLength": 60},
        "app": {"type": "string", "minLength": 1},
        "fallback_url": {"type": "string", "format": "uri"},
        "url": {"type": "string", "format": "uri", "minLength": 1},
        "function": {"type": "string", "minLength": 1},
        "args": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "key": {"type": "string", "minLength": 1},
        "command": {"type": "boolean"},
        "shift": {"type": "boolean"},
        "ctrl": {"type": "boolean"},
        "option": {"type": "boolean"},
        "text": {"type": "string", "minLength": 1},
        "seconds": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "condition": {"type": "string", "enum": sorted(WAIT_FOR_STATE_CONDITIONS)},
        "label": {"type": "string", "minLength": 1},
        "role": {"type": "string", "minLength": 1},
        "substring": {"type": "string", "minLength": 1},
        "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "poll_interval": {"type": "number", "minimum": 0.1, "maximum": 1.0},
        "task": {"type": "string", "minLength": 5},
    },
    "required": ["type", "reason"],
    "additionalProperties": False,
}


WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1, "maxLength": 200},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": WORKFLOW_STEP_RESPONSE_JSON_SCHEMA,
        },
    },
    "required": ["description", "steps"],
    "additionalProperties": False,
}


WORKFLOW_REPLAN_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": dict(WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA["properties"]),
    "required": list(WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA["required"]),
    "additionalProperties": False,
}


VISION_ACTION_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "steps", "not_found", "noop"],
        },
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "description": {"type": "string", "minLength": 1, "maxLength": 120},
        "target_label": {"type": "string", "minLength": 1, "maxLength": 120},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "minLength": 3, "maxLength": 240},
        "expected_postcondition": {"type": "string", "minLength": 3, "maxLength": 240},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["click", "wait"]},
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "seconds": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                    "description": {"type": "string", "minLength": 1, "maxLength": 120},
                    "target_label": {"type": "string", "minLength": 1, "maxLength": 120},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string", "minLength": 3, "maxLength": 240},
                    "expected_postcondition": {"type": "string", "minLength": 3, "maxLength": 240},
                },
                "required": ["type", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "description"],
    "additionalProperties": False,
}


VISION_POSTCONDITION_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 160},
    },
    "required": ["satisfied", "reason"],
    "additionalProperties": False,
}


ROUTER_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RouterOutput",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"const": "command"},
                "function": {"type": "string", "minLength": 1},
                "args": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
                "reason": {"type": "string", "minLength": 3, "maxLength": 60},
            },
            "required": ["action", "function", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "workflow"},
                "reason": {"type": "string", "minLength": 3, "maxLength": 60},
            },
            "required": ["action", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "unknown"},
                "reason": {"type": "string", "minLength": 3, "maxLength": 60},
            },
            "required": ["action", "reason"],
            "additionalProperties": False,
        },
    ],
}


WORKFLOW_PLANNER_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorkflowPlannerOutput",
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1, "maxLength": 200},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "open_app"},
                            "app": {"type": "string", "minLength": 1},
                            "fallback_url": {"type": "string", "format": "uri"},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "app", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "focus_app"},
                            "app": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "app", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "quit_app"},
                            "app": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "app", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "open_url"},
                            "url": {"type": "string", "format": "uri", "minLength": 1},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "url", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "command"},
                            "function": {"type": "string", "minLength": 1},
                            "args": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "minLength": 1},
                                },
                                "additionalProperties": False,
                            },
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "function", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "keypress"},
                            "key": {"type": "string", "minLength": 1},
                            "command": {"type": "boolean", "default": False},
                            "shift": {"type": "boolean", "default": False},
                            "ctrl": {"type": "boolean", "default": False},
                            "option": {"type": "boolean", "default": False},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "key", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "type_text"},
                            "text": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "text", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "wait"},
                            "seconds": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "seconds", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "wait_for_state"},
                            "condition": {"type": "string", "enum": sorted(WAIT_FOR_STATE_CONDITIONS)},
                            "app": {"type": "string", "minLength": 1},
                            "label": {"type": "string", "minLength": 1},
                            "role": {"type": "string", "minLength": 1},
                            "substring": {"type": "string", "minLength": 1},
                            "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                            "poll_interval": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "condition", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "vision"},
                            "task": {"type": "string", "minLength": 5},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "task", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "say"},
                            "text": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 3, "maxLength": 60},
                        },
                        "required": ["type", "text", "reason"],
                        "additionalProperties": False,
                    },
                ],
            },
        },
    },
    "required": ["description", "steps"],
    "additionalProperties": False,
}

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
# Utility functions
# ---------------------------------------------------------------------------

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
    """Return (detected, trailing_text) for a wake word at utterance start.

    We anchor detection to the start of the utterance so conversational mentions
    like "yeah my biggie is stupid" do not accidentally enter command mode.
    A conservative fuzzy check is still allowed on the leading wake-word token
    to absorb common STT mistakes such as "biggy" or "bigie".
    """
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


# ===========================================================================
# COMMANDS — Add your own commands here! Just copy the pattern.
# ===========================================================================

# --- App Launching ---

@command(
    phrases=["open claude", "launch claude", "start claude", "go to claude",
             "open cloud", "launch cloud"],
    description="Opens the Claude Mac app"
)
def open_claude():
    mac_open_app("Claude", fallback_url="https://claude.ai")
    say("Opening Claude")

@command(
    phrases=["open terminal", "launch terminal", "start terminal"],
    description="Opens the Terminal app"
)
def open_terminal():
    mac_open_app("Terminal")
    say("Opening Terminal")

@command(
    phrases=["open vs code", "open vscode", "launch vs code", "open code",
             "launch code", "start vs code"],
    description="Opens Visual Studio Code"
)
def open_vscode():
    mac_open_app("Visual Studio Code")
    say("Opening VS Code")

@command(
    phrases=["open safari", "launch safari"],
    description="Opens Safari"
)
def open_safari():
    mac_open_app("Safari")
    say("Opening Safari")

@command(
    phrases=["open chrome", "launch chrome", "open google chrome",
             "open browser", "launch browser", "open my browser"],
    description="Opens Google Chrome"
)
def open_chrome():
    mac_open_app("Google Chrome")
    say("Opening Chrome")

@command(
    phrases=["open finder", "launch finder", "open files"],
    description="Opens Finder"
)
def open_finder():
    mac_open_app("Finder")
    say("Opening Finder")

@command(
    phrases=["open spotify", "launch spotify", "go to spotify"],
    description="Opens Spotify"
)
def open_spotify():
    mac_open_app("Spotify")
    say("Opening Spotify")


@command(
    phrases=[
        "play spotify",
        "play music on spotify",
        "play some music on spotify",
        "resume spotify",
        "resume music",
        "play music",
    ],
    description="Starts or resumes Spotify playback"
)
def play_spotify():
    spotify_command("play")
    say("Playing Spotify")


@command(
    phrases=["pause spotify", "pause music", "stop spotify", "pause the music"],
    description="Pauses Spotify playback"
)
def pause_spotify():
    spotify_command("pause")
    say("Pausing Spotify")


@command(
    phrases=["next song", "next track", "skip song", "skip track"],
    description="Skips to the next Spotify track"
)
def next_spotify_track():
    spotify_command("next track")
    say("Skipping track")


@command(
    phrases=["previous song", "previous track", "last song", "go back song"],
    description="Returns to the previous Spotify track"
)
def previous_spotify_track():
    spotify_command("previous track")
    say("Going back a track")

@command(
    phrases=["open github", "go to github", "launch github"],
    description="Opens GitHub in browser"
)
def open_github():
    mac_open("https://github.com")
    say("Opening GitHub")


# --- Window Management ---

@command(
    phrases=["close window", "close this", "close it"],
    description="Closes the current window (Cmd+W)"
)
def close_window():
    hotkey("w", command=True)
    say("Closing window")

@command(
    phrases=["close app", "quit app", "quit this", "force quit"],
    description="Quits the current app (Cmd+Q)"
)
def quit_app():
    hotkey("q", command=True)
    say("Quitting app")

@command(
    phrases=["scroll down", "page down"],
    description="Scrolls the current view down"
)
def scroll_down():
    press_key("page down")
    say("Scrolling down")

@command(
    phrases=["scroll up", "page up"],
    description="Scrolls the current view up"
)
def scroll_up():
    press_key("page up")
    say("Scrolling up")

@command(
    phrases=["new tab", "open new tab"],
    description="Opens a new tab (Cmd+T)"
)
def new_tab():
    hotkey("t", command=True)
    say("New tab")

@command(
    phrases=["next tab", "switch tab", "go to next tab"],
    description="Switches to next tab"
)
def next_tab():
    hotkey("}", command=True, shift=True)

@command(
    phrases=["previous tab", "go back tab", "last tab"],
    description="Switches to previous tab"
)
def prev_tab():
    hotkey("{", command=True, shift=True)

@command(
    phrases=["minimise", "minimize", "minimise window", "hide window"],
    description="Minimises current window"
)
def minimise():
    hotkey("m", command=True)
    say("Minimised")

@command(
    phrases=["full screen", "fullscreen", "go full screen", "maximise", "maximize"],
    description="Toggles fullscreen"
)
def fullscreen():
    hotkey("f", command=True, ctrl=True)


# --- Coding Shortcuts ---

@command(
    phrases=["save", "save file", "save this"],
    description="Saves current file (Cmd+S)"
)
def save_file():
    hotkey("s", command=True)
    say("Saved")

@command(
    phrases=["undo", "undo that"],
    description="Undo (Cmd+Z)"
)
def undo():
    hotkey("z", command=True)

@command(
    phrases=["redo", "redo that"],
    description="Redo (Cmd+Shift+Z)"
)
def redo():
    hotkey("z", command=True, shift=True)

@command(
    phrases=["copy", "copy that", "copy this"],
    description="Copy selection (Cmd+C)"
)
def copy():
    hotkey("c", command=True)

@command(
    phrases=["paste", "paste that", "paste it"],
    description="Paste (Cmd+V)"
)
def paste():
    hotkey("v", command=True)

@command(
    phrases=["cut", "cut that", "cut this"],
    description="Cut selection (Cmd+X)"
)
def cut():
    hotkey("x", command=True)

@command(
    phrases=["select all", "select everything"],
    description="Select all (Cmd+A)"
)
def select_all():
    hotkey("a", command=True)

@command(
    phrases=["find", "search", "find in file", "search file"],
    description="Find (Cmd+F)"
)
def find():
    hotkey("f", command=True)
    say("Find activated")

@command(
    phrases=["comment", "comment line", "toggle comment", "comment out",
             "comment this"],
    description="Toggle comment in code editor (Cmd+/)"
)
def comment_line():
    hotkey("/", command=True)

@command(
    phrases=["go to line", "jump to line"],
    description="Go to line in VS Code (Ctrl+G)"
)
def goto_line():
    hotkey("g", ctrl=True)
    say("Go to line")

@command(
    phrases=["open command palette", "command palette", "palette"],
    description="Opens VS Code command palette (Cmd+Shift+P)"
)
def command_palette():
    hotkey("p", command=True, shift=True)
    say("Command palette")

@command(
    phrases=["open file", "quick open"],
    description="Quick open file in VS Code (Cmd+P)"
)
def quick_open():
    hotkey("p", command=True)

@command(
    phrases=["toggle sidebar", "hide sidebar", "show sidebar"],
    description="Toggles VS Code sidebar (Cmd+B)"
)
def toggle_sidebar():
    hotkey("b", command=True)

@command(
    phrases=["toggle terminal", "show terminal", "hide terminal",
             "open terminal panel"],
    description="Toggles integrated terminal in VS Code (Ctrl+`)"
)
def toggle_terminal():
    hotkey("`", ctrl=True)

@command(
    phrases=["split editor", "split screen", "split view"],
    description="Splits the editor in VS Code"
)
def split_editor():
    hotkey("\\", command=True)

@command(
    phrases=["run code", "run this", "run file", "execute"],
    description="Runs code — sends Cmd+Shift+B (build task) or you can change this"
)
def run_code():
    hotkey("b", command=True, shift=True)
    say("Running")


# --- System Commands ---

@command(
    phrases=["take screenshot", "screenshot", "screen capture"],
    description="Takes a screenshot (Cmd+Shift+3)"
)
def screenshot():
    hotkey("3", command=True, shift=True)
    say("Screenshot taken")

@command(
    phrases=["lock screen", "lock computer", "lock mac"],
    description="Locks the screen"
)
def lock_screen():
    hotkey("q", command=True, ctrl=True)
    say("Locking")

@command(
    phrases=["volume up", "louder", "turn it up", "make it louder",
             "turn volume up", "a bit louder"],
    description="Increases volume"
)
def volume_up():
    applescript('set volume output volume ((output volume of (get volume settings)) + 10)')

@command(
    phrases=["volume down", "quieter", "turn it down"],
    description="Decreases volume"
)
def volume_down():
    applescript('set volume output volume ((output volume of (get volume settings)) - 10)')

@command(
    phrases=["mute", "mute volume", "shut up", "silence"],
    description="Mutes volume"
)
def mute():
    applescript('set volume output muted true')
    say("Muted")

@command(
    phrases=["unmute", "unmute volume"],
    description="Unmutes volume"
)
def unmute():
    applescript('set volume output muted false')
    say("Unmuted")


# --- Typing / Dictation ---

@command(
    phrases=["type", "write", "dictate"],
    description="Types out whatever you say after 'type'. E.g. 'Biggie, type hello world'"
)
def type_text():
    """Special handler — the actual text is injected by the command router."""
    pass  # Handled specially in route_command()


# --- Biggie Meta ---

@command(
    phrases=["help", "what can you do", "list commands", "show commands"],
    description="Lists all available commands"
)
def show_help():
    say("Here are my commands")
    print("\n" + "=" * 60)
    print("  BIGGIE COMMANDS")
    print("=" * 60)
    for cmd in COMMANDS:
        trigger = cmd["phrases"][0]
        desc = cmd["description"]
        print(f"  \"{trigger}\"  ->  {desc}")
    print("=" * 60 + "\n")

@command(
    phrases=["stop listening", "go to sleep", "sleep", "pause",
             "stop", "shut down", "goodbye"],
    description="Stops Biggie"
)
def stop_listening():
    say("Going to sleep. Goodbye!")
    raise SystemExit(0)

 
# ===========================================================================
# macOS CONTROL FUNCTIONS
# ===========================================================================

def play_sound(sound_path):
    """Play a macOS system sound non-blocking for instant audio feedback."""
    if not AUDIO_FEEDBACK:
        return
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def say(text):
    """Speak text aloud using macOS built-in TTS."""
    if not VOICE_FEEDBACK:
        print(f"  [speaker] {text}")
        return
    subprocess.Popen(["say", "-v", "Samantha", text])


def mac_open(url_or_path):
    """Open a URL or file path using macOS `open` command."""
    subprocess.Popen(["open", url_or_path])


def mac_open_app(app_name, fallback_url=None):
    """Open an application by name, optionally falling back to a URL."""
    result = subprocess.run(
        ["open", "-a", app_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    debug_event("open_app_failed", app=app_name, fallback_url=fallback_url,
                stderr=result.stderr.strip())
    if fallback_url:
        print(f"  !!  Could not open app \"{app_name}\". Falling back to {fallback_url}")
        mac_open(fallback_url)
        return False
    raise RuntimeError(result.stderr.strip() or f"Failed to open app: {app_name}")


def applescript(script):
    """Run an AppleScript command."""
    subprocess.Popen(["osascript", "-e", script])


def run_applescript(script):
    """Run an AppleScript command synchronously and raise on failure."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AppleScript failed")
    return result.stdout.strip()


def focus_app(app_name):
    """Bring an application to the foreground."""
    escaped = app_name.replace('"', '\\"')
    applescript(f'tell application "{escaped}" to activate')


def quit_named_app(app_name):
    """Quit a named application reliably even when it is not frontmost."""
    escaped = app_name.replace('"', '\\"')
    run_applescript(
        f'''
tell application "System Events"
    set appRunning to exists process "{escaped}"
end tell
if appRunning then
    tell application "{escaped}" to activate
    delay 0.15
    tell application "{escaped}" to quit
    return "quit"
end if
return "not_running"
'''.strip()
    )
    return True


def spotify_command(command):
    """Control Spotify playback via AppleScript."""
    run_applescript(
        f'''
tell application "Spotify"
    activate
    {command}
end tell
'''.strip()
    )


def press_key(key, command=False, shift=False, ctrl=False, option=False):
    """Simulate a single key press using AppleScript/System Events."""
    modifiers = []
    if command:
        modifiers.append("command down")
    if shift:
        modifiers.append("shift down")
    if ctrl:
        modifiers.append("control down")
    if option:
        modifiers.append("option down")

    modifier_str = ", ".join(modifiers)
    special_key_codes = {
        "\r": 36,
        "return": 36,
        "enter": 36,
        "tab": 48,
        "space": 49,
        "escape": 53,
        "esc": 53,
        "delete": 51,
        "backspace": 51,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
        "page up": 116,
        "pageup": 116,
        "page down": 121,
        "pagedown": 121,
    }
    normalized_key = str(key).lower()
    if normalized_key in special_key_codes:
        key_expr = f"key code {special_key_codes[normalized_key]}"
    else:
        escaped = str(key).replace("\\", "\\\\").replace('"', '\\"')
        key_expr = f'keystroke "{escaped}"'

    if modifier_str:
        script = f'tell application "System Events" to {key_expr} using {{{modifier_str}}}'
    else:
        script = f'tell application "System Events" to {key_expr}'
    applescript(script)


def hotkey(key, command=False, shift=False, ctrl=False, option=False):
    """Simulate a keyboard shortcut using AppleScript."""
    press_key(key, command=command, shift=shift, ctrl=ctrl, option=option)


def type_string(text):
    """Type a string character by character using AppleScript."""
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    script = f'tell application "System Events" to keystroke "{escaped}"'
    applescript(script)


def get_input_device():
    """Return a usable input device index and metadata, or (None, [])."""
    import sounddevice as sd
    devices = sd.query_devices()
    try:
        default_input = sd.query_devices(kind="input")
        return default_input["index"], devices
    except Exception:
        pass
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            return index, devices
    return None, devices


def load_custom_commands():
    """Load optional user commands from custom_commands.py without duplicating the module."""
    custom_path = os.path.join(os.path.dirname(__file__), "custom_commands.py")
    if not os.path.exists(custom_path):
        return
    try:
        sys.modules.setdefault("apollo", sys.modules[__name__])
        importlib.import_module("custom_commands")
        debug_event("custom_commands_loaded", path=custom_path)
    except Exception as e:
        print(f"  !!  Failed to load custom commands: {e}")
        debug_event("custom_commands_error", path=custom_path, error=str(e))


def parse_sips_dimensions(path):
    """Return image pixel dimensions using macOS sips."""
    result = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sips failed")

    width = None
    height = None
    for line in result.stdout.splitlines():
        if "pixelWidth:" in line:
            width = int(line.split("pixelWidth:")[1].strip())
        elif "pixelHeight:" in line:
            height = int(line.split("pixelHeight:")[1].strip())

    if not width or not height:
        raise RuntimeError(f"Could not parse image dimensions for {path}")
    return width, height


def get_main_screen_bounds():
    """Return the main screen bounds in logical coordinates."""
    jxa_script = """
ObjC.import("AppKit");
function run() {
  const screen = $.NSScreen.mainScreen;
  if (!screen) return "";
  const frame = screen.frame;
  return JSON.stringify({
    x: Number(frame.origin.x),
    y: Number(frame.origin.y),
    width: Number(frame.size.width),
    height: Number(frame.size.height)
  });
}
"""
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", jxa_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "Could not determine main screen bounds")
    return json.loads(result.stdout.strip())


def get_app_window_bounds(app_name):
    """Return the front window bounds for a macOS app in logical coordinates."""
    jxa_script = """
function run(argv) {
  const appName = argv[0];
  const systemEvents = Application("System Events");
  const process = systemEvents.processes.byName(appName);
  if (!process.exists()) return "";
  const windows = process.windows();
  if (!windows.length) return "";
  const front = windows[0];
  const position = front.position();
  const size = front.size();
  return JSON.stringify({
    x: Number(position[0]),
    y: Number(position[1]),
    width: Number(size[0]),
    height: Number(size[1])
  });
}
"""
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", jxa_script, app_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or f"Could not determine window bounds for {app_name}")
    return json.loads(result.stdout.strip())


def infer_target_app_name(text):
    """Infer a likely app name from a command or vision task."""
    normalized = strip_request_wrappers(text)
    for alias, app_name in sorted(GENERIC_APP_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in normalized:
            return app_name
    if "claude" in normalized:
        return "Claude"
    if "chrome" in normalized or "browser" in normalized:
        return "Google Chrome"
    if "safari" in normalized:
        return "Safari"
    if "message" in normalized:
        return "Messages"
    if "vs code" in normalized or "visual studio code" in normalized or "code" in normalized:
        return "Visual Studio Code"
    return ""


def _run_ax_query_json(query_name, jxa_script, argv, timeout_seconds=AX_QUERY_TIMEOUT_SECONDS):
    """Run a bounded JXA accessibility query and return parsed JSON."""
    args = [str(item) for item in argv]
    debug_event("ax_query_start", query=query_name, args=args, timeout_seconds=timeout_seconds)
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", jxa_script, *args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        debug_event("ax_query_timeout", query=query_name, args=args, timeout_seconds=timeout_seconds)
        return None

    if result.returncode != 0:
        debug_event(
            "ax_query_error",
            query=query_name,
            args=args,
            timeout_seconds=timeout_seconds,
            error=result.stderr.strip() or f"osascript exited {result.returncode}",
        )
        return None

    stdout = result.stdout.strip()
    if not stdout:
        debug_event("ax_query_result", query=query_name, args=args, timeout_seconds=timeout_seconds, empty=True)
        return None

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        debug_event(
            "ax_query_error",
            query=query_name,
            args=args,
            timeout_seconds=timeout_seconds,
            error=f"invalid json: {exc}",
            stdout=stdout[:500],
        )
        return None

    summary = {"query": query_name, "args": args, "timeout_seconds": timeout_seconds}
    if isinstance(payload, dict):
        summary["status"] = payload.get("status", "ok")
        if isinstance(payload.get("matches"), list):
            summary["match_count"] = len(payload["matches"])
        if "visited" in payload:
            summary["visited"] = payload.get("visited")
    debug_event("ax_query_result", **summary)
    return payload


def extract_first_quoted_text(text):
    """Return the first quoted span in a task description, if present."""
    if not isinstance(text, str):
        return ""
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        match = re.search(pattern, text)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return ""


def extract_target_label_from_text(text):
    """Best-effort extraction of a human-visible label from a task description."""
    quoted = extract_first_quoted_text(text)
    if quoted:
        return quoted
    normalized = strip_request_wrappers(text or "")
    if not normalized:
        return ""
    patterns = (
        r"\b(?:labeled|labelled|named|called)\s+(.+)$",
        r"^click(?:\s+on)?\s+(.+)$",
        r"^select\s+(.+)$",
        r"^open\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        label = match.group(1).strip(" .")
        label = re.split(r"\b(?:if|when|that|which)\b", label, maxsplit=1)[0].strip(" .")
        if label:
            return label
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


def query_ax_element(app_name, target_label="", target_role="", max_depth=5, max_results=10):
    """Query the app accessibility tree for matching elements, bounded by depth and results."""
    resolved_app = resolve_generic_app_name(app_name) if app_name else ""
    if not resolved_app:
        return []

    bounded_depth = max(1, min(int(max_depth), AX_QUERY_MAX_DEPTH))
    bounded_results = max(1, min(int(max_results), AX_QUERY_MAX_RESULTS))
    bounded_children = max(1, AX_QUERY_MAX_CHILDREN)
    label_hint = (target_label or "").strip()
    role_hint = (target_role or "").strip()
    jxa_script = """
function run(argv) {
  var appName = argv[0] || "";
  var targetLabel = (argv[1] || "").toLowerCase();
  var targetRole = (argv[2] || "").toLowerCase();
  var maxDepth = Math.max(1, parseInt(argv[3] || "5", 10));
  var maxResults = Math.max(1, parseInt(argv[4] || "10", 10));
  var maxChildren = Math.max(1, parseInt(argv[5] || "40", 10));
  var maxVisited = Math.max(maxResults * maxChildren * (maxDepth + 1), maxResults * 4);
  var systemEvents = Application("System Events");
  var process = systemEvents.processes.byName(appName);
  if (!process.exists()) {
    return JSON.stringify({status: "missing_process", matches: [], visited: 0});
  }

  function safeCall(fn, fallback) {
    try {
      var value = fn();
      if (value === undefined || value === null) return fallback;
      return value;
    } catch (e) {
      return fallback;
    }
  }

  function asList(value) {
    if (!value) return [];
    if (value instanceof Array) return value;
    var items = [];
    try {
      var length = value.length || 0;
      for (var index = 0; index < length; index += 1) {
        items.push(value[index]);
      }
    } catch (e) {}
    return items;
  }

  function safeList(fn) {
    return asList(safeCall(fn, []));
  }

  function scalar(value) {
    if (value === undefined || value === null) return "";
    if (value instanceof Array) {
      var pieces = [];
      for (var index = 0; index < value.length; index += 1) {
        pieces.push(String(value[index]));
      }
      return pieces.join(" ");
    }
    return String(value);
  }

  function maybeNumber(value) {
    var number = Number(value);
    return isFinite(number) ? number : null;
  }

  function normalize(value) {
    return scalar(value).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  }

  function describe(element, depth) {
    var name = scalar(safeCall(function() { return element.name(); }, ""));
    var description = scalar(safeCall(function() { return element.description(); }, ""));
    var role = scalar(safeCall(function() { return element.role(); }, ""));
    var subrole = scalar(safeCall(function() { return element.subrole(); }, ""));
    var value = scalar(safeCall(function() { return element.value(); }, ""));
    var position = safeCall(function() { return element.position(); }, null);
    var size = safeCall(function() { return element.size(); }, null);
    var x = position && position.length > 1 ? maybeNumber(position[0]) : null;
    var y = position && position.length > 1 ? maybeNumber(position[1]) : null;
    var width = size && size.length > 1 ? maybeNumber(size[0]) : null;
    var height = size && size.length > 1 ? maybeNumber(size[1]) : null;
    return {
      label: name || description || value,
      description: description,
      role: role,
      subrole: subrole,
      value: value,
      x: x,
      y: y,
      width: width,
      height: height,
      depth: depth
    };
  }

  function matchesTarget(info) {
    var roleText = normalize(info.role + " " + info.subrole);
    if (targetRole && roleText.indexOf(targetRole) === -1) {
      return false;
    }
    if (!targetLabel) {
      return !!targetRole;
    }
    var labelText = normalize(info.label + " " + info.description + " " + info.value);
    return labelText.indexOf(targetLabel) !== -1;
  }

  function childrenOf(element) {
    var rawChildren = safeList(function() { return element.uiElements(); });
    var children = [];
    for (var index = 0; index < rawChildren.length && index < maxChildren; index += 1) {
      children.push(rawChildren[index]);
    }
    return children;
  }

  var roots = safeList(function() { return process.windows(); });
  if (!roots.length) {
    roots = childrenOf(process);
  }

  var matches = [];
  var visited = 0;

  function walk(element, depth) {
    if (!element || visited >= maxVisited || matches.length >= maxResults || depth > maxDepth) {
      return;
    }
    visited += 1;
    var info = describe(element, depth);
    if (matchesTarget(info)) {
      matches.push(info);
      if (matches.length >= maxResults) {
        return;
      }
    }
    if (depth >= maxDepth) {
      return;
    }
    var children = childrenOf(element);
    for (var index = 0; index < children.length; index += 1) {
      if (matches.length >= maxResults || visited >= maxVisited) {
        break;
      }
      walk(children[index], depth + 1);
    }
  }

  for (var rootIndex = 0; rootIndex < roots.length; rootIndex += 1) {
    if (matches.length >= maxResults || visited >= maxVisited) {
      break;
    }
    walk(roots[rootIndex], 0);
  }

  return JSON.stringify({status: "ok", matches: matches, visited: visited});
}
"""
    payload = _run_ax_query_json(
        "element",
        jxa_script,
        [resolved_app, label_hint.lower(), role_hint.lower(), bounded_depth, bounded_results, bounded_children],
    )
    if payload is None:
        return None

    normalized_matches = []
    for raw_match in payload.get("matches", []):
        if not isinstance(raw_match, dict):
            continue
        x = raw_match.get("x")
        y = raw_match.get("y")
        width = raw_match.get("width")
        height = raw_match.get("height")
        match = {
            "label": str(raw_match.get("label", "")).strip(),
            "description": str(raw_match.get("description", "")).strip(),
            "role": str(raw_match.get("role", "")).strip(),
            "subrole": str(raw_match.get("subrole", "")).strip(),
            "value": str(raw_match.get("value", "")).strip(),
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "depth": int(raw_match.get("depth", 0) or 0),
        }
        if all(isinstance(value, (int, float)) for value in (x, y, width, height)):
            match["center_x"] = int(round(float(x) + (float(width) / 2.0)))
            match["center_y"] = int(round(float(y) + (float(height) / 2.0)))
        normalized_matches.append(match)
    return normalized_matches


def ax_check_app_frontmost(app_name):
    """Return True when the named app is frontmost, False when not, and None on AX failure."""
    resolved_app = resolve_generic_app_name(app_name) if app_name else ""
    if not resolved_app:
        return False
    payload = _run_ax_query_json(
        "frontmost",
        """
function run(argv) {
  var appName = argv[0] || "";
  var systemEvents = Application("System Events");
  var process = systemEvents.processes.byName(appName);
  if (!process.exists()) return JSON.stringify({status: "missing_process", frontmost: false});
  try {
    return JSON.stringify({status: "ok", frontmost: !!process.frontmost()});
  } catch (e) {
    return JSON.stringify({status: "ok", frontmost: false});
  }
}
""",
        [resolved_app],
    )
    if payload is None:
        return None
    return bool(payload.get("frontmost", False))


def ax_get_window_count(app_name):
    """Return the number of app windows, or None when AX is unavailable."""
    resolved_app = resolve_generic_app_name(app_name) if app_name else ""
    if not resolved_app:
        return 0
    payload = _run_ax_query_json(
        "window_count",
        """
function run(argv) {
  var appName = argv[0] || "";
  var systemEvents = Application("System Events");
  var process = systemEvents.processes.byName(appName);
  if (!process.exists()) return JSON.stringify({status: "missing_process", count: 0});
  var count = 0;
  try {
    count = process.windows().length;
  } catch (e) {
    count = 0;
  }
  return JSON.stringify({status: "ok", count: count});
}
""",
        [resolved_app],
    )
    if payload is None:
        return None
    try:
        return int(payload.get("count", 0))
    except (TypeError, ValueError):
        return None


def ax_get_focused_element_value(app_name):
    """Return the focused AX element value when readable, else None."""
    resolved_app = resolve_generic_app_name(app_name) if app_name else ""
    if not resolved_app:
        return ""
    payload = _run_ax_query_json(
        "focused_value",
        """
function run(argv) {
  var appName = argv[0] || "";
  var systemEvents = Application("System Events");
  var process = systemEvents.processes.byName(appName);
  if (!process.exists()) return JSON.stringify({status: "missing_process", value: ""});
  try {
    var focused = process.focusedUIElement();
    var value = focused ? focused.value() : "";
    if (value instanceof Array) {
      value = value.join(" ");
    }
    return JSON.stringify({status: "ok", value: value === undefined || value === null ? "" : String(value)});
  } catch (e) {
    return JSON.stringify({status: "ok", value: ""});
  }
}
""",
        [resolved_app],
    )
    if payload is None:
        return None
    return str(payload.get("value", ""))


def wait_for_state(condition_fn, timeout_seconds=5.0, poll_interval=0.3, condition_name=""):
    """Poll a condition function until it succeeds or the timeout expires."""
    timeout_seconds = max(0.1, float(timeout_seconds or WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS))
    poll_interval = min(
        WAIT_FOR_STATE_MAX_POLL_INTERVAL,
        max(WAIT_FOR_STATE_MIN_POLL_INTERVAL, float(poll_interval or WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL)),
    )
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    debug_event(
        "wait_for_state_start",
        condition=condition_name or "unnamed_condition",
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
    )
    while True:
        attempts += 1
        try:
            if condition_fn():
                debug_event(
                    "wait_for_state_satisfied",
                    condition=condition_name or "unnamed_condition",
                    attempts=attempts,
                )
                return True
        except Exception as exc:
            debug_event(
                "wait_for_state_condition_error",
                condition=condition_name or "unnamed_condition",
                attempts=attempts,
                error=str(exc),
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))
    debug_event(
        "wait_for_state_timeout",
        condition=condition_name or "unnamed_condition",
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )
    return False


def condition_app_frontmost(app_name):
    """Build a wait predicate that succeeds when the app is frontmost."""
    return lambda: ax_check_app_frontmost(app_name) is True


def condition_window_exists(app_name):
    """Build a wait predicate that succeeds when the app has at least one window."""
    return lambda: (ax_get_window_count(app_name) or 0) > 0


def condition_element_exists(app_name, label, role):
    """Build a wait predicate that succeeds when a named AX element exists."""
    return lambda: bool(query_ax_element(app_name, target_label=label, target_role=role, max_results=1) or [])


def condition_element_value_contains(app_name, substring):
    """Build a wait predicate that succeeds when the focused AX value contains a substring."""
    needle = (substring or "").strip().lower()
    return lambda: needle in (ax_get_focused_element_value(app_name) or "").lower()


def build_wait_for_state_condition(step):
    """Construct the executable predicate for a wait_for_state workflow step."""
    condition = step.get("condition", "").strip().lower()
    app_name = step.get("app", "").strip()
    label = step.get("label", "").strip()
    role = step.get("role", "").strip()
    substring = step.get("substring", "").strip()
    if condition == "app_frontmost":
        return condition_app_frontmost(app_name), f"app_frontmost:{app_name}"
    if condition == "window_exists":
        return condition_window_exists(app_name), f"window_exists:{app_name}"
    if condition == "element_exists":
        return condition_element_exists(app_name, label, role), f"element_exists:{app_name}:{label or role}"
    if condition == "element_value_contains":
        return condition_element_value_contains(app_name, substring), f"element_value_contains:{app_name}:{substring}"
    raise ValueError(f"Unsupported wait_for_state condition: {condition}")


def should_use_accessibility(app_name, target_description):
    """Use accessibility first for standard macOS apps and standard controls."""
    resolved_app = resolve_generic_app_name(app_name) if app_name else ""
    if not resolved_app:
        return False
    if resolved_app in AX_AVOID_APPS:
        return False
    if resolved_app in AX_PREFERRED_APPS:
        return True
    normalized = strip_request_wrappers(target_description or "")
    if any(
        word in normalized
        for word in ("video", "thumbnail", "album", "playlist", "song", "timeline", "canvas", "editor")
    ):
        return False
    return bool(extract_target_role_from_text(target_description))


# ===========================================================================
# GEMINI / LLM FALLBACK
# ===========================================================================

_gemini_client = None
_gemini_transport = None


def get_gemini_transport():
    """Prefer the SDK when installed, otherwise use Gemini's REST API directly."""
    global _gemini_transport
    if _gemini_transport is not None:
        return _gemini_transport
    if not GEMINI_API_KEY:
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
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    from google import genai
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


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
    if not GEMINI_API_KEY:
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
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}",
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
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    correlation_id = uuid.uuid4().hex[:12]
    models = preferred_models or [GEMINI_MODEL]
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
            prompt_text = f"{user_text}\n\n{GEMINI_STRUCTURED_REPAIR_INSTRUCTION}"

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
            raw_response, transport = run_with_timeout(
                timeout_seconds,
                gemini_generate_structured_candidate,
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
            if model_name != GEMINI_MODEL:
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
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    candidate_models = preferred_models or [GEMINI_MODEL]
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
                if model_name != GEMINI_MODEL:
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
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
            if model_name != GEMINI_MODEL:
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


# ===========================================================================
# VISION — screenshot capture and screen-aware clicking
# ===========================================================================

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


def capture_vision_frame(target_app, debug_prefix="vision"):
    """Capture a fresh screenshot for a vision step, preferring the target app window."""
    screenshot_region = None
    if target_app:
        try:
            screenshot_region = get_app_window_bounds(target_app)
            debug_event("vision_target_window", app=target_app, region=screenshot_region)
        except Exception as exc:
            debug_event("vision_target_window_failed", app=target_app, error=str(exc))

    image_bytes, metadata = capture_screenshot(region=screenshot_region)
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
    if not GEMINI_API_KEY:
        return {
            "status": "unavailable",
            "reason": "Gemini verification unavailable",
            "method": "none",
            "used_gemini": False,
        }

    try:
        image_bytes, metadata = capture_vision_frame(app_name, debug_prefix="vision_verify")
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


def resolve_ui_target(app_name, target_description, transcript):
    """Resolve a UI target with AX first when appropriate, then fall back to screenshot vision."""
    target_app = resolve_generic_app_name(app_name) if app_name else ""
    target_label = extract_target_label_from_text(target_description) or extract_target_label_from_text(transcript)
    target_role = extract_target_role_from_text(target_description)
    original_metadata = {
        "target_app": target_app,
        "task": target_description,
        "transcript": transcript,
        "target_label": target_label,
        "target_role": target_role,
        "window_count_before": ax_get_window_count(target_app) if target_app else None,
        "focused_value_before": ax_get_focused_element_value(target_app) if target_app else None,
    }

    if should_use_accessibility(target_app, target_description) and (target_label or target_role):
        ax_matches = query_ax_element(
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

    image_bytes, metadata = capture_vision_frame(target_app, debug_prefix="vision")
    metadata = {
        **original_metadata,
        **metadata,
    }
    result = request_vision_action(target_description, transcript, target_app, image_bytes, metadata)
    debug_event("vision_response", response=result, metadata=metadata, target_app=target_app)
    return {
        "source": "vision",
        "target_app": target_app,
        "action": result,
        "metadata": metadata,
    }


def execute_vision_steps_action(result, metadata, transcript):
    """Execute a legacy multi-step vision action while enforcing click confidence gating."""
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
                say("I couldn't safely identify where to click")
                return False
            x, y = resolve_click_coordinates(step, metadata)
            print(f"  [eye] Vision step: click at ({x}, {y})")
            say(step.get("description", "Clicking"))
            click_at(x, y)
        elif step["type"] == "wait":
            time.sleep(step.get("seconds", 1))
    log_command(transcript, "vision:steps")
    return True


def execute_vision_task(task, transcript):
    """Resolve and execute a screen task with AX-first targeting and post-click verification."""
    say("Let me look at the screen")

    target_app = infer_target_app_name(f"{task} {transcript}")
    gemini_verification_used = False

    for attempt in range(1, VISION_CLICK_RETRY_LIMIT + 1):
        try:
            resolution = resolve_ui_target(target_app, task, transcript)
        except StructuredOutputError as exc:
            planner_failure(
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
            say("I couldn't figure out where to click")
            return False
        except Exception as exc:
            error_text = str(exc)
            if any(token in error_text.lower() for token in ("screencapture", "capture", "screen")):
                print(f"  !! Screenshot failed: {exc}")
                debug_event("screenshot_error", error=error_text)
                say("I couldn't capture the screen")
            else:
                print(f"  !! Vision step failed: {exc}")
                debug_event("vision_error", task=task, transcript=transcript, error=error_text)
                say("I couldn't figure out where to click")
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
                say("I couldn't safely identify where to click")
                return False

            x, y = resolve_click_coordinates(result, metadata)
            desc = result.get("description", f"Clicking at ({x}, {y})")
            print(f"  [eye] Vision: {desc}")
            say(desc)
            click_at(x, y)
            time.sleep(VISION_CLICK_SETTLE_SECONDS)

            verification = verify_postcondition(
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
                log_command(transcript, f"vision:click:{x},{y}")
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
            say("I couldn't confirm the click worked")
            return False

        if action == "steps":
            return execute_vision_steps_action(result, metadata, transcript)

        if action == "not_found":
            desc = result.get("description", "Could not find the target")
            print(f"  [eye] Vision: {desc}")
            say(desc)
            return False

        if action == "noop":
            desc = result.get("description", "Nothing else to do")
            print(f"  [eye] Vision: {desc}")
            return True

    return False


# ===========================================================================
# LLM FALLBACK — Gemini interprets natural language commands
# ===========================================================================

def build_command_context():
    """Generate a context string listing all registered commands for the LLM."""
    lines = []
    for cmd in COMMANDS:
        func_name = cmd["action"].__name__
        phrases = ", ".join(cmd["phrases"][:3])
        desc = cmd["description"]
        lines.append(f"- {func_name}(): triggers=[{phrases}] -- {desc}")
    return "\n".join(lines)


def build_router_system_prompt():
    """Return the strict Stage-1 router prompt."""
    context_block = get_command_context()
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
    context_block = get_command_context()
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


def call_router(transcript):
    """Call the Stage-1 router and return a validated router payload."""
    try:
        return call_gemini_structured(
            system_instruction=build_router_system_prompt(),
            user_text=transcript,
            response_json_schema=ROUTER_RESPONSE_JSON_SCHEMA,
            validator=validate_router_output,
            preferred_models=model_candidates(GEMINI_ROUTER_MODEL, GEMINI_MODEL, "gemini-2.5-flash"),
            call_type="router",
            max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS,
            timeout_seconds=ROUTER_TIMEOUT_SECONDS,
            trace_context={"transcript": transcript},
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


def call_workflow_planner(transcript, router_reason):
    """Call the Stage-2 planner and return a validated workflow payload."""
    try:
        return call_gemini_structured(
            system_instruction=build_workflow_planner_system_prompt(transcript, router_reason),
            user_text="Return the workflow object.",
            response_json_schema=WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
            validator=validate_workflow_output,
            preferred_models=model_candidates(GEMINI_PLANNER_MODEL, "gemini-2.5-pro", "gemini-2.5-flash"),
            call_type="workflow_planner",
            max_output_tokens=WORKFLOW_PLANNER_MAX_OUTPUT_TOKENS,
            timeout_seconds=WORKFLOW_PLANNER_TIMEOUT_SECONDS,
            trace_context={"transcript": transcript, "router_reason": router_reason},
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
    context_block = get_command_context()
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


def llm_interpret_command(transcript, complex=False):
    """Ask Gemini to interpret a voice command that didn't match any predefined phrase.

    When complex=True (multi-step workflows), use Pro for better planning quality.
    For simple disambiguation, use Flash for speed (~2-3x faster).
    """
    if complex:
        models = model_candidates(GEMINI_PLANNER_MODEL, GEMINI_MODEL, "gemini-2.5-flash")
    else:
        models = model_candidates("gemini-2.5-flash", GEMINI_PLANNER_MODEL, GEMINI_MODEL)
    return gemini_generate_json(
        system_instruction=build_planner_system_prompt(),
        user_text=f'The user said: "{transcript}"',
        max_output_tokens=700,
        preferred_models=models,
        trace_label="planner",
        trace_context={"transcript": transcript, "complex": complex},
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


class PlannerValidationError(ValueError):
    """Raised when router or workflow JSON fails schema or semantic validation."""


class StructuredOutputError(RuntimeError):
    """Raised when a structured Gemini response cannot be safely used."""

    def __init__(
        self,
        category,
        message,
        *,
        call_type,
        model="",
        attempt=0,
        correlation_id="",
        raw_response="",
        parse_result="not_attempted",
        validation_result="not_attempted",
        fallback_used=False,
    ):
        super().__init__(message)
        self.category = category
        self.call_type = call_type
        self.model = model
        self.attempt = attempt
        self.correlation_id = correlation_id
        self.raw_response = raw_response
        self.parse_result = parse_result
        self.validation_result = validation_result
        self.fallback_used = fallback_used


def registered_command_function_names():
    """Return all registered command function names."""
    return [cmd["action"].__name__ for cmd in COMMANDS]


def find_registered_command(func_name):
    """Return the registered command dict for a function name, if any."""
    for cmd in COMMANDS:
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


def is_valid_uri(value):
    """Return True when a string looks like a valid absolute URI."""
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def _schema_type_matches(expected_type, value):
    """Return True when a value matches a JSON-schema primitive type."""
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _validate_schema_subset(instance, schema, path="$"):
    """Validate the subset of JSON Schema features used by Biggie's planner."""
    if "oneOf" in schema:
        failures = []
        for option in schema["oneOf"]:
            try:
                _validate_schema_subset(instance, option, path)
                return
            except PlannerValidationError as exc:
                failures.append(str(exc))
        raise PlannerValidationError(f"{path}: did not match any schema option")

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        matched_type = next((expected for expected in schema_type if _schema_type_matches(expected, instance)), None)
        if matched_type is None:
            raise PlannerValidationError(f"{path}: expected one of {schema_type}")
        schema_type = matched_type
    if schema_type and not _schema_type_matches(schema_type, instance):
        raise PlannerValidationError(f"{path}: expected {schema_type}")

    if "const" in schema and instance != schema["const"]:
        raise PlannerValidationError(f'{path}: expected constant "{schema["const"]}"')
    if "enum" in schema and instance not in schema["enum"]:
        raise PlannerValidationError(f"{path}: expected one of {schema['enum']}")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in instance:
                raise PlannerValidationError(f'{path}: missing required field "{name}"')
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance.keys()) - set(properties.keys()))
            if extras:
                raise PlannerValidationError(f"{path}: unexpected field(s): {', '.join(extras)}")
        for key, value in instance.items():
            if key in properties:
                _validate_schema_subset(value, properties[key], f"{path}.{key}")
        return

    if schema_type == "array":
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(instance) < min_items:
            raise PlannerValidationError(f"{path}: expected at least {min_items} item(s)")
        if max_items is not None and len(instance) > max_items:
            raise PlannerValidationError(f"{path}: expected at most {max_items} item(s)")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                _validate_schema_subset(item, item_schema, f"{path}[{index}]")
        return

    if schema_type == "string":
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if min_length is not None and len(instance) < min_length:
            raise PlannerValidationError(f"{path}: string shorter than {min_length}")
        if max_length is not None and len(instance) > max_length:
            raise PlannerValidationError(f"{path}: string longer than {max_length}")
        if schema.get("format") == "uri" and not is_valid_uri(instance):
            raise PlannerValidationError(f"{path}: invalid uri")
        return

    if schema_type in {"number", "integer"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and instance < minimum:
            raise PlannerValidationError(f"{path}: value below minimum {minimum}")
        if maximum is not None and instance > maximum:
            raise PlannerValidationError(f"{path}: value above maximum {maximum}")


def validate_json_schema(instance, schema, stage):
    """Validate a payload against JSON Schema, with a local fallback validator."""
    try:
        import jsonschema  # type: ignore

        jsonschema.Draft202012Validator(schema).validate(instance)
    except ImportError:
        _validate_schema_subset(instance, schema)
    except Exception as exc:
        raise PlannerValidationError(f"{stage}: {exc}") from exc


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
    active_app_context = bool(_last_command.get("app"))
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


def execute_registered_command(func_name, transcript, args=None, source="llm"):
    """Execute a registered command by function name, supporting type_text args."""
    cmd = find_registered_command(func_name)
    if not cmd:
        print(f"  !! {source.upper()} referenced unknown function: {func_name}")
        say("Sorry, I couldn't find that command")
        return False

    print(f"  [robot] {source.upper()} matched command: {func_name}")
    try:
        if cmd["action"] == type_text:
            text = extract_text_argument(args)
            if not text:
                raise ValueError("type_text requires args.text")
            print(f"  [keyboard] Typing: \"{text}\"")
            type_string(text)
        else:
            cmd["action"]()
        log_command(transcript, f"{source}:{func_name}")
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"  !! Error executing {func_name}: {e}")
        debug_event(f"{source}_command_error", function=func_name, error=str(e))
        say("Sorry, something went wrong")
        return False


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
        planner_failure(
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
    steps = workflow.get("steps", [])
    description = workflow.get("description", "Working on it")
    if not isinstance(steps, list) or not steps:
        return False, {"reason": "empty_workflow", "message": "The workflow contained no steps"}
    if len(steps) > MAX_WORKFLOW_STEPS:
        return False, {"reason": "workflow_too_long", "message": f"Workflow exceeded {MAX_WORKFLOW_STEPS} steps"}

    print(f"  [robot] Workflow: {description}")
    if description:
        say(description)

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
                say(step.get("text", ""))
                ok = True
            elif step_type == "open_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("open_app requires app")
                mac_open_app(app_name, fallback_url=step.get("fallback_url"))
                focus_app(app_name)
                ok = True
            elif step_type == "focus_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("focus_app requires app")
                focus_app(app_name)
                ok = True
            elif step_type == "quit_app":
                app_name = step.get("app", "").strip()
                if not app_name:
                    raise ValueError("quit_app requires app")
                quit_named_app(app_name)
                ok = True
            elif step_type == "open_url":
                url = step.get("url", "").strip()
                if not url:
                    raise ValueError("open_url requires url")
                mac_open(url)
                ok = True
            elif step_type == "wait":
                seconds = max(0.0, min(float(step.get("seconds", 1)), 10.0))
                time.sleep(seconds)
                ok = True
            elif step_type == "wait_for_state":
                condition_fn, condition_name = build_wait_for_state_condition(step)
                ok = wait_for_state(
                    condition_fn,
                    timeout_seconds=step.get("timeout_seconds", WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS),
                    poll_interval=step.get("poll_interval", WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL),
                    condition_name=condition_name,
                )
            elif step_type == "type_text":
                text = extract_text_argument(step.get("text"))
                if not text:
                    raise ValueError("type_text requires text")
                print(f"  [keyboard] Typing: \"{text}\"")
                type_string(text)
                ok = True
            elif step_type == "keypress":
                key = step.get("key", "").strip()
                if not key:
                    raise ValueError("keypress requires key")
                press_key(
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
                ok = execute_registered_command(func_name, transcript, step.get("args"), source="workflow")
            elif step_type == "vision":
                task = step.get("task", "").strip()
                if not task:
                    raise ValueError("vision requires task")
                ok = execute_vision_task(task, transcript)
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

        completed_steps.append({"index": index, "step": step, "result": "ok" if ok else "skipped"})

    return True, {"completed_steps": completed_steps}


def execute_workflow(workflow, transcript):
    """Run a structured multi-step workflow emitted by the planner with replans."""
    current_workflow = workflow
    for attempt in range(MAX_WORKFLOW_REPLANS + 1):
        ok, details = execute_workflow_once(current_workflow, transcript)
        if ok:
            log_command(transcript, "llm:workflow")
            return True

        failure_reason = details.get("message", details.get("reason", "unknown workflow failure"))
        if is_quota_error(failure_reason):
            retry_delay = extract_retry_delay_seconds(failure_reason)
            debug_event("workflow_quota_exhausted", attempt=attempt + 1, retry_delay=retry_delay, details=details)
            message = "Gemini quota was exceeded during the workflow"
            if retry_delay:
                message += f". Retry in about {int(round(retry_delay))} seconds"
            print(f"  !! {message}. See {AI_TRACE_LOG}")
            say(message)
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

    say("Sorry, I couldn't finish that workflow")
    return False


def execute_llm_response(response_text, transcript):
    """Parse and execute the LLM's JSON response."""
    try:
        result = extract_json_object(response_text)
    except json.JSONDecodeError:
        print(f"  !! LLM returned invalid JSON: {response_text[:100]}")
        say("Sorry, I didn't understand that")
        return False

    action = result.get("action")

    if action == "command":
        return execute_registered_command(
            result.get("function", ""),
            transcript,
            args=result.get("args"),
            source="llm",
        )

    elif action == "workflow":
        try:
            workflow = validate_workflow_output(result)
        except Exception as exc:
            planner_failure("legacy_workflow", str(exc), transcript=transcript, payload=result)
            say("Sorry, I didn't understand that")
            return False
        return execute_workflow(workflow, transcript)

    elif action == "code":
        planner_failure("legacy_code", "rejected legacy code action", transcript=transcript, payload=result)
        print("  !! Planner returned forbidden action: code")
        say("Sorry, I couldn't plan that safely")
        return False

    else:
        print(f"  [robot] LLM could not interpret: \"{transcript}\"")
        say("Sorry, I didn't understand that. Say help for a list of commands.")
        return False


# ===========================================================================
# COMMAND MATCHING — fuzzy matching so you don't have to be exact
# ===========================================================================

def match_command(transcript, min_confidence=MIN_COMMAND_CONFIDENCE):
    """
    Find the best matching command for a transcript.
    Returns (command_dict, confidence, extra_text) or (None, 0, "")
    """
    best_match = None
    best_score = 0
    extra_text = ""

    for text in build_match_candidates(transcript):
        for cmd in COMMANDS:
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


def should_defer_match_to_llm(transcript, cmd, confidence, extra=""):
    """Decide whether to defer to the LLM planner or execute a local match directly.

    Fast-path: high-confidence single-action matches execute immediately without
    an LLM round-trip, saving 1-3 seconds of latency. Multi-step and ambiguous
    requests still go through the planner for smart handling.
    """
    if not LLM_FALLBACK_ENABLED:
        return False
    normalized = strip_request_wrappers(transcript)

    # --- Always fast-path these (no LLM needed) ---
    if cmd and cmd["action"] in {show_help, stop_listening} and confidence >= 0.9 and not extra:
        return False
    if cmd and cmd["action"] == type_text and extra:
        return False
    # High-confidence single-action match: execute directly (biggest speed win)
    if cmd and confidence >= 0.85 and not looks_like_multi_step_request(transcript, extra):
        return False

    # --- Always defer these (LLM adds value) ---
    if looks_like_multi_step_request(transcript, extra):
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


_route_lock = threading.Lock()


def execute_named_app_quit_request(app_name, transcript):
    """Run a normalized named-app quit request."""
    quit_named_app(app_name)
    print(f'  [robot] Quitting app: "{app_name}"')
    say(f"Closing {app_name}")
    log_command(transcript, f"quit app:{app_name}")
    return True


def execute_click_target_request(target, transcript):
    """Run a single-step vision workflow for a named on-screen click target."""
    if not GEMINI_API_KEY:
        print("  !! Vision click requested but Gemini is not configured")
        say("Screen-aware clicking needs Gemini vision enabled")
        return False
    workflow = build_click_target_workflow(target)
    return execute_workflow(workflow, transcript)


def emit_route_latency(route_start):
    """Log end-to-end route latency in milliseconds."""
    debug_event("route_latency_ms", ms=int((time.time() - route_start) * 1000))


def execute_matched_command(cmd, transcript, confidence, extra, route_start):
    """Execute a locally matched command and handle logging/state updates."""
    print(f"  [check] Matched: \"{cmd['phrases'][0]}\" (confidence: {confidence:.0%})")

    if cmd["action"] == type_text and extra:
        print(f"  [keyboard] Typing: \"{extra}\"")
        type_string(extra)
        play_sound(COMMAND_SUCCESS_SOUND)
        update_command_state(transcript, "type_text", True)
        emit_route_latency(route_start)
        return True

    try:
        cmd["action"]()
        log_command(transcript, cmd["phrases"][0])
        play_sound(COMMAND_SUCCESS_SOUND)
        update_command_state(transcript, cmd["phrases"][0], True)
        emit_route_latency(route_start)
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"  !! Error: {e}")
        debug_event("command_error", transcript=transcript, matched=cmd["phrases"][0], error=str(e))
        say("Sorry, something went wrong")
        play_sound(COMMAND_FAIL_SOUND)
        update_command_state(transcript, cmd["phrases"][0], False)
        return False


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
    if not cmd or confidence < min_confidence:
        return False
    if cmd["action"] == type_text and extra:
        return True
    return not has_router_workflow_signal(transcript, extra, cmd)


def classify_route(transcript, cmd, confidence, extra=""):
    """Deterministic pre-router for the new two-stage planner path."""
    normalized = strip_request_wrappers(transcript)
    trailing = strip_request_wrappers(extra)
    word_count = len(normalized.split())
    has_soft_prefix = starts_with_soft_request_prefix(transcript)

    if word_count == 0:
        return Route.UNKNOWN

    if cmd and cmd["action"] == type_text and extra and not looks_like_multi_step_request(transcript, extra):
        return Route.DIRECT

    if cmd and cmd["action"] in {show_help, stop_listening} and confidence >= CONF_DIRECT_HIGH:
        return Route.DIRECT

    if any(phrase in normalized for phrase in NEGATION_PHRASES):
        return Route.ROUTER

    if looks_like_multi_step_request(transcript, extra):
        return Route.WORKFLOW

    if extract_click_target_request(transcript):
        return Route.WORKFLOW

    if extract_quit_app_request(transcript):
        return Route.WORKFLOW

    if cmd is None and is_simple_open_request(transcript):
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
    click_target = extract_click_target_request(transcript)
    if click_target:
        return f"click target: {click_target}"

    quit_app = extract_quit_app_request(transcript)
    if quit_app:
        return f"quit app: {quit_app}"

    if cmd is None and is_simple_open_request(transcript):
        app_name = extract_open_app_request(transcript)
        return f"open app: {app_name}" if app_name else "generic app launch"

    return "multi-step request"


def announce_quota_issue(context, error):
    """Speak a concise quota error message with retry guidance when available."""
    retry_delay = extract_retry_delay_seconds(error)
    message = f"Gemini quota was exceeded while {context}"
    if retry_delay:
        message += f". Retry in about {int(round(retry_delay))} seconds"
    print(f"  !! {message}. See {AI_TRACE_LOG}")
    say(message)


def route_command_two_stage(transcript):
    """Run the strict two-stage router + planner flow."""
    route_start = time.time()
    cmd, confidence, extra = match_command(transcript, min_confidence=CONF_MIN_MATCH)
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
        return execute_matched_command(cmd, transcript, confidence, extra, route_start)

    if route == Route.UNKNOWN:
        say("Sorry, I didn't understand that")
        play_sound(COMMAND_FAIL_SOUND)
        update_command_state(transcript, "unknown", False)
        return False

    if not LLM_FALLBACK_ENABLED:
        if cmd and confidence >= CONF_MIN_MATCH:
            return execute_matched_command(cmd, transcript, confidence, extra, route_start)
        print(f"  ? Didn't understand: \"{transcript}\"")
        log_transcript("unmatched_command", transcript)
        say("Sorry, I didn't understand that. Say help for a list of commands.")
        play_sound(COMMAND_FAIL_SOUND)
        update_command_state(transcript, "unmatched", False)
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
                router_result = call_router(transcript)
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
                    return execute_matched_command(cmd, transcript, confidence, extra, route_start)
                say("Sorry, something went wrong")
                play_sound(COMMAND_FAIL_SOUND)
                update_command_state(transcript, "router_error", False)
                return False

            if router_result["action"] == "unknown":
                print(f"  [robot] Router returned unknown for: \"{transcript}\"")
                say("Sorry, I didn't understand that")
                play_sound(COMMAND_FAIL_SOUND)
                update_command_state(transcript, "unknown", False)
                emit_route_latency(route_start)
                return False

            if router_result["action"] == "command":
                handled = execute_registered_command(
                    router_result["function"],
                    transcript,
                    args=router_result.get("args"),
                    source="router",
                )
                if handled:
                    play_sound(COMMAND_SUCCESS_SOUND)
                    update_command_state(transcript, f'router:{router_result["function"]}', True)
                    emit_route_latency(route_start)
                    return True
                play_sound(COMMAND_FAIL_SOUND)
                update_command_state(transcript, f'router:{router_result["function"]}', False)
                return False

            workflow_reason = router_result["reason"]

        try:
            workflow = call_workflow_planner(transcript, workflow_reason)
            debug_event("workflow_planner_response", transcript=transcript, workflow=workflow)
        except TimeoutError as exc:
            planner_failure("workflow", "workflow planner timed out", transcript=transcript, error=str(exc))
            say("I understood but couldn't plan the steps")
            play_sound(COMMAND_FAIL_SOUND)
            update_command_state(transcript, "workflow_timeout", False)
            return False
        except Exception as exc:
            debug_event("workflow_planner_error", transcript=transcript, error=str(exc))
            if is_quota_error(exc):
                announce_quota_issue("planning the steps", exc)
            elif can_fallback_to_local_match(transcript, cmd, confidence, extra, min_confidence=CONF_MIN_MATCH):
                debug_event("workflow_local_fallback", transcript=transcript, confidence=round(confidence, 3))
                return execute_matched_command(cmd, transcript, confidence, extra, route_start)
            else:
                say("Sorry, something went wrong")
            play_sound(COMMAND_FAIL_SOUND)
            update_command_state(transcript, "workflow_error", False)
            return False

        handled = execute_workflow(workflow, transcript)
        if handled:
            play_sound(COMMAND_SUCCESS_SOUND)
            update_command_state(transcript, "workflow", True)
            emit_route_latency(route_start)
            return True
        play_sound(COMMAND_FAIL_SOUND)
        update_command_state(transcript, "workflow", False)
        return False
    finally:
        _route_lock.release()


def route_command_legacy(transcript):
    """Match a transcript to a command and execute it using the legacy planner."""
    route_start = time.time()
    cmd, confidence, extra = match_command(transcript)
    quit_app_name = extract_quit_app_request(transcript) if cmd is None else ""
    click_target = extract_click_target_request(transcript) if cmd is None else ""
    is_multi_step = looks_like_multi_step_request(transcript, extra)
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
                    update_command_state(transcript, f"quit:{quit_app_name}", True, app=quit_app_name)
                    emit_route_latency(route_start)
                    return True
            except Exception as e:
                debug_event("quit_named_app_failed", transcript=transcript, app=quit_app_name, error=str(e))

        if click_target:
            handled = execute_click_target_request(click_target, transcript)
            if handled:
                play_sound(COMMAND_SUCCESS_SOUND)
                update_command_state(transcript, "vision:click_target", True)
                emit_route_latency(route_start)
                return True
            play_sound(COMMAND_FAIL_SOUND)
            update_command_state(transcript, "vision:click_target", False)
            return False

        llm_attempted = False
        if LLM_FALLBACK_ENABLED:
            if not _route_lock.acquire(blocking=False):
                print("  ... LLM already processing another command...")
                play_sound(COMMAND_FAIL_SOUND)
                return False
            try:
                print(f"  [robot] No exact match. Asking AI...")
                debug_event("llm_fallback_start", transcript=transcript)
                llm_response = llm_interpret_command(transcript, complex=is_multi_step)
                debug_event("llm_fallback_response", response=llm_response[:200])
                llm_attempted = True
                handled = execute_llm_response(llm_response, transcript)
                if handled:
                    play_sound(COMMAND_SUCCESS_SOUND)
                    update_command_state(transcript, "llm", True)
                    emit_route_latency(route_start)
                    return True
            except Exception as e:
                print(f"  !! LLM fallback error: {e}")
                debug_event("llm_fallback_error", error=str(e))
            finally:
                _route_lock.release()

        # Try generic "open X" app pattern only after the planner had its shot.
        app_name = extract_open_app_request(transcript)
        if app_name:
            try:
                mac_open_app(app_name)
                print(f"  [check] Opened app: \"{app_name}\"")
                say(f"Opening {app_name}")
                log_command(transcript, f"open app:{app_name}")
                play_sound(COMMAND_SUCCESS_SOUND)
                update_command_state(transcript, f"open:{app_name}", True, app=app_name)
                return True
            except Exception as e:
                debug_event("generic_app_open_failed", transcript=transcript,
                            app=app_name, error=str(e))

        if llm_attempted:
            play_sound(COMMAND_FAIL_SOUND)
            update_command_state(transcript, "llm", False)
            return False

        print(f"  ? Didn't understand: \"{transcript}\"")
        log_transcript("unmatched_command", transcript)
        say("Sorry, I didn't understand that. Say help for a list of commands.")
        play_sound(COMMAND_FAIL_SOUND)
        update_command_state(transcript, "unmatched", False)
        return False

    return execute_matched_command(cmd, transcript, confidence, extra, route_start)


def route_command(transcript):
    """Match a transcript to a command and execute it."""
    if APOLLO_2STAGE_PLANNER:
        return route_command_two_stage(transcript)
    return route_command_legacy(transcript)


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


# ===========================================================================
# AUDIO LISTENER — Deepgram V1/nova-3 WebSocket streaming
# ===========================================================================

class AudioListener:
    """
    Continuously listens to the microphone using Deepgram's V1 streaming API
    with nova-3. Detects the wake word "Biggie", captures the command, and
    routes it for execution.

    Flow:
      1. Deepgram delivers small is_final transcript segments in real-time
      2. We watch for the wake word in each segment
      3. After wake word, we buffer subsequent segments as the command
      4. On UtteranceEnd (1.0s of silence) or high-confidence match, we route
    """

    def __init__(self):
        self.is_listening = False
        self.is_capturing_command = False
        self.command_buffer = ""
        self.command_started_at = None
        self._dg_connection = None
        self._command_timer = None
        self._preferred_deepgram_variant = None

    def _build_deepgram_connect_variants(self):
        """Build Deepgram handshake variants from safest to richest."""
        official_minimal_kwargs = {
            "model": DEEPGRAM_MODEL,
            "language": "en-US",
            "smart_format": True,
            "interim_results": True,
            "utterance_end_ms": DEEPGRAM_UTTERANCE_END_MS,
            "endpointing": DEEPGRAM_ENDPOINTING_MS,
            "vad_events": True,
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        }
        variants = [
            {
                "name": "bare",
                "kwargs": {
                    "model": DEEPGRAM_MODEL,
                    "encoding": "linear16",
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                },
            },
            {
                "name": "official_minimal",
                "kwargs": official_minimal_kwargs,
            },
        ]

        if DEEPGRAM_KEYTERM:
            variants.append(
                {
                    "name": "official_minimal_with_keyterm",
                    "kwargs": {
                        **official_minimal_kwargs,
                        "keyterm": DEEPGRAM_KEYTERM,
                    },
                }
            )

        if DEEPGRAM_MODEL == "nova-3":
            variants.append(
                {
                    "name": "nova_2_bare",
                    "kwargs": {
                        "model": "nova-2",
                        "encoding": "linear16",
                        "sample_rate": SAMPLE_RATE,
                        "channels": 1,
                    },
                }
            )

        if not self._preferred_deepgram_variant:
            return variants

        preferred = []
        fallback = []
        for variant in variants:
            if variant["name"] == self._preferred_deepgram_variant:
                preferred.append(variant)
            else:
                fallback.append(variant)
        return preferred + fallback

    def _handle_transcript(self, text):
        """Process a final transcript segment from Deepgram."""
        text = canonicalize_text(text)
        if not text:
            return

        print(f"  [ear] Heard: \"{text}\"")
        debug_event("heard_text", text=text, capturing=self.is_capturing_command)
        log_transcript("deepgram_final", text, capturing=self.is_capturing_command)

        now = time.time()

        if self.is_capturing_command:
            detected, trailing = detect_wake_word(text)
            if detected:
                self._reset_command_state("wake_word_restart")
                self._handle_wake_phrase(trailing, now, restarted=True)
                return
            if text in CAPTURE_CANCEL_PHRASES:
                print("  [headphones] Capture cancelled")
                self._reset_command_state("cancelled")
                print(f"\n  [mic] Listening for \"{WAKE_WORD}\"...\n")
                return
            if text:
                self._append_command_text(text)
                combined = self.command_buffer

                matched_cmd, confidence, _ = match_command(combined)
                if matched_cmd and confidence >= 0.75:
                    self._dispatch_buffered_command("high_confidence_match")
            return

        wake_detected, after_wake = detect_wake_word(text)
        if not wake_detected:
            return

        print(f"  [green] Wake word detected!")
        play_sound(WAKE_SOUND)
        debug_event("wake_word_detected", text=text, trailing=after_wake)
        self._handle_wake_phrase(after_wake, now)

    def _handle_utterance_end(self):
        """Called when Deepgram detects the speaker has stopped talking (1.0s silence)."""
        if not self.is_capturing_command:
            return

        if self.command_buffer:
            self._dispatch_buffered_command("utterance_end")
        # If the buffer is empty, user said only the wake word and then paused.
        # The command timer will eventually reset if no command arrives.

    def _handle_wake_phrase(self, after_wake, now, restarted=False):
        """Start or dispatch a command after a wake word has been detected."""
        trailing = after_wake.strip()
        if len(trailing) > 3:
            matched_cmd, confidence, _ = match_command(trailing)
            if matched_cmd and confidence >= 0.75:
                self._dispatch_command(trailing, "wake_word_inline_command")
                return
            self._start_command_capture(trailing, now, "wake_word_restart" if restarted else "wake_word_partial")
            print(f"  [headphones] Partial command: \"{trailing}\"")
            return

        self._start_command_capture("", now, "wake_word_only")
        print("  [headphones] Awaiting command...")

    def _start_command_capture(self, initial_text, now, reason):
        """Start capture for exactly one candidate command buffer."""
        self._cancel_command_timer()
        self.is_capturing_command = True
        self.command_buffer = initial_text.strip()
        self.command_started_at = now
        debug_event("capture_state_start", reason=reason, buffer=self.command_buffer)
        if self.command_buffer:
            log_transcript("command_buffer", self.command_buffer, reason=reason)
        self._start_command_timer()

    def _append_command_text(self, text):
        """Append speech to the current command buffer without crossing wake-word boundaries."""
        self.command_buffer = " ".join(part for part in [self.command_buffer, text.strip()] if part).strip()
        log_transcript("command_buffer", self.command_buffer)
        debug_event("capture_state_update", buffer=self.command_buffer)

    def _dispatch_command(self, command_text, reason):
        """Route a finished command after clearing capture state first."""
        command_text = command_text.strip()
        if not command_text:
            self._reset_command_state(reason)
            return
        self._reset_command_state(f"dispatch:{reason}")
        print(f"  [speech] Command: \"{command_text}\"")
        debug_event("capture_state_dispatch", reason=reason, command=command_text)
        threading.Thread(target=route_command, args=(command_text,), daemon=True).start()
        print(f"\n  [mic] Listening for \"{WAKE_WORD}\"...\n")

    def _dispatch_buffered_command(self, reason):
        """Dispatch the current command buffer as a single candidate command."""
        self._dispatch_command(self.command_buffer, reason)

    def _start_command_timer(self):
        """Start a background timer that resets capture state after COMMAND_TIMEOUT."""
        self._cancel_command_timer()

        def timeout():
            if self.is_capturing_command:
                buffered = self.command_buffer.strip()
                if buffered:
                    print(f"  [timer] Command timeout, routing: \"{buffered}\"")
                    self._dispatch_command(buffered, "timeout")
                else:
                    print("  [timer] No command received, going back to listening.")
                    self._reset_command_state("timeout_empty")
                    print(f"\n  [mic] Listening for \"{WAKE_WORD}\"...\n")

        self._command_timer = threading.Timer(COMMAND_TIMEOUT, timeout)
        self._command_timer.daemon = True
        self._command_timer.start()

    def _cancel_command_timer(self):
        """Cancel the command timeout timer if active."""
        if self._command_timer is not None:
            self._command_timer.cancel()
            self._command_timer = None

    def _reset_command_state(self, reason="reset"):
        """Reset command capture state."""
        previous_buffer = self.command_buffer
        self._cancel_command_timer()
        self.is_capturing_command = False
        self.command_buffer = ""
        self.command_started_at = None
        debug_event("capture_state_reset", reason=reason, buffer=previous_buffer)

    def _run_deepgram_session(self):
        """Run a single Deepgram V1/nova-3 streaming session."""
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
        import sounddevice as sd

        dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        def on_message(msg):
            message_type = str(getattr(msg, "type", type(msg).__name__))

            if message_type == "Results" and hasattr(msg, "channel"):
                alternatives = getattr(getattr(msg, "channel", None), "alternatives", []) or []
                transcript = ""
                if alternatives:
                    transcript = getattr(alternatives[0], "transcript", "").strip()
                if not transcript:
                    return
                if getattr(msg, "is_final", False):
                    self._handle_transcript(transcript)
                elif DEBUG_AUDIO:
                    print(f"  [interim] \"{transcript}\"")
                return

            if message_type == "UtteranceEnd":
                self._handle_utterance_end()
                return

        def on_error(error):
            print(f"  !! Deepgram error: {error}")
            debug_event("deepgram_error", error=str(error))

        connect_variants = self._build_deepgram_connect_variants()

        last_connect_error = None
        for variant_index, variant in enumerate(connect_variants, start=1):
            connect_kwargs = variant["kwargs"]
            connected_model = connect_kwargs.get("model", DEEPGRAM_MODEL)
            try:
                print(f"  [net] Deepgram connect attempt {variant_index}: {variant['name']}")
                debug_event("deepgram_connect_attempt", variant=variant_index, name=variant["name"], kwargs=connect_kwargs)
                with dg_client.listen.v1.connect(**connect_kwargs) as dg_socket:
                    self._dg_connection = dg_socket
                    dg_socket.on(EventType.MESSAGE, on_message)
                    dg_socket.on(EventType.ERROR, on_error)

                    self._preferred_deepgram_variant = variant["name"]
                    print(f"  [check] Connected to Deepgram ({connected_model}, {variant['name']})")
                    debug_event(
                        "deepgram_connected",
                        requested_model=DEEPGRAM_MODEL,
                        connected_model=connected_model,
                        variant=variant_index,
                        name=variant["name"],
                    )

                    # Set up audio streaming
                    input_device_index, _ = get_input_device()
                    audio_queue = queue.Queue(maxsize=32)
                    session_active = threading.Event()
                    session_active.set()

                    def audio_callback(indata, frames, time_info, status):
                        if status and DEBUG_AUDIO:
                            print(f"  !! Audio status: {status}")
                        chunk = indata.copy().tobytes()
                        try:
                            audio_queue.put_nowait(chunk)
                        except queue.Full:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                audio_queue.put_nowait(chunk)
                            except queue.Full:
                                pass

                    def send_audio():
                        keepalive_at = time.time()
                        while self.is_listening and session_active.is_set():
                            try:
                                data = audio_queue.get(timeout=0.25)
                            except queue.Empty:
                                data = None
                            if data is not None:
                                try:
                                    dg_socket.send_media(data)
                                except Exception as exc:
                                    debug_event("deepgram_send_error", error=str(exc))
                                    session_active.clear()
                                    break
                            if time.time() - keepalive_at >= 10:
                                try:
                                    dg_socket.send_keep_alive()
                                except Exception:
                                    pass
                                keepalive_at = time.time()

                    sender_thread = threading.Thread(target=send_audio, daemon=True)
                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="int16",
                        blocksize=int(SAMPLE_RATE * 0.1),
                        device=input_device_index,
                        callback=audio_callback,
                    )

                    with stream:
                        sender_thread.start()
                        print(f"\n  [mic] Listening for \"{WAKE_WORD}\"...\n")
                        dg_socket.start_listening()

                    session_active.clear()
                    sender_thread.join(timeout=0.5)

                self._dg_connection = None
                debug_event("deepgram_disconnected", variant=variant_index, name=variant["name"])
                return
            except Exception as exc:
                detailed_error = describe_websocket_error(exc)
                last_connect_error = RuntimeError(detailed_error)
                self._dg_connection = None
                print(f"  !! Deepgram variant {variant['name']} failed: {detailed_error}")
                debug_event(
                    "deepgram_connect_variant_failed",
                    variant=variant_index,
                    name=variant["name"],
                    kwargs=connect_kwargs,
                    error=detailed_error,
                )

        raise last_connect_error or RuntimeError("Deepgram connect failed")

    def start(self):
        """Main listening loop with auto-reconnection."""
        self.is_listening = True
        consecutive_init_failures = 0
        while self.is_listening:
            try:
                self._run_deepgram_session()
                consecutive_init_failures = 0
                if self.is_listening:
                    debug_event("deepgram_session_ended")
                    print(f"  [refresh] Session ended. Reconnecting in {DEEPGRAM_RECONNECT_DELAY:.0f}s...")
                    time.sleep(DEEPGRAM_RECONNECT_DELAY)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  !! Deepgram disconnected: {e}")
                debug_event("deepgram_reconnect", error=str(e))
                if "server rejected WebSocket connection" in str(e):
                    consecutive_init_failures += 1
                    if consecutive_init_failures >= DEEPGRAM_MAX_INIT_FAILURES:
                        raise DeepgramUnavailableError(str(e)) from e
                else:
                    consecutive_init_failures = 0
                if self.is_listening:
                    print(f"  [refresh] Reconnecting in {DEEPGRAM_RECONNECT_DELAY:.0f}s...")
                    time.sleep(DEEPGRAM_RECONNECT_DELAY)


class DeepgramUnavailableError(RuntimeError):
    """Raised when repeated Deepgram websocket initialization fails."""


class WhisperAudioListener(AudioListener):
    """Fallback local listener using openai-whisper on short microphone chunks."""

    def __init__(self):
        super().__init__()
        self._silence_chunks = 0

    def start(self):
        import numpy as np
        import sounddevice as sd
        import whisper

        self.is_listening = True
        input_device_index, _ = get_input_device()
        if input_device_index is None:
            raise RuntimeError("No input microphone found for Whisper fallback")

        print(f"  [fallback] Using local Whisper ({WHISPER_MODEL})")
        debug_event("whisper_fallback_started", model=WHISPER_MODEL)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        chunk_frames = max(1, int(SAMPLE_RATE * WHISPER_CHUNK_SECONDS))

        print(f"\n  [mic] Listening for \"{WAKE_WORD}\" with Whisper fallback...\n")

        while self.is_listening:
            try:
                audio = sd.rec(
                    chunk_frames,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=input_device_index,
                )
                sd.wait()
            except KeyboardInterrupt:
                raise

            samples = audio[:, 0]
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            if rms < WHISPER_SILENCE_THRESHOLD:
                if self.is_capturing_command:
                    self._silence_chunks += 1
                    if self._silence_chunks >= 1:
                        self._handle_utterance_end()
                continue

            self._silence_chunks = 0
            try:
                result = whisper_model.transcribe(
                    samples,
                    language="en",
                    fp16=False,
                    temperature=0,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                )
            except Exception as e:
                debug_event("whisper_transcribe_error", error=str(e))
                continue

            text = canonicalize_text(result.get("text", ""))
            if not text:
                continue

            print(f"  [ear] Heard (Whisper): \"{text}\"")
            log_transcript("whisper_final", text, rms=round(rms, 5))
            self._handle_transcript(text)


# ===========================================================================
# MAIN
# ===========================================================================

def print_banner():
    print("""
    ============================================================
    |                                                          |
    |              BIGGIE  Voice Assistant                     |
    |                                                          |
    |   Say "Biggie" followed by a command.                    |
    |   Say "Biggie, help" to see all commands.                |
    |   Say "Biggie, stop" to quit.                            |
    |                                                          |
    |   Run with --text for silent text input mode.            |
    |   Press Ctrl+C to force quit.                            |
    |                                                          |
    ============================================================
    """)


def check_dependencies(text_mode=False):
    """Check all required packages are installed."""
    if not text_mode:
        missing = []
        try:
            import sounddevice
        except ImportError:
            missing.append("sounddevice")
        try:
            import numpy
        except ImportError:
            missing.append("numpy")
        try:
            from deepgram import DeepgramClient
        except ImportError:
            missing.append("deepgram-sdk")

        if missing:
            print("  !! Missing packages. Run this command:\n")
            print(f"     pip3 install {' '.join(missing)}")
            print()
            return False

        if not DEEPGRAM_API_KEY:
            if has_local_whisper():
                print("  [info] DEEPGRAM_API_KEY not set -- will use local Whisper fallback for voice input")
            else:
                print("  !! DEEPGRAM_API_KEY not set and local Whisper is unavailable.")
                print('     Either export DEEPGRAM_API_KEY="your-key-here" or install openai-whisper.')
                print()
                return False

        if DEEPGRAM_API_KEY and len(DEEPGRAM_API_KEY) < 20:
            if has_local_whisper():
                print("  [warn] DEEPGRAM_API_KEY looks invalid or truncated -- Whisper fallback is available")
            else:
                print("  !! DEEPGRAM_API_KEY looks invalid or truncated.")
                print("     Re-copy the key from Deepgram and export it again.")
                print()
                return False

    # Optional: check for Gemini (LLM fallback + vision)
    if GEMINI_API_KEY:
        transport = get_gemini_transport()
        if transport == "sdk":
            print(f"  [check] Gemini AI fallback enabled (SDK) planner={GEMINI_PLANNER_MODEL} vision={GEMINI_VISION_MODEL}")
        elif transport == "rest":
            print(f"  [check] Gemini AI fallback enabled (REST) planner={GEMINI_PLANNER_MODEL} vision={GEMINI_VISION_MODEL}")
        else:
            print("  !! Gemini key present but transport unavailable")
        print(f"  [log] AI trace: {AI_TRACE_LOG}")
        if SAVE_VISION_DEBUG:
            print(f"  [log] Vision debug screenshots: {VISION_DEBUG_DIR}")
    else:
        print("  [info] GEMINI_API_KEY not set -- AI fallback disabled (exact commands only)")

    if has_local_whisper():
        print(f"  [check] Local Whisper fallback available ({WHISPER_MODEL})")
    elif not text_mode:
        print("  [info] Local Whisper fallback not installed")

    return True


# ===========================================================================
# TEXT INPUT MODE — type commands when you can't speak
# ===========================================================================

def run_text_mode():
    """
    Interactive text mode: type commands directly, no wake word needed.
    Everything goes through the same matching + LLM pipeline as voice.
    """
    print("""
    ============================================================
    |                                                          |
    |              BIGGIE  Text Input Mode                     |
    |                                                          |
    |   Type commands directly -- no wake word needed.         |
    |   Examples:                                              |
    |     open chrome                                          |
    |     type hello world                                     |
    |     save                                                 |
    |     help                                                 |
    |                                                          |
    |   Type "quit" or "exit" to stop.                         |
    |                                                          |
    ============================================================
    """)

    load_custom_commands()

    print("  Checking dependencies...")
    if not check_dependencies(text_mode=True):
        return

    debug_event("text_mode_started")
    print("  Biggie text mode is ready.\n")

    while True:
        try:
            user_input = input("  biggie> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Biggie shut down. See you later!")
            debug_event("text_mode_exit", method="interrupt")
            break

        if not user_input:
            continue

        # Quick exit commands (no need to route these through the full pipeline)
        if user_input.lower() in {"quit", "exit", "q"}:
            print("  Biggie shut down. See you later!")
            debug_event("text_mode_exit", method="quit_command")
            break

        # Strip wake word if they type it out of habit
        _, after_wake = detect_wake_word(user_input)
        command_text = after_wake if after_wake else canonicalize_text(user_input)

        if not command_text:
            continue

        log_transcript("text_input", command_text)

        try:
            route_command(command_text)
        except SystemExit:
            break
        except Exception as e:
            print(f"  !! Error: {e}")
            debug_event("text_mode_error", input=command_text, error=str(e))

        print()  # Blank line between commands for readability


def main():
    # Check for --text flag
    text_mode = "--text" in sys.argv or "-t" in sys.argv

    if text_mode:
        run_text_mode()
        return

    print_banner()
    load_custom_commands()

    print("  Checking dependencies...")
    if not check_dependencies():
        return

    listener = AudioListener()

    # Quick test: can we access the mic?
    try:
        import sounddevice as sd
        input_device_index, devices = get_input_device()
        if input_device_index is None:
            print("  !! No input microphone found.")
            print("  Check System Settings -> Sound -> Input and connect/select a microphone.")
            print("  If Terminal was just granted microphone access, restart Terminal and try again.")
            debug_event("microphone_missing")
            return
        print(f"  [mic] Using mic: {devices[input_device_index]['name']}")
        print(f"  [log] Debug log: {DEBUG_LOG}")
        print(f"  [log] Transcript log: {TRANSCRIPT_LOG}")
        debug_event("microphone_ready", device=devices[input_device_index]["name"])
    except Exception as e:
        print(f"  !! Microphone error: {e}")
        print("  Make sure Terminal has microphone access in System Settings")
        debug_event("microphone_error", error=str(e))
        return

    say("Biggie is ready")

    try:
        listener.start()
    except DeepgramUnavailableError as e:
        print(f"\n  !! Deepgram unavailable after repeated handshake failures: {e}")
        debug_event("deepgram_unavailable", error=str(e))
        if has_local_whisper():
            print(f"  [fallback] Switching to local Whisper voice mode ({WHISPER_MODEL})")
            say("Deepgram is unavailable. Switching to local voice mode.")
            fallback_listener = WhisperAudioListener()
            fallback_listener.start()
        else:
            print("  !! Local Whisper fallback is not installed.")
            print("     Install it with: pip3 install openai-whisper")
            print("     Then rerun Biggie, or use --text mode.")
            say("Deepgram failed and local voice fallback is not installed")
    except KeyboardInterrupt:
        print("\n  Biggie shut down. See you later!")
        debug_event("shutdown_keyboard_interrupt")
        say("Goodbye")
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n  !! Biggie crashed: {e}")
        debug_event("listener_crash", error=str(e))


if __name__ == "__main__":
    main()
