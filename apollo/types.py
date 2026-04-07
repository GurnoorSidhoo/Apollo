"""
Apollo shared type definitions — TypedDicts, Enums, JSON schemas,
and validation helpers.

Depends only on apollo.config (for WAIT_FOR_STATE_CONDITIONS).
"""

from enum import Enum
from typing import List, Literal, Optional, TypedDict, Union
from urllib.parse import urlparse

from apollo.config import WAIT_FOR_STATE_CONDITIONS


# ---------------------------------------------------------------------------
# Command / Router output types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Workflow step types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vision action types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JSON Schemas (used for Gemini structured output validation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JSON Schema validation helpers
# ---------------------------------------------------------------------------

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
