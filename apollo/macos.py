"""
Apollo macOS control functions — AppleScript, JXA, accessibility helpers.

Depends on: apollo.config, apollo.logging_utils, apollo.utils.
Uses late-binding ``import apollo`` for resolve_generic_app_name (defined
in the planner section, not yet extracted).
"""

import json
import os
import re
import subprocess
import sys
import time

from apollo.config import (
    AUDIO_FEEDBACK,
    AX_AVOID_APPS,
    AX_PREFERRED_APPS,
    AX_QUERY_MAX_CHILDREN,
    AX_QUERY_MAX_DEPTH,
    AX_QUERY_MAX_RESULTS,
    AX_QUERY_TIMEOUT_SECONDS,
    AX_ROLE_KEYWORDS,
    GENERIC_APP_ALIASES,
    LIP_SYNC_ENABLED,
    VOICE_FEEDBACK,
    WAIT_FOR_STATE_DEFAULT_POLL_INTERVAL,
    WAIT_FOR_STATE_DEFAULT_TIMEOUT_SECONDS,
    WAIT_FOR_STATE_MAX_POLL_INTERVAL,
    WAIT_FOR_STATE_MIN_POLL_INTERVAL,
)
from apollo.logging_utils import debug_event
from apollo.utils import (
    extract_first_quoted_text,
    extract_target_role_from_text,
    strip_request_wrappers,
)


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
    """Speak text aloud using macOS built-in TTS, with optional lip sync."""
    if not VOICE_FEEDBACK:
        print(f"  [speaker] {text}")
        if LIP_SYNC_ENABLED:
            try:
                from apollo.lip_sync import get_lip_sync_animator
                # Use 'say' just for lip sync animation even when voice is off
                process = subprocess.Popen(["say", "-v", "Samantha", text])
                get_lip_sync_animator().start_speaking(text, process)
            except Exception:
                pass
        return
    process = subprocess.Popen(["say", "-v", "Samantha", text])
    if LIP_SYNC_ENABLED:
        try:
            from apollo.lip_sync import get_lip_sync_animator
            get_lip_sync_animator().start_speaking(text, process)
        except Exception:
            pass


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
    import importlib as _importlib
    custom_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "custom_commands.py")
    if not os.path.exists(custom_path):
        return
    try:
        sys.modules.setdefault("apollo", sys.modules.get("apollo"))
        _importlib.import_module("custom_commands")
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


def _resolve_generic_app_name(app_name):
    """Late-binding wrapper for resolve_generic_app_name (defined in planner section)."""
    import apollo
    return apollo.resolve_generic_app_name(app_name)


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


def query_ax_element(app_name, target_label="", target_role="", max_depth=5, max_results=10):
    """Query the app accessibility tree for matching elements, bounded by depth and results."""
    resolved_app = _resolve_generic_app_name(app_name) if app_name else ""
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
    resolved_app = _resolve_generic_app_name(app_name) if app_name else ""
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
    resolved_app = _resolve_generic_app_name(app_name) if app_name else ""
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
    resolved_app = _resolve_generic_app_name(app_name) if app_name else ""
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
    resolved_app = _resolve_generic_app_name(app_name) if app_name else ""
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
