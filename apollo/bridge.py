"""
Apollo WebSocket bridge — broadcasts real-time events to the Biggie UI.

Runs a lightweight asyncio WebSocket server on a background thread.
Other Apollo modules push events via ``emit(event, **data)`` which
is a synchronous, fire-and-forget call safe to use from any thread.

Integration:
  1. ``start_server()`` — called once from main.py at startup
  2. ``install_hooks()`` — monkey-patches debug_event / say to also emit
  3. The UI connects via ws://127.0.0.1:8765 and receives JSON events
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# ---------------------------------------------------------------------------
# In-process event bus
# ---------------------------------------------------------------------------
_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None

_state = {
    "agent_state": "idle",       # idle | listening | thinking | executing
    "transcript": "",            # live command buffer
    "ptt_held": False,           # push-to-talk button state
}

_messages: list[dict] = []       # conversation history
_actions: list[dict] = []        # action/execution feed


def get_snapshot() -> dict:
    """Return the current state for newly-connected clients."""
    return {
        "type": "snapshot",
        "state": _state.copy(),
        "messages": list(_messages[-50:]),
        "actions": list(_actions[-30:]),
    }


# ---------------------------------------------------------------------------
# Emit (synchronous, any thread)
# ---------------------------------------------------------------------------
def emit(event: str, **data):
    """Push an event to all connected UI clients.  Non-blocking."""
    payload = {
        "type": event,
        "time": datetime.now().isoformat(),
        **data,
    }

    _update_internal_state(event, data)

    if _loop is None or _loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)
    except RuntimeError:
        pass


def _update_internal_state(event: str, data: dict):
    if event == "state_change":
        _state["agent_state"] = data.get("state", _state["agent_state"])
    elif event == "ptt":
        _state["ptt_held"] = data.get("held", False)
    elif event == "transcript":
        _state["transcript"] = data.get("text", "")
    elif event == "message":
        _messages.append({
            "id": str(len(_messages)),
            "role": data.get("role", "assistant"),
            "text": data.get("text", ""),
            "timestamp": datetime.now().isoformat(),
        })
    elif event == "action":
        _actions.append({
            "id": data.get("id", str(len(_actions))),
            "label": data.get("label", ""),
            "status": data.get("status", "done"),
            "detail": data.get("detail", ""),
            "timestamp": datetime.now().isoformat(),
        })
    elif event == "action_update":
        aid = data.get("id")
        for a in _actions:
            if a["id"] == aid:
                if "status" in data:
                    a["status"] = data["status"]
                if "detail" in data:
                    a["detail"] = data["detail"]
                break


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------
async def _broadcast(payload: dict):
    if not _clients:
        return
    msg = json.dumps(payload)
    dead = set()
    for ws in _clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _clients -= dead


async def _handler(ws):
    _clients.add(ws)
    try:
        await ws.send(json.dumps(get_snapshot()))
        async for raw in ws:
            try:
                msg = json.loads(raw)
                await _handle_ui_message(msg)
            except json.JSONDecodeError:
                pass
    finally:
        _clients.discard(ws)


async def _handle_ui_message(msg: dict):
    """Handle messages sent FROM the UI."""
    kind = msg.get("type")
    if kind == "command":
        text = msg.get("text", "").strip()
        if text:
            threading.Thread(
                target=_run_command_from_ui,
                args=(text,),
                daemon=True,
            ).start()


def _run_command_from_ui(text: str):
    """Execute a typed command from the UI."""
    import apollo
    from apollo.utils import canonicalize_text, detect_wake_word

    _, after_wake = detect_wake_word(text)
    command_text = after_wake if after_wake else canonicalize_text(text)
    if not command_text:
        return

    emit("message", role="user", text=text)
    emit("state_change", state="thinking")
    try:
        apollo.route_command(command_text)
    except Exception as e:
        emit("message", role="assistant", text=f"Error: {e}")
    finally:
        emit("state_change", state="idle")


# ---------------------------------------------------------------------------
# Hooks — translate existing Apollo events into bridge emissions
# ---------------------------------------------------------------------------
_original_debug_event = None
_original_say = None
_original_update_command_state = None


def install_hooks():
    """Monkey-patch core Apollo functions to emit bridge events.

    This is called once at startup.  It wraps debug_event, say, and
    update_command_state so the UI gets real-time data with zero changes
    to the rest of the codebase.
    """
    global _original_debug_event, _original_say, _original_update_command_state
    import apollo
    import apollo.logging_utils as lu
    import apollo.macos as mac

    # --- debug_event wrapper ---
    _original_debug_event = lu.debug_event

    def _hooked_debug_event(event, **fields):
        _original_debug_event(event, **fields)
        _translate_debug_event(event, fields)

    lu.debug_event = _hooked_debug_event
    apollo.debug_event = _hooked_debug_event

    # --- say wrapper ---
    _original_say = mac.say

    def _hooked_say(text):
        _original_say(text)
        emit("message", role="assistant", text=text)

    mac.say = _hooked_say
    apollo.say = _hooked_say

    # --- update_command_state wrapper ---
    _original_update_command_state = apollo.update_command_state

    def _hooked_update_command_state(transcript, action, success, app=""):
        _original_update_command_state(transcript, action, success, app)
        status = "done" if success else "failed"
        emit("action_update", id=f"cmd_{hash(transcript) % 10000}",
             status=status, detail=f"{action} — {'success' if success else 'failed'}")

    apollo.update_command_state = _hooked_update_command_state


def _translate_debug_event(event: str, fields: dict):
    """Map Apollo debug events to bridge emissions for the UI."""

    # --- Push-to-talk ---
    if event == "ptt_press":
        emit("state_change", state="listening")
        emit("ptt", held=True)

    elif event == "ptt_release":
        emit("ptt", held=False)

    # --- State transitions ---
    elif event == "wake_word_detected":
        emit("state_change", state="listening")
        emit("message", role="user", text=fields.get("trailing", "").strip() or "(wake word)")

    elif event == "capture_state_start":
        emit("state_change", state="listening")
        buf = fields.get("buffer", "")
        if buf:
            emit("transcript", text=buf)

    elif event == "capture_state_update":
        emit("transcript", text=fields.get("buffer", ""))

    elif event == "capture_state_dispatch":
        cmd = fields.get("command", "")
        emit("transcript", text=cmd)
        emit("state_change", state="thinking")
        if cmd:
            emit("message", role="user", text=cmd)

    elif event == "capture_state_reset":
        reason = fields.get("reason", "")
        if not reason.startswith("dispatch:"):
            emit("state_change", state="idle")
            emit("transcript", text="")

    # --- Routing ---
    elif event == "route_command_two_stage":
        emit("state_change", state="thinking")
        matched = fields.get("matched")
        if matched:
            emit("action", id=f"cmd_{hash(fields.get('transcript', '')) % 10000}",
                 label=f"Matched: {matched}",
                 status="running",
                 detail=f"confidence {fields.get('confidence', 0)}")

    elif event == "classify_route":
        route = fields.get("route", "")
        if route in ("router", "workflow"):
            emit("action", label=f"Routing via {route}",
                 status="running", detail=fields.get("matched", ""))

    elif event == "router_response":
        result = fields.get("result")
        if isinstance(result, dict):
            emit("action", label=f"Router: {result.get('action', '?')}",
                 status="done", detail=result.get("reason", "")[:80])

    elif event in ("router_timeout", "router_error"):
        emit("action", label="Router failed",
             status="failed", detail=fields.get("error", "")[:80])

    # --- Workflow ---
    elif event == "workflow_planner_response":
        workflow = fields.get("workflow")
        if isinstance(workflow, dict):
            desc = workflow.get("description", "Planning workflow")
            steps = workflow.get("steps", [])
            emit("action", label=desc, status="running",
                 detail=f"{len(steps)} steps")
            emit("state_change", state="executing")

    elif event == "workflow_step_start":
        step = fields.get("step", {})
        index = fields.get("index", 0)
        step_type = step.get("type", "")
        reason = fields.get("reason", step.get("reason", ""))
        emit("action", label=f"Step {index + 1}: {step_type}",
             status="running", detail=reason[:80])

    elif event == "workflow_step_error":
        emit("action", label=f"Step {fields.get('index', '?')} failed",
             status="failed", detail=fields.get("error", "")[:80])

    # --- Command execution ---
    elif event == "route_command":
        pass  # Handled by route_command_two_stage

    elif event == "route_latency_ms":
        ms = fields.get("ms", 0)
        emit("action", label="Command complete",
             status="done", detail=f"{ms}ms")
        emit("state_change", state="idle")
        emit("transcript", text="")

    elif event == "llm_fallback_start":
        emit("state_change", state="thinking")
        emit("action", label="Asking Gemini...", status="running")

    elif event == "llm_fallback_response":
        emit("action", label="Gemini responded", status="done")

    elif event == "command_error":
        emit("action", label="Command error",
             status="failed", detail=fields.get("error", "")[:80])
        emit("state_change", state="idle")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
_server_thread: threading.Thread | None = None


async def _serve():
    import websockets
    async with websockets.serve(_handler, WS_HOST, WS_PORT):
        await asyncio.Future()


def start_server():
    """Start the WebSocket server on a daemon thread.  Call once at startup."""
    global _loop, _server_thread

    if _server_thread is not None and _server_thread.is_alive():
        return

    def _run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_serve())

    _server_thread = threading.Thread(target=_run, daemon=True, name="biggie-ws")
    _server_thread.start()
    time.sleep(0.15)  # let server bind


def stop_server():
    """Gracefully stop the server."""
    global _loop
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_loop.stop)
