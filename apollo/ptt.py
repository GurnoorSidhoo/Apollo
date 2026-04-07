"""
Push-to-talk controller for Apollo.

Supports both keyboard keys and mouse buttons via pynput.
Hold the configured key/button to capture, release to dispatch.

Default: F9 key.  Set APOLLO_PTT_BUTTON to change:
  f9, f10, f11, f12, scroll_lock, pause, caps_lock  (keyboard keys)
  back, forward, middle, x1, x2                     (mouse buttons)

Wake word detection still works as a fallback when PTT is not held.
"""

from __future__ import annotations

import os
import time

from apollo.config import WAKE_SOUND
from apollo.logging_utils import debug_event

_PTT_BUTTON_NAME = os.environ.get("APOLLO_PTT_BUTTON", "middle").strip().lower()

# Keys that map to pynput Key enum attributes
_KEY_MAP = {
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
    "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
    "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
    "scroll_lock": "scroll_lock",
    "pause": "pause",
    "caps_lock": "caps_lock",
}

# Mouse button names that map to pynput Button attributes
_MOUSE_MAP = {
    "back": "x1",
    "forward": "x2",
    "x1": "x1",
    "x2": "x2",
    "middle": "middle",
}


def _is_keyboard_button(name: str) -> bool:
    return name in _KEY_MAP


class PushToTalkController:
    """
    Binds a keyboard key or mouse button to push-to-talk control.

    Press  → start capture (plays wake sound, begins buffering speech)
    Release → dispatch buffered command
    """

    def __init__(self, listener):
        self._listener = listener
        self._input_listener = None
        self._is_held = False
        self._press_time: float | None = None
        self._use_keyboard = _is_keyboard_button(_PTT_BUTTON_NAME)
        self._resolved_key = None   # pynput Key (keyboard mode)
        self._resolved_btn = None   # pynput Button (mouse mode)

    def start(self):
        """Start the PTT listener in a background thread."""
        if self._use_keyboard:
            self._start_keyboard_listener()
        else:
            self._start_mouse_listener()

    def _start_keyboard_listener(self):
        from pynput.keyboard import Key, Listener as KeyListener

        attr = _KEY_MAP.get(_PTT_BUTTON_NAME, "f9")
        self._resolved_key = getattr(Key, attr, Key.f9)
        print(f"  [ptt] Push-to-talk enabled on key {_PTT_BUTTON_NAME.upper()}")
        debug_event("ptt_started", button=_PTT_BUTTON_NAME)

        self._input_listener = KeyListener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._input_listener.daemon = True
        self._input_listener.start()

    def _start_mouse_listener(self):
        from pynput.mouse import Button, Listener as MouseListener

        attr = _MOUSE_MAP.get(_PTT_BUTTON_NAME, "x1")
        self._resolved_btn = getattr(Button, attr, Button.x1)
        print(f"  [ptt] Push-to-talk enabled on mouse {_PTT_BUTTON_NAME} button")
        debug_event("ptt_started", button=_PTT_BUTTON_NAME)

        self._input_listener = MouseListener(on_click=self._on_click)
        self._input_listener.daemon = True
        self._input_listener.start()

    def stop(self):
        if self._input_listener:
            self._input_listener.stop()
            self._input_listener = None
        debug_event("ptt_stopped")

    # --- keyboard callbacks ---

    def _on_key_press(self, key):
        if key == self._resolved_key:
            self._on_press()

    def _on_key_release(self, key):
        if key == self._resolved_key:
            self._on_release()

    # --- mouse callbacks ---

    def _on_click(self, _x, _y, button, pressed):
        if button != self._resolved_btn:
            return
        if pressed:
            self._on_press()
        else:
            self._on_release()

    # --- shared logic ---

    def _on_press(self):
        import apollo
        if self._is_held:
            return
        self._is_held = True
        self._press_time = time.time()
        debug_event("ptt_press")
        apollo.play_sound(WAKE_SOUND)
        if self._listener.is_capturing_command:
            self._listener._reset_command_state("ptt_restart")
        self._listener._start_command_capture("", time.time(), "ptt_press")
        print(f"  [ptt] Listening... (release {_PTT_BUTTON_NAME.upper()} to send)")

    def _on_release(self):
        if not self._is_held:
            return
        self._is_held = False
        hold_ms = int((time.time() - (self._press_time or time.time())) * 1000)
        debug_event("ptt_release", hold_ms=hold_ms)
        if not self._listener.is_capturing_command:
            return
        buffered = self._listener.command_buffer.strip()
        if buffered:
            print(f'  [ptt] Dispatching: "{buffered}"')
            self._listener._dispatch_command(buffered, "ptt_release")
        else:
            print("  [ptt] No speech captured, cancelled.")
            self._listener._reset_command_state("ptt_empty")

    @property
    def is_held(self) -> bool:
        return self._is_held
