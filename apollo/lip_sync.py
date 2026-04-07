"""
Apollo lip sync — animated mouth overlay during TTS speech.

Displays a small floating window with mouth-shape animations that sync
with the macOS ``say`` command. Uses a simplified phoneme-to-viseme
mapping to drive the animation, and polls the TTS subprocess to know
when speech finishes.

Requires: pyobjc-framework-Cocoa (usually pre-installed on macOS with
Homebrew Python). Falls back gracefully if unavailable.
"""

import subprocess
import threading
import time

from apollo.config import LIP_SYNC_POSITION, LIP_SYNC_WINDOW_SIZE
from apollo.logging_utils import debug_event


# ---------------------------------------------------------------------------
# Viseme mapping: simplified character-group -> mouth shape
# ---------------------------------------------------------------------------
# Each viseme is a tuple: (mouth_width_ratio, mouth_height_ratio)
# relative to the window.  1.0 = full window dimension.
_VISEMES = {
    "closed":   (0.50, 0.05),   # m, b, p — lips together
    "slight":   (0.45, 0.15),   # s, t, d, n, l, z — slight opening
    "open":     (0.50, 0.35),   # a, e, i — open mouth
    "wide":     (0.60, 0.40),   # ah, eh — wide open
    "rounded":  (0.30, 0.30),   # o, u, w — rounded lips
    "teeth":    (0.45, 0.10),   # f, v — upper teeth on lower lip
    "rest":     (0.40, 0.02),   # silence / neutral
}

_CHAR_TO_VISEME = {}
for _ch in "mbp":
    _CHAR_TO_VISEME[_ch] = "closed"
for _ch in "stdnlzc":
    _CHAR_TO_VISEME[_ch] = "slight"
for _ch in "aei":
    _CHAR_TO_VISEME[_ch] = "open"
for _ch in "hk":
    _CHAR_TO_VISEME[_ch] = "wide"
for _ch in "ouwq":
    _CHAR_TO_VISEME[_ch] = "rounded"
for _ch in "fv":
    _CHAR_TO_VISEME[_ch] = "teeth"
for _ch in "rgxyjhj":
    _CHAR_TO_VISEME[_ch] = "slight"


def _text_to_viseme_sequence(text):
    """Convert text into a list of viseme names, one per character-group."""
    visemes = []
    for ch in text.lower():
        if ch.isalpha():
            visemes.append(_CHAR_TO_VISEME.get(ch, "slight"))
        elif ch == " ":
            visemes.append("rest")
        # skip punctuation
    if not visemes:
        visemes = ["rest"]
    return visemes


def _estimate_speech_duration(text):
    """Estimate how long macOS 'say' takes for this text (seconds).

    Samantha speaks at roughly 180 words per minute.
    """
    words = len(text.split())
    return max(0.5, words / 3.0)


# ---------------------------------------------------------------------------
# Cocoa-based floating window animator
# ---------------------------------------------------------------------------

def _try_cocoa_animator():
    """Attempt to create a Cocoa-based lip sync animator. Returns class or None."""
    try:
        import AppKit
        import Foundation
        return _build_cocoa_animator(AppKit, Foundation)
    except ImportError:
        return None


def _build_cocoa_animator(AppKit, Foundation):
    """Build and return the CocoaLipSyncAnimator class."""

    class CocoaLipSyncAnimator:
        """Lip sync using a native macOS overlay window."""

        def __init__(self):
            self._window = None
            self._mouth_view = None
            self._is_animating = False
            self._lock = threading.Lock()
            self._app_started = False

        def start_speaking(self, text, process):
            """Begin mouth animation for the given text and say process."""
            thread = threading.Thread(
                target=self._animate,
                args=(text, process),
                daemon=True,
                name="lip-sync",
            )
            thread.start()

        def _ensure_app(self):
            """Ensure NSApplication is initialized (for window creation)."""
            if not self._app_started:
                app = AppKit.NSApplication.sharedApplication()
                app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
                self._app_started = True

        def _create_window(self):
            """Create a small borderless floating window."""
            size = LIP_SYNC_WINDOW_SIZE
            screen = AppKit.NSScreen.mainScreen()
            screen_frame = screen.frame()

            # Position the window
            pos = LIP_SYNC_POSITION.lower().replace("-", "")
            if "topleft" in pos:
                x, y = 20, screen_frame.size.height - size - 60
            elif "topright" in pos:
                x, y = screen_frame.size.width - size - 20, screen_frame.size.height - size - 60
            elif "bottomleft" in pos:
                x, y = 20, 60
            else:  # bottom-right default
                x, y = screen_frame.size.width - size - 20, 60

            rect = Foundation.NSMakeRect(x, y, size, size)
            style = (
                AppKit.NSWindowStyleMaskBorderless
            )
            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style, AppKit.NSBackingStoreBuffered, False,
            )
            window.setLevel_(AppKit.NSFloatingWindowLevel)
            window.setOpaque_(False)
            window.setBackgroundColor_(AppKit.NSColor.clearColor())
            window.setIgnoresMouseEvents_(True)
            window.setHasShadow_(False)

            # Create the content view for drawing
            view = _MouthView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, size, size))
            window.setContentView_(view)

            self._window = window
            self._mouth_view = view

        def _show_window(self):
            if self._window:
                self._window.orderFront_(None)

        def _hide_window(self):
            if self._window:
                self._window.orderOut_(None)

        def _update_mouth(self, viseme_name):
            """Update the mouth shape on the overlay."""
            if self._mouth_view:
                w_ratio, h_ratio = _VISEMES.get(viseme_name, _VISEMES["rest"])
                self._mouth_view.setMouthShape_(w_ratio, h_ratio)
                self._mouth_view.setNeedsDisplay_(True)

        def _animate(self, text, process):
            """Run animation loop until the say process completes."""
            with self._lock:
                if self._is_animating:
                    return
                self._is_animating = True

            try:
                # All Cocoa calls must happen; we'll use performSelectorOnMainThread
                # but since we're a background daemon, we do it directly with GIL
                self._ensure_app()
                self._create_window()
                self._show_window()

                visemes = _text_to_viseme_sequence(text)
                duration = _estimate_speech_duration(text)
                time_per_viseme = duration / len(visemes)
                # Clamp to reasonable range
                time_per_viseme = max(0.04, min(0.2, time_per_viseme))

                debug_event("lip_sync_start", text=text[:50], viseme_count=len(visemes))

                idx = 0
                while process.poll() is None:
                    viseme = visemes[idx % len(visemes)]
                    self._update_mouth(viseme)
                    idx += 1
                    time.sleep(time_per_viseme)

                # Finish: close mouth and hide
                self._update_mouth("rest")
                time.sleep(0.15)
                self._hide_window()
                debug_event("lip_sync_end")

            except Exception as exc:
                debug_event("lip_sync_error", error=str(exc))
            finally:
                with self._lock:
                    self._is_animating = False

    # NSView subclass for drawing the mouth
    class _MouthView(AppKit.NSView):
        _w_ratio = 0.40
        _h_ratio = 0.02

        def setMouthShape_(self, w_ratio, h_ratio):
            self._w_ratio = w_ratio
            self._h_ratio = h_ratio

        def drawRect_(self, rect):
            # Clear
            AppKit.NSColor.clearColor().set()
            AppKit.NSRectFill(rect)

            size = rect.size.width
            cx = size / 2
            cy = size / 2

            # Draw face circle (subtle)
            face_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.3, 0.3, 0.35, 0.7,
            )
            face_color.set()
            face_rect = Foundation.NSMakeRect(size * 0.1, size * 0.1, size * 0.8, size * 0.8)
            face = AppKit.NSBezierPath.bezierPathWithOvalInRect_(face_rect)
            face.fill()

            # Draw eyes
            eye_color = AppKit.NSColor.whiteColor()
            eye_color.set()
            eye_size = size * 0.1
            left_eye = Foundation.NSMakeRect(cx - size * 0.18 - eye_size / 2, cy + size * 0.12, eye_size, eye_size)
            right_eye = Foundation.NSMakeRect(cx + size * 0.18 - eye_size / 2, cy + size * 0.12, eye_size, eye_size)
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(left_eye).fill()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(right_eye).fill()

            # Draw pupils
            pupil_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.1, 0.1, 0.15, 1.0,
            )
            pupil_color.set()
            pupil_size = eye_size * 0.5
            left_pupil = Foundation.NSMakeRect(
                cx - size * 0.18 - pupil_size / 2, cy + size * 0.12 + (eye_size - pupil_size) / 2,
                pupil_size, pupil_size,
            )
            right_pupil = Foundation.NSMakeRect(
                cx + size * 0.18 - pupil_size / 2, cy + size * 0.12 + (eye_size - pupil_size) / 2,
                pupil_size, pupil_size,
            )
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(left_pupil).fill()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(right_pupil).fill()

            # Draw mouth
            mouth_w = size * self._w_ratio
            mouth_h = max(2, size * self._h_ratio)
            mouth_rect = Foundation.NSMakeRect(
                cx - mouth_w / 2,
                cy - size * 0.15 - mouth_h / 2,
                mouth_w,
                mouth_h,
            )

            # Mouth interior (dark)
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.15, 0.05, 0.05, 1.0,
            ).set()
            mouth = AppKit.NSBezierPath.bezierPathWithOvalInRect_(mouth_rect)
            mouth.fill()

            # Lip outline
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.8, 0.35, 0.35, 1.0,
            ).set()
            mouth.setLineWidth_(2.0)
            mouth.stroke()

    return CocoaLipSyncAnimator


# ---------------------------------------------------------------------------
# Terminal fallback: ASCII mouth in the console
# ---------------------------------------------------------------------------

class TerminalLipSyncAnimator:
    """Fallback lip sync that prints ASCII mouth shapes to the terminal."""

    _MOUTHS = {
        "closed":  "  ( --- )",
        "slight":  "  ( -o- )",
        "open":    "  ( =O= )",
        "wide":    "  ( =OO= )",
        "rounded": "  (  O  )",
        "teeth":   "  ( =E= )",
        "rest":    "  ( --- )",
    }

    def __init__(self):
        self._is_animating = False
        self._lock = threading.Lock()

    def start_speaking(self, text, process):
        thread = threading.Thread(
            target=self._animate, args=(text, process), daemon=True, name="lip-sync-term",
        )
        thread.start()

    def _animate(self, text, process):
        with self._lock:
            if self._is_animating:
                return
            self._is_animating = True

        try:
            visemes = _text_to_viseme_sequence(text)
            duration = _estimate_speech_duration(text)
            time_per_viseme = max(0.06, min(0.2, duration / len(visemes)))

            idx = 0
            while process.poll() is None:
                viseme = visemes[idx % len(visemes)]
                mouth = self._MOUTHS.get(viseme, self._MOUTHS["rest"])
                # Overwrite the same line
                print(f"\r  [face] {mouth}", end="", flush=True)
                idx += 1
                time.sleep(time_per_viseme)

            print(f"\r  [face] {self._MOUTHS['rest']}", flush=True)

        except Exception as exc:
            debug_event("lip_sync_terminal_error", error=str(exc))
        finally:
            with self._lock:
                self._is_animating = False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_animator_instance = None
_animator_lock = threading.Lock()


def get_lip_sync_animator():
    """Return the singleton lip sync animator (Cocoa or terminal fallback)."""
    global _animator_instance
    with _animator_lock:
        if _animator_instance is not None:
            return _animator_instance

        cocoa_cls = _try_cocoa_animator()
        if cocoa_cls is not None:
            _animator_instance = cocoa_cls()
            debug_event("lip_sync_backend", backend="cocoa")
            print("  [check] Lip sync enabled (Cocoa overlay)")
        else:
            _animator_instance = TerminalLipSyncAnimator()
            debug_event("lip_sync_backend", backend="terminal")
            print("  [check] Lip sync enabled (terminal fallback)")

        return _animator_instance
