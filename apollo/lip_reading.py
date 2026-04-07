"""
Apollo lip reading — visual VAD (Voice Activity Detection) via webcam.

Uses MediaPipe Face Mesh to track lip landmarks in real-time. Exposes a
simple ``is_speaking`` property that the AudioListener can query to decide
whether to extend capture (the user's lips are still moving even though
audio silence was detected).

This module is fully optional. If opencv-python or mediapipe are not
installed, or if no webcam is available, ``LipReader.start()`` returns
False and ``is_speaking`` always returns False.
"""

import threading
import time

from apollo.config import (
    LIP_MOVING_WINDOW_SECONDS,
    LIP_OPEN_THRESHOLD,
    LIP_READING_FPS,
    WEBCAM_DEVICE_INDEX,
)
from apollo.logging_utils import debug_event
from apollo.utils import has_mediapipe_face_mesh


# MediaPipe Face Mesh landmark indices for lip measurement
# Upper inner lip: 13, Lower inner lip: 14
# Left mouth corner: 61, Right mouth corner: 291
# Nose tip: 1 (used for face-height normalization)
# Chin: 152
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_CORNER = 61
_RIGHT_CORNER = 291
_NOSE_TIP = 1
_CHIN = 152


def _lip_open_ratio(landmarks):
    """Compute a normalized lip-opening ratio from Face Mesh landmarks.

    Returns the vertical lip opening divided by the mouth width.
    A closed mouth is ~0.0; a wide-open mouth is ~0.5+.
    """
    upper = landmarks[_UPPER_LIP]
    lower = landmarks[_LOWER_LIP]
    left = landmarks[_LEFT_CORNER]
    right = landmarks[_RIGHT_CORNER]

    vertical = ((upper.x - lower.x) ** 2 + (upper.y - lower.y) ** 2) ** 0.5
    horizontal = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


class LipReader:
    """Background webcam lip-movement detector.

    Usage::

        reader = LipReader()
        if reader.start():
            # ... later, in the audio pipeline ...
            if reader.is_speaking:
                # lips are moving — extend capture window
                ...
            reader.stop()
    """

    def __init__(self):
        self._running = False
        self._thread = None
        self._cap = None
        self._face_mesh = None
        self._lips_moving = False
        self._lip_ratio = 0.0
        self._last_movement_time = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the background webcam + Face Mesh loop.

        Returns True if started successfully, False if dependencies or
        webcam are unavailable (Apollo continues without lip reading).
        """
        try:
            import cv2  # noqa: F401 — availability check
        except ImportError as exc:
            print(f"  [warn] Lip reading deps missing ({exc}) — feature disabled")
            debug_event("lip_reader_import_error", error=str(exc))
            return False
        if not has_mediapipe_face_mesh():
            print("  [warn] Installed mediapipe package does not include Face Mesh — lip reading disabled")
            debug_event("lip_reader_import_error", error="mediapipe Face Mesh unavailable")
            return False

        import cv2

        cap = cv2.VideoCapture(WEBCAM_DEVICE_INDEX)
        if not cap.isOpened():
            print("  [warn] Cannot open webcam — lip reading disabled")
            print("         Grant camera access: System Settings → Privacy & Security → Camera → Terminal")
            debug_event("lip_reader_webcam_unavailable", device=WEBCAM_DEVICE_INDEX)
            cap.release()
            return False

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="lip-reader")
        self._thread.start()
        debug_event("lip_reader_started", fps=LIP_READING_FPS, threshold=LIP_OPEN_THRESHOLD)
        print(f"  [check] Lip reader started (webcam {WEBCAM_DEVICE_INDEX}, {LIP_READING_FPS} fps)")
        return True

    def stop(self):
        """Stop the background thread and release the webcam."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        debug_event("lip_reader_stopped")

    @property
    def is_speaking(self):
        """True if lip movement was detected within the sliding window."""
        with self._lock:
            if not self._running:
                return False
            if self._lips_moving:
                return True
            # Check if we're still inside the trailing window
            return (time.monotonic() - self._last_movement_time) < LIP_MOVING_WINDOW_SECONDS

    @property
    def lips_open_ratio(self):
        """Current normalized lip opening (0.0 = closed, ~0.5 = wide open)."""
        with self._lock:
            return self._lip_ratio

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self):
        """Grab frames, run Face Mesh, compute lip ratio."""
        import cv2
        try:
            from mediapipe.solutions import face_mesh as mp_face_mesh
        except ImportError:
            try:
                from mediapipe.python.solutions import face_mesh as mp_face_mesh
            except ImportError as exc:
                raise RuntimeError("MediaPipe Face Mesh is unavailable in the installed mediapipe package") from exc

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frame_interval = 1.0 / max(1, LIP_READING_FPS)

        try:
            while self._running:
                start = time.monotonic()

                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(frame_interval)
                    continue

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    ratio = _lip_open_ratio(landmarks)

                    with self._lock:
                        self._lip_ratio = ratio
                        if ratio > LIP_OPEN_THRESHOLD:
                            self._lips_moving = True
                            self._last_movement_time = time.monotonic()
                        else:
                            self._lips_moving = False
                else:
                    with self._lock:
                        self._lips_moving = False
                        self._lip_ratio = 0.0

                # Throttle to target FPS
                elapsed = time.monotonic() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            debug_event("lip_reader_error", error=str(exc))
            print(f"  [warn] Lip reader error: {exc}")
        finally:
            face_mesh.close()
            with self._lock:
                self._lips_moving = False
                self._lip_ratio = 0.0
