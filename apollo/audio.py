"""
Audio listener classes for Apollo voice assistant.

Contains the Deepgram-based AudioListener and the local Whisper fallback.

IMPORTANT — lazy-import pattern:
Functions that tests mock via ``mock.patch.object(apollo, "X")`` are accessed
at *call time* through ``import apollo; apollo.X(...)`` so that patches on
the ``apollo`` namespace are respected.  Only constants and helpers that are
never mock-patched are imported directly from their canonical modules.
"""

import queue
import threading
import time

# ---------------------------------------------------------------------------
# Direct imports: config constants (never mock-patched)
# ---------------------------------------------------------------------------
from apollo.config import (
    CAPTURE_CANCEL_PHRASES,
    COMMAND_TIMEOUT,
    DEBUG_AUDIO,
    DEEPGRAM_API_KEY,
    DEEPGRAM_ENDPOINTING_MS,
    DEEPGRAM_KEYTERM,
    DEEPGRAM_MAX_INIT_FAILURES,
    DEEPGRAM_MODEL,
    DEEPGRAM_PREFER_RICH_VARIANTS,
    DEEPGRAM_RECONNECT_DELAY,
    DEEPGRAM_UTTERANCE_END_MS,
    LIP_READING_ENABLED,
    SAMPLE_RATE,
    WAKE_SOUND,
    WAKE_WORD,
    WHISPER_CHUNK_SECONDS,
    WHISPER_MODEL,
    WHISPER_SILENCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Direct imports: logging helpers (debug_event *is* mocked in some tests but
# the audio classes are not the target of those mocks — however, to be safe
# we route through apollo for the ones that ARE mocked on the apollo ns).
# ---------------------------------------------------------------------------
from apollo.logging_utils import log_transcript

# ---------------------------------------------------------------------------
# Direct imports: utils that are never mock-patched through ``apollo``
# ---------------------------------------------------------------------------
from apollo.macos import get_input_device
from apollo.utils import canonicalize_text, describe_websocket_error


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
        self._lip_reader = None
        self.ptt_controller = None  # set by main.py when PTT is enabled

    def _build_deepgram_connect_variants(self):
        """Build Deepgram handshake variants from richest to safest."""
        bare_variant = {
            "name": "bare",
            "kwargs": {
                "model": DEEPGRAM_MODEL,
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
            },
        }
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
        rich_variants = []

        if DEEPGRAM_KEYTERM:
            rich_variants.append(
                {
                    "name": "official_minimal_with_keyterm",
                    "kwargs": {
                        **official_minimal_kwargs,
                        "keyterm": DEEPGRAM_KEYTERM,
                    },
                }
            )

        rich_variants.append(
            {
                "name": "official_minimal",
                "kwargs": official_minimal_kwargs,
            },
        )

        variants = [*rich_variants, bare_variant] if DEEPGRAM_PREFER_RICH_VARIANTS else [bare_variant, *rich_variants]

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
        import apollo

        text = canonicalize_text(text)
        if not text:
            return

        print(f'  [ear] Heard: "{text}"')
        apollo.debug_event("heard_text", text=text, capturing=self.is_capturing_command)
        log_transcript("deepgram_final", text, capturing=self.is_capturing_command)

        now = time.time()

        # --- PTT mode: while button is held, buffer everything directly ---
        if self.ptt_controller and self.ptt_controller.is_held:
            if self.is_capturing_command:
                self._append_command_text(text)
            # If PTT is held but capture hasn't started yet (race), ignore.
            return

        # --- Standard wake-word mode ---
        if self.is_capturing_command:
            detected, trailing = apollo.detect_wake_word(text)
            if detected:
                self._reset_command_state("wake_word_restart")
                self._handle_wake_phrase(trailing, now, restarted=True)
                return
            if text in CAPTURE_CANCEL_PHRASES:
                print("  [headphones] Capture cancelled")
                self._reset_command_state("cancelled")
                print(f'\n  [mic] Listening for "{WAKE_WORD}"...\n')
                return
            if text:
                self._append_command_text(text)
                combined = self.command_buffer

                matched_cmd, confidence, _ = apollo.match_command(combined)
                if matched_cmd and confidence >= 0.75:
                    self._dispatch_buffered_command("high_confidence_match")
            return

        wake_detected, after_wake = apollo.detect_wake_word(text)
        if not wake_detected:
            return

        print(f"  [green] Wake word detected!")
        apollo.play_sound(WAKE_SOUND)
        apollo.debug_event("wake_word_detected", text=text, trailing=after_wake)
        self._handle_wake_phrase(after_wake, now)

    def _handle_utterance_end(self):
        """Called when Deepgram detects the speaker has stopped talking (silence)."""
        if not self.is_capturing_command:
            return

        # PTT mode: don't dispatch on silence, only on button release
        if self.ptt_controller and self.ptt_controller.is_held:
            return

        # If lips are still moving, defer — the user is still talking
        if self._lip_reader and self._lip_reader.is_speaking:
            import apollo
            apollo.debug_event("utterance_end_deferred_lips_moving")
            self._schedule_lip_recheck()
            return

        if self.command_buffer:
            self._dispatch_buffered_command("utterance_end")
        # If the buffer is empty, user said only the wake word and then paused.
        # The command timer will eventually reset if no command arrives.

    def _schedule_lip_recheck(self):
        """Re-check lip state after a short delay; dispatch if lips stopped."""
        def check():
            if not self.is_capturing_command:
                return
            if self._lip_reader and self._lip_reader.is_speaking:
                # Still moving — check again (COMMAND_TIMEOUT is the hard ceiling)
                self._schedule_lip_recheck()
            else:
                if self.command_buffer:
                    self._dispatch_buffered_command("utterance_end_after_lip_wait")

        timer = threading.Timer(0.3, check)
        timer.daemon = True
        timer.start()

    def _handle_wake_phrase(self, after_wake, now, restarted=False):
        """Start or dispatch a command after a wake word has been detected."""
        import apollo

        trailing = after_wake.strip()
        if len(trailing) > 3:
            matched_cmd, confidence, _ = apollo.match_command(trailing)
            if matched_cmd and confidence >= 0.75:
                self._dispatch_command(trailing, "wake_word_inline_command")
                return
            self._start_command_capture(trailing, now, "wake_word_restart" if restarted else "wake_word_partial")
            print(f'  [headphones] Partial command: "{trailing}"')
            return

        self._start_command_capture("", now, "wake_word_only")
        print("  [headphones] Awaiting command...")

    def _start_command_capture(self, initial_text, now, reason):
        """Start capture for exactly one candidate command buffer."""
        import apollo

        self._cancel_command_timer()
        self.is_capturing_command = True
        self.command_buffer = initial_text.strip()
        self.command_started_at = now
        apollo.debug_event("capture_state_start", reason=reason, buffer=self.command_buffer)
        if self.command_buffer:
            log_transcript("command_buffer", self.command_buffer, reason=reason)
        self._start_command_timer()

    def _append_command_text(self, text):
        """Append speech to the current command buffer without crossing wake-word boundaries."""
        import apollo

        self.command_buffer = " ".join(part for part in [self.command_buffer, text.strip()] if part).strip()
        log_transcript("command_buffer", self.command_buffer)
        apollo.debug_event("capture_state_update", buffer=self.command_buffer)

    def _dispatch_command(self, command_text, reason):
        """Route a finished command after clearing capture state first."""
        import apollo

        command_text = command_text.strip()
        if not command_text:
            self._reset_command_state(reason)
            return
        self._reset_command_state(f"dispatch:{reason}")
        print(f'  [speech] Command: "{command_text}"')
        apollo.debug_event("capture_state_dispatch", reason=reason, command=command_text)
        threading.Thread(target=apollo.route_command, args=(command_text,), daemon=True).start()
        print(f'\n  [mic] Listening for "{WAKE_WORD}"...\n')

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
                    print(f'  [timer] Command timeout, routing: "{buffered}"')
                    self._dispatch_command(buffered, "timeout")
                else:
                    print("  [timer] No command received, going back to listening.")
                    self._reset_command_state("timeout_empty")
                    print(f'\n  [mic] Listening for "{WAKE_WORD}"...\n')

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
        import apollo

        previous_buffer = self.command_buffer
        self._cancel_command_timer()
        self.is_capturing_command = False
        self.command_buffer = ""
        self.command_started_at = None
        apollo.debug_event("capture_state_reset", reason=reason, buffer=previous_buffer)

    def _run_deepgram_session(self):
        """Run a single Deepgram V1/nova-3 streaming session."""
        import apollo

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
                    print(f'  [interim] "{transcript}"')
                return

            if message_type == "UtteranceEnd":
                self._handle_utterance_end()
                return

        def on_error(error):
            print(f"  !! Deepgram error: {error}")
            apollo.debug_event("deepgram_error", error=str(error))

        connect_variants = self._build_deepgram_connect_variants()

        last_connect_error = None
        for variant_index, variant in enumerate(connect_variants, start=1):
            connect_kwargs = variant["kwargs"]
            connected_model = connect_kwargs.get("model", DEEPGRAM_MODEL)
            try:
                print(f"  [net] Deepgram connect attempt {variant_index}: {variant['name']}")
                apollo.debug_event("deepgram_connect_attempt", variant=variant_index, name=variant["name"], kwargs=connect_kwargs)
                with dg_client.listen.v1.connect(**connect_kwargs) as dg_socket:
                    self._dg_connection = dg_socket
                    dg_socket.on(EventType.MESSAGE, on_message)
                    dg_socket.on(EventType.ERROR, on_error)

                    self._preferred_deepgram_variant = variant["name"]
                    print(f"  [check] Connected to Deepgram ({connected_model}, {variant['name']})")
                    apollo.debug_event(
                        "deepgram_connected",
                        requested_model=DEEPGRAM_MODEL,
                        connected_model=connected_model,
                        variant=variant_index,
                        name=variant["name"],
                    )

                    # Optionally start lip reader for visual VAD
                    if LIP_READING_ENABLED and self._lip_reader is None:
                        try:
                            from apollo.lip_reading import LipReader
                            lr = LipReader()
                            if lr.start():
                                self._lip_reader = lr
                        except Exception as exc:
                            apollo.debug_event("lip_reader_init_error", error=str(exc))

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
                                    apollo.debug_event("deepgram_send_error", error=str(exc))
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
                        print(f'\n  [mic] Listening for "{WAKE_WORD}"...\n')
                        dg_socket.start_listening()

                    session_active.clear()
                    sender_thread.join(timeout=0.5)

                self._dg_connection = None
                if self._lip_reader:
                    self._lip_reader.stop()
                    self._lip_reader = None
                apollo.debug_event("deepgram_disconnected", variant=variant_index, name=variant["name"])
                return
            except Exception as exc:
                detailed_error = describe_websocket_error(exc)
                last_connect_error = RuntimeError(detailed_error)
                self._dg_connection = None
                print(f"  !! Deepgram variant {variant['name']} failed: {detailed_error}")
                apollo.debug_event(
                    "deepgram_connect_variant_failed",
                    variant=variant_index,
                    name=variant["name"],
                    kwargs=connect_kwargs,
                    error=detailed_error,
                )

        raise last_connect_error or RuntimeError("Deepgram connect failed")

    def start(self):
        """Main listening loop with auto-reconnection."""
        import apollo

        self.is_listening = True
        consecutive_init_failures = 0
        while self.is_listening:
            try:
                self._run_deepgram_session()
                consecutive_init_failures = 0
                if self.is_listening:
                    apollo.debug_event("deepgram_session_ended")
                    print(f"  [refresh] Session ended. Reconnecting in {DEEPGRAM_RECONNECT_DELAY:.0f}s...")
                    time.sleep(DEEPGRAM_RECONNECT_DELAY)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  !! Deepgram disconnected: {e}")
                apollo.debug_event("deepgram_reconnect", error=str(e))
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

        import apollo

        self.is_listening = True
        input_device_index, _ = get_input_device()
        if input_device_index is None:
            raise RuntimeError("No input microphone found for Whisper fallback")

        print(f"  [fallback] Using local Whisper ({WHISPER_MODEL})")
        apollo.debug_event("whisper_fallback_started", model=WHISPER_MODEL)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        chunk_frames = max(1, int(SAMPLE_RATE * WHISPER_CHUNK_SECONDS))

        # Optionally start lip reader
        if LIP_READING_ENABLED and self._lip_reader is None:
            try:
                from apollo.lip_reading import LipReader
                lr = LipReader()
                if lr.start():
                    self._lip_reader = lr
            except Exception:
                pass

        print(f'\n  [mic] Listening for "{WAKE_WORD}" with Whisper fallback...\n')

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
                    # If lips are still moving, don't count as silence
                    if self._lip_reader and self._lip_reader.is_speaking:
                        self._silence_chunks = 0
                        continue
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
                apollo.debug_event("whisper_transcribe_error", error=str(e))
                continue

            text = canonicalize_text(result.get("text", ""))
            if not text:
                continue

            print(f'  [ear] Heard (Whisper): "{text}"')
            log_transcript("whisper_final", text, rms=round(rms, 5))
            self._handle_transcript(text)
