"""
Apollo main entrypoint — banner, dependency checks, text mode, and main().

Uses ``import apollo`` at call time for all cross-module references
so that mock.patch.object(apollo, "X") works correctly in tests.
"""

import sys

from apollo.config import (
    AI_TRACE_LOG,
    DEBUG_LOG,
    DEEPGRAM_API_KEY,
    GEMINI_API_KEY,
    GEMINI_PLANNER_MODEL,
    GEMINI_VISION_MODEL,
    LIP_READING_ENABLED,
    LIP_SYNC_ENABLED,
    SAVE_VISION_DEBUG,
    TRANSCRIPT_LOG,
    VISION_DEBUG_DIR,
    WEBCAM_DEVICE_INDEX,
    WHISPER_MODEL,
)
from apollo.logging_utils import debug_event
from apollo.utils import has_lip_reading_deps, has_local_whisper, has_mediapipe_face_mesh


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
    import apollo
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
        transport = apollo.get_gemini_transport()
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

    # Lip reading / lip sync (optional)
    if LIP_READING_ENABLED or LIP_SYNC_ENABLED:
        if has_lip_reading_deps():
            if LIP_READING_ENABLED:
                print(f"  [check] Lip reading enabled (webcam {WEBCAM_DEVICE_INDEX})")
            if LIP_SYNC_ENABLED:
                print("  [check] Lip sync animation enabled")
        else:
            missing_lip = []
            try:
                import cv2  # noqa: F401
            except ImportError:
                missing_lip.append("opencv-python")
            try:
                import mediapipe  # noqa: F401
            except ImportError:
                missing_lip.append("mediapipe")
            else:
                if not has_mediapipe_face_mesh():
                    missing_lip.append("mediapipe-face-mesh support")
            if missing_lip:
                print(f"  [warn] Lip features requested but missing: {', '.join(missing_lip)}")
                print("         Install a mediapipe build that still ships Face Mesh, or disable APOLLO_LIP_READING.")
    else:
        print("  [info] Lip reading/sync disabled (set APOLLO_LIP_READING=1 or APOLLO_LIP_SYNC=1)")

    return True


def run_text_mode():
    """
    Interactive text mode: type commands directly, no wake word needed.
    Everything goes through the same matching + LLM pipeline as voice.
    """
    import apollo
    from apollo.logging_utils import log_transcript
    from apollo.utils import canonicalize_text, detect_wake_word

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

    apollo.load_custom_commands()

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
            apollo.route_command(command_text)
        except SystemExit:
            break
        except Exception as e:
            print(f"  !! Error: {e}")
            debug_event("text_mode_error", input=command_text, error=str(e))

        print()  # Blank line between commands for readability


def main():
    import apollo

    # Check for --text flag
    text_mode = "--text" in sys.argv or "-t" in sys.argv

    if text_mode:
        run_text_mode()
        return

    print_banner()
    apollo.load_custom_commands()

    print("  Checking dependencies...")
    if not check_dependencies():
        return

    listener = apollo.AudioListener()

    # Quick test: can we access the mic?
    try:
        import sounddevice as sd
        input_device_index, devices = apollo.get_input_device()
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

    apollo.say("Biggie is ready")

    try:
        listener.start()
    except apollo.DeepgramUnavailableError as e:
        print(f"\n  !! Deepgram unavailable after repeated handshake failures: {e}")
        debug_event("deepgram_unavailable", error=str(e))
        if has_local_whisper():
            print(f"  [fallback] Switching to local Whisper voice mode ({WHISPER_MODEL})")
            apollo.say("Deepgram is unavailable. Switching to local voice mode.")
            fallback_listener = apollo.WhisperAudioListener()
            fallback_listener.start()
        else:
            print("  !! Local Whisper fallback is not installed.")
            print("     Install it with: pip3 install openai-whisper")
            print("     Then rerun Biggie, or use --text mode.")
            apollo.say("Deepgram failed and local voice fallback is not installed")
    except KeyboardInterrupt:
        print("\n  Biggie shut down. See you later!")
        debug_event("shutdown_keyboard_interrupt")
        apollo.say("Goodbye")
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n  !! Biggie crashed: {e}")
        debug_event("listener_crash", error=str(e))
