# Apollo (Biggie)

Apollo is a hands-free macOS voice assistant. It listens for the wake word "Biggie", matches direct commands locally, and falls back to Gemini for routing, planning, and screen-aware UI actions when needed.

## Features

- Wake word detection with fuzzy aliases such as "Biggy" and "Big E"
- Direct macOS shortcuts for apps, editing, navigation, and Spotify control
- Text mode for typed commands when voice input is not practical
- Gemini-backed router and workflow planner for multi-step requests
- AX-first UI targeting with screenshot-based vision fallback
- Custom command support through `custom_commands.py`
- Deterministic and adversarial test coverage for routing behavior

## Repository Layout

```text
apollo/
  __init__.py       Compatibility shim and public API surface
  __main__.py       `python -m apollo` entrypoint
  audio.py          Deepgram and Whisper listeners
  commands.py       Built-in command registry
  config.py         Environment variables and behavior thresholds
  gemini.py         Gemini transport and structured-output helpers
  logging_utils.py  Debug, transcript, and AI trace logging
  macos.py          AppleScript, Accessibility, and app-control helpers
  main.py           Runtime startup and text mode
  planner.py        Router and workflow planner logic
  routing.py        Matching, pre-routing, and dispatch
  types.py          Enums, schemas, and validation types
  utils.py          Text normalization and shared helpers
  vision.py         Screenshot capture and vision execution
  workflow.py       Workflow execution and replanning
biggie-ui/          Desktop UI (React + TypeScript + Tailwind + Framer Motion)
custom_commands.py  User-defined commands
docs/agents/        Maintainer and agent-facing operating docs
tests/              Reliability and adversarial test suites
setup.sh            Local setup helper
requirements*.txt   Python dependency manifests
```

## Requirements

- macOS
- Python 3.10+
- Node.js 20+ (for the desktop UI)
- Homebrew
- Microphone access
- Accessibility access for your terminal app
- `DEEPGRAM_API_KEY` for cloud speech-to-text, or optional local Whisper
- `GEMINI_API_KEY` for routing, planning, and vision features

## Quick Start

```bash
chmod +x setup.sh
./setup.sh
```

The setup script installs Homebrew dependencies, creates `.venv` if needed, and installs the Python packages from `requirements-dev.txt`.

Set your API keys:

```bash
export DEEPGRAM_API_KEY="your-deepgram-key"
export GEMINI_API_KEY="your-gemini-key"
```

Optional extras:

```bash
.venv/bin/pip install openai-whisper
.venv/bin/pip install opencv-python mediapipe
```

Grant these macOS permissions under `System Settings -> Privacy & Security`:

- `Microphone` for Terminal or your terminal app
- `Accessibility` for Terminal or your terminal app
- `Camera` only if you enable lip-reading features

## Running Apollo

```bash
# Voice mode
.venv/bin/python -m apollo

# Text mode
.venv/bin/python -m apollo --text
```

With push-to-talk (default), hold the mouse back button and speak:

```text
save
open Chrome
type hello world
click the submit button
```

Or use the wake word if PTT is disabled:

```text
Biggie, save
Biggie, open Chrome
```

## Configuration

Important environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | unset | Cloud speech-to-text key |
| `GEMINI_API_KEY` | unset | Enables Gemini router, planner, and vision |
| `APOLLO_2STAGE_PLANNER` | `0` | Enables dedicated router and planner stages |
| `APOLLO_GEMINI_MODEL` | `gemini-3-pro-preview` | General Gemini fallback model |
| `APOLLO_GEMINI_ROUTER_MODEL` | `gemini-2.5-flash` | Stage 1 router model |
| `APOLLO_GEMINI_PLANNER_MODEL` | `gemini-3-pro-preview` | Stage 2 planner model |
| `APOLLO_GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Vision model |
| `APOLLO_WHISPER_MODEL` | `tiny` | Local Whisper model |
| `APOLLO_SAVE_VISION_DEBUG` | `0` | Saves vision screenshots locally |
| `APOLLO_VERBOSE_AI` | `0` | Prints verbose AI trace output |
| `APOLLO_PTT` | `1` | Push-to-talk via mouse button (enabled by default) |
| `APOLLO_PTT_BUTTON` | `middle` | Key or button: `f9`–`f12`, `scroll_lock`, `pause`, `caps_lock`, or mouse `back`, `forward`, `middle` |
| `APOLLO_LIP_READING` | `0` | Enables webcam-based lip reading |
| `APOLLO_LIP_SYNC` | `0` | Enables lip-sync animation |

See `apollo/config.py` for the full list.

## Push-to-Talk

Push-to-talk is enabled by default on the **middle mouse button**. Hold it to talk, release to send. Wake word detection ("Biggie") still works as a fallback when PTT is not held.

To use a different key or mouse button:

```bash
export APOLLO_PTT_BUTTON=middle    # default
export APOLLO_PTT_BUTTON=f9
export APOLLO_PTT_BUTTON=f12
export APOLLO_PTT_BUTTON=back      # MX Master 3 rear thumb button
export APOLLO_PTT_BUTTON=forward   # MX Master 3 front thumb button
```

To disable PTT and use wake word only:

```bash
export APOLLO_PTT=0
```

PTT requires the `pynput` package and macOS Accessibility access for your terminal app.

## Custom Commands

Add your own commands in `custom_commands.py`:

```python
from apollo import command, say, mac_open_app

@command(phrases=["open my notes"], description="Opens Apple Notes")
def open_notes():
    mac_open_app("Notes")
    say("Opening Notes")
```

Restart Apollo after editing the file.

## Logs and Local Artifacts

Apollo can write runtime artifacts to the repo root while you are debugging:

- `apollo_debug.jsonl`
- `apollo_transcripts.jsonl`
- `apollo_ai_trace.jsonl`
- `apollo_history.json`
- `apollo_vision_debug/`

These files are intentionally ignored by Git and should stay local.

## Tests

Run the full local suite:

```bash
.venv/bin/pytest tests -q
```

Run the core reliability tests only:

```bash
.venv/bin/pytest tests/test_apollo_reliability.py -q
```

Some adversarial tests are Gemini-backed and will skip automatically if `GEMINI_API_KEY` is not set.

## Desktop UI

Biggie includes a desktop interface that connects to the running Apollo backend via WebSocket. The UI displays real-time state: voice transcripts as they stream in, assistant responses, action execution status, and the current agent state (idle, listening, thinking, executing). You can also type commands directly from the UI.

The backend starts a WebSocket bridge on `ws://127.0.0.1:8765` automatically when Apollo launches. The UI connects to it and receives live events.

```bash
# 1. Start Apollo (voice or text mode)
.venv/bin/python -m apollo          # voice mode
.venv/bin/python -m apollo --text   # text mode

# 2. In a separate terminal, start the UI
cd biggie-ui
npm install
npm run dev
# → http://localhost:5173
```

If you prefer to stay in the repo root, the same UI commands are also available there:

```bash
npm run install:ui
npm run dev
```

The `websockets` Python package is required for the bridge (included in `requirements.txt`).

Build the UI for production:

```bash
cd biggie-ui && npm run build   # output in biggie-ui/dist/
# or from the repo root:
npm run build
```

## Maintainer Docs

- `docs/agents/OPERATING_MANUAL.md`
- `CLAUDE.md`
