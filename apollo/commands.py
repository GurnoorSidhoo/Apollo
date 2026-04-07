"""
Apollo built-in voice commands.

This module is imported at the END of apollo/__init__.py (after all helper
functions are defined) to avoid circular imports. Each command uses the
``command`` decorator and helper functions from the parent package.
"""

import apollo


# --- App Launching ---

@apollo.command(
    phrases=["open claude", "launch claude", "start claude", "go to claude",
             "open cloud", "launch cloud"],
    description="Opens the Claude Mac app"
)
def open_claude():
    apollo.mac_open_app("Claude", fallback_url="https://claude.ai")
    apollo.say("Opening Claude")

@apollo.command(
    phrases=["open terminal", "launch terminal", "start terminal"],
    description="Opens the Terminal app"
)
def open_terminal():
    apollo.mac_open_app("Terminal")
    apollo.say("Opening Terminal")

@apollo.command(
    phrases=["open vs code", "open vscode", "launch vs code", "open code",
             "launch code", "start vs code"],
    description="Opens Visual Studio Code"
)
def open_vscode():
    apollo.mac_open_app("Visual Studio Code")
    apollo.say("Opening VS Code")

@apollo.command(
    phrases=["open safari", "launch safari"],
    description="Opens Safari"
)
def open_safari():
    apollo.mac_open_app("Safari")
    apollo.say("Opening Safari")

@apollo.command(
    phrases=["open chrome", "launch chrome", "open google chrome",
             "open browser", "launch browser", "open my browser"],
    description="Opens Google Chrome"
)
def open_chrome():
    apollo.mac_open_app("Google Chrome")
    apollo.say("Opening Chrome")

@apollo.command(
    phrases=["open finder", "launch finder", "open files"],
    description="Opens Finder"
)
def open_finder():
    apollo.mac_open_app("Finder")
    apollo.say("Opening Finder")

@apollo.command(
    phrases=["open spotify", "launch spotify", "go to spotify"],
    description="Opens Spotify"
)
def open_spotify():
    apollo.mac_open_app("Spotify")
    apollo.say("Opening Spotify")


@apollo.command(
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
    apollo.spotify_command("play")
    apollo.say("Playing Spotify")


@apollo.command(
    phrases=["pause spotify", "pause music", "stop spotify", "pause the music"],
    description="Pauses Spotify playback"
)
def pause_spotify():
    apollo.spotify_command("pause")
    apollo.say("Pausing Spotify")


@apollo.command(
    phrases=["next song", "next track", "skip song", "skip track"],
    description="Skips to the next Spotify track"
)
def next_spotify_track():
    apollo.spotify_command("next track")
    apollo.say("Skipping track")


@apollo.command(
    phrases=["previous song", "previous track", "last song", "go back song"],
    description="Returns to the previous Spotify track"
)
def previous_spotify_track():
    apollo.spotify_command("previous track")
    apollo.say("Going back a track")

@apollo.command(
    phrases=["open github", "go to github", "launch github"],
    description="Opens GitHub in browser"
)
def open_github():
    apollo.mac_open("https://github.com")
    apollo.say("Opening GitHub")


# --- Window Management ---

@apollo.command(
    phrases=["close window", "close this", "close it"],
    description="Closes the current window (Cmd+W)"
)
def close_window():
    apollo.hotkey("w", command=True)
    apollo.say("Closing window")

@apollo.command(
    phrases=["close app", "quit app", "quit this", "force quit"],
    description="Quits the current app (Cmd+Q)"
)
def quit_app():
    apollo.hotkey("q", command=True)
    apollo.say("Quitting app")

@apollo.command(
    phrases=["scroll down", "page down"],
    description="Scrolls the current view down"
)
def scroll_down():
    apollo.press_key("page down")
    apollo.say("Scrolling down")

@apollo.command(
    phrases=["scroll up", "page up"],
    description="Scrolls the current view up"
)
def scroll_up():
    apollo.press_key("page up")
    apollo.say("Scrolling up")

@apollo.command(
    phrases=["new tab", "open new tab"],
    description="Opens a new tab (Cmd+T)"
)
def new_tab():
    apollo.hotkey("t", command=True)
    apollo.say("New tab")

@apollo.command(
    phrases=["next tab", "switch tab", "go to next tab"],
    description="Switches to next tab"
)
def next_tab():
    apollo.hotkey("}", command=True, shift=True)

@apollo.command(
    phrases=["previous tab", "go back tab", "last tab"],
    description="Switches to previous tab"
)
def prev_tab():
    apollo.hotkey("{", command=True, shift=True)

@apollo.command(
    phrases=["minimise", "minimize", "minimise window", "hide window"],
    description="Minimises current window"
)
def minimise():
    apollo.hotkey("m", command=True)
    apollo.say("Minimised")

@apollo.command(
    phrases=["full screen", "fullscreen", "go full screen", "maximise", "maximize"],
    description="Toggles fullscreen"
)
def fullscreen():
    apollo.hotkey("f", command=True, ctrl=True)


# --- Coding Shortcuts ---

@apollo.command(
    phrases=["save", "save file", "save this"],
    description="Saves current file (Cmd+S)"
)
def save_file():
    apollo.hotkey("s", command=True)
    apollo.say("Saved")

@apollo.command(
    phrases=["undo", "undo that"],
    description="Undo (Cmd+Z)"
)
def undo():
    apollo.hotkey("z", command=True)

@apollo.command(
    phrases=["redo", "redo that"],
    description="Redo (Cmd+Shift+Z)"
)
def redo():
    apollo.hotkey("z", command=True, shift=True)

@apollo.command(
    phrases=["copy", "copy that", "copy this"],
    description="Copy selection (Cmd+C)"
)
def copy():
    apollo.hotkey("c", command=True)

@apollo.command(
    phrases=["paste", "paste that", "paste it"],
    description="Paste (Cmd+V)"
)
def paste():
    apollo.hotkey("v", command=True)

@apollo.command(
    phrases=["cut", "cut that", "cut this"],
    description="Cut selection (Cmd+X)"
)
def cut():
    apollo.hotkey("x", command=True)

@apollo.command(
    phrases=["select all", "select everything"],
    description="Select all (Cmd+A)"
)
def select_all():
    apollo.hotkey("a", command=True)

@apollo.command(
    phrases=["find", "search", "find in file", "search file"],
    description="Find (Cmd+F)"
)
def find():
    apollo.hotkey("f", command=True)
    apollo.say("Find activated")

@apollo.command(
    phrases=["comment", "comment line", "toggle comment", "comment out",
             "comment this"],
    description="Toggle comment in code editor (Cmd+/)"
)
def comment_line():
    apollo.hotkey("/", command=True)

@apollo.command(
    phrases=["go to line", "jump to line"],
    description="Go to line in VS Code (Ctrl+G)"
)
def goto_line():
    apollo.hotkey("g", ctrl=True)
    apollo.say("Go to line")

@apollo.command(
    phrases=["open command palette", "command palette", "palette"],
    description="Opens VS Code command palette (Cmd+Shift+P)"
)
def command_palette():
    apollo.hotkey("p", command=True, shift=True)
    apollo.say("Command palette")

@apollo.command(
    phrases=["open file", "quick open"],
    description="Quick open file in VS Code (Cmd+P)"
)
def quick_open():
    apollo.hotkey("p", command=True)

@apollo.command(
    phrases=["toggle sidebar", "hide sidebar", "show sidebar"],
    description="Toggles VS Code sidebar (Cmd+B)"
)
def toggle_sidebar():
    apollo.hotkey("b", command=True)

@apollo.command(
    phrases=["toggle terminal", "show terminal", "hide terminal",
             "open terminal panel"],
    description="Toggles integrated terminal in VS Code (Ctrl+`)"
)
def toggle_terminal():
    apollo.hotkey("`", ctrl=True)

@apollo.command(
    phrases=["split editor", "split screen", "split view"],
    description="Splits the editor in VS Code"
)
def split_editor():
    apollo.hotkey("\\", command=True)

@apollo.command(
    phrases=["run code", "run this", "run file", "execute"],
    description="Runs code — sends Cmd+Shift+B (build task) or you can change this"
)
def run_code():
    apollo.hotkey("b", command=True, shift=True)
    apollo.say("Running")


# --- System Commands ---

@apollo.command(
    phrases=["take screenshot", "screenshot", "screen capture"],
    description="Takes a screenshot (Cmd+Shift+3)"
)
def screenshot():
    apollo.hotkey("3", command=True, shift=True)
    apollo.say("Screenshot taken")

@apollo.command(
    phrases=["lock screen", "lock computer", "lock mac"],
    description="Locks the screen"
)
def lock_screen():
    apollo.hotkey("q", command=True, ctrl=True)
    apollo.say("Locking")

@apollo.command(
    phrases=["volume up", "louder", "turn it up", "make it louder",
             "turn volume up", "a bit louder"],
    description="Increases volume"
)
def volume_up():
    apollo.applescript('set volume output volume ((output volume of (get volume settings)) + 10)')

@apollo.command(
    phrases=["volume down", "quieter", "turn it down"],
    description="Decreases volume"
)
def volume_down():
    apollo.applescript('set volume output volume ((output volume of (get volume settings)) - 10)')

@apollo.command(
    phrases=["mute", "mute volume", "shut up", "silence"],
    description="Mutes volume"
)
def mute():
    apollo.applescript('set volume output muted true')
    apollo.say("Muted")

@apollo.command(
    phrases=["unmute", "unmute volume"],
    description="Unmutes volume"
)
def unmute():
    apollo.applescript('set volume output muted false')
    apollo.say("Unmuted")


# --- Typing / Dictation ---

@apollo.command(
    phrases=["type", "write", "dictate"],
    description="Types out whatever you say after 'type'. E.g. 'Biggie, type hello world'"
)
def type_text():
    """Special handler — the actual text is injected by the command router."""
    pass  # Handled specially in route_command()


# --- Biggie Meta ---

@apollo.command(
    phrases=["help", "what can you do", "list commands", "show commands"],
    description="Lists all available commands"
)
def show_help():
    apollo.say("Here are my commands")
    print("\n" + "=" * 60)
    print("  BIGGIE COMMANDS")
    print("=" * 60)
    for cmd in apollo.COMMANDS:
        trigger = cmd["phrases"][0]
        desc = cmd["description"]
        print(f"  \"{trigger}\"  ->  {desc}")
    print("=" * 60 + "\n")

@apollo.command(
    phrases=["stop listening", "go to sleep", "sleep", "pause",
             "stop", "shut down", "goodbye"],
    description="Stops Biggie"
)
def stop_listening():
    apollo.say("Going to sleep. Goodbye!")
    raise SystemExit(0)
