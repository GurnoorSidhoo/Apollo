"""
Biggie Custom Commands
=======================
Add your own commands here! Just follow the pattern below.

After editing this file, restart Biggie to pick up changes.

HOW TO ADD A COMMAND:
    1. Write a function
    2. Decorate it with @command(phrases=[...])
    3. That's it!

The 'phrases' list is what you can SAY to trigger it.
You don't have to be exact — Biggie does fuzzy matching.
"""

# Import helpers from the apollo package.
from apollo import command, say, mac_open, mac_open_app, applescript, hotkey, type_string


# ===========================================================================
# YOUR CUSTOM COMMANDS
# ===========================================================================

# --- Example: Open your project ---
@command(
    phrases=["open my project", "open project", "go to project"],
    description="Opens your main project folder in VS Code"
)
def open_my_project():
    # Change this path to your actual project!
    import subprocess
    project_path = "~/Developer/my-project"
    subprocess.Popen(["code", project_path])
    say("Opening your project")


# --- Example: Open multiple things at once ---
@command(
    phrases=["start coding", "coding mode", "let's code", "work mode"],
    description="Opens VS Code, Terminal, and Claude — your coding setup"
)
def coding_mode():
    say("Setting up your coding environment")
    mac_open_app("Visual Studio Code")
    import time
    time.sleep(1)
    mac_open_app("Claude")
    time.sleep(0.5)
    say("Ready to code")


# --- Example: Git shortcuts ---
@command(
    phrases=["git status", "check git", "status"],
    description="Runs git status in the current terminal"
)
def git_status():
    type_string("git status")
    hotkey("\r")  # Press Enter
    say("Checking git status")


@command(
    phrases=["git commit", "commit changes", "commit"],
    description="Starts a git commit (you say the message after)"
)
def git_commit():
    type_string('git commit -m "')
    say("Type your commit message")


@command(
    phrases=["git push", "push code", "push it", "push"],
    description="Pushes to current branch"
)
def git_push():
    type_string("git push")
    hotkey("\r")
    say("Pushing code")


@command(
    phrases=["git pull", "pull code", "pull latest"],
    description="Pulls latest from remote"
)
def git_pull():
    type_string("git pull")
    hotkey("\r")
    say("Pulling latest")


# --- Example: Quick navigation ---
@command(
    phrases=["go to stack overflow", "open stack overflow", "stack overflow"],
    description="Opens Stack Overflow"
)
def open_stackoverflow():
    mac_open("https://stackoverflow.com")
    say("Opening Stack Overflow")


@command(
    phrases=["go to docs", "open documentation", "open docs"],
    description="Opens your go-to documentation site"
)
def open_docs():
    # Change this to whatever docs you use most!
    mac_open("https://developer.mozilla.org")
    say("Opening docs")


# --- Example: Brightness / Display ---
@command(
    phrases=["dark mode", "go dark", "lights off"],
    description="Toggles macOS dark mode"
)
def dark_mode():
    applescript('''
        tell application "System Events"
            tell appearance preferences
                set dark mode to not dark mode
            end tell
        end tell
    ''')
    say("Toggling dark mode")


# --- Example: Timer ---
@command(
    phrases=["set timer", "start timer", "pomodoro", "focus mode"],
    description="Starts a 25-minute focus timer"
)
def focus_timer():
    import threading
    say("Starting 25 minute focus timer")
    def timer_done():
        say("Time's up! Take a break.")
    threading.Timer(25 * 60, timer_done).start()


# --- Example: Quick type templates ---
@command(
    phrases=["type console log", "console log"],
    description="Types console.log() for you"
)
def type_console_log():
    type_string("console.log()")
    # Move cursor back one character to be inside the parentheses
    hotkey("[", ctrl=True)  # This might vary by editor


@command(
    phrases=["type function", "new function"],
    description="Types a function template"
)
def type_function():
    type_string("function () {\n  \n}")


@command(
    phrases=["type print", "print statement"],
    description="Types print() for Python"
)
def type_print():
    type_string("print()")
