#!/usr/bin/env python3
"""One-shot launcher: opens mGBA, loads the Lua server script, continues the
saved game, and dismisses the "previously on your quest" recap.

    .venv/bin/python launch_game.py            # get to the overworld and stop
    .venv/bin/python launch_game.py --play     # ...then start player_ai on the
                                               # next run number
    .venv/bin/python launch_game.py --turns 500 --play

Needs Accessibility permission for whatever app runs it (System Settings →
Privacy & Security → Accessibility) — the Lua script can only be loaded by
driving mGBA's menus, since mGBA 0.10 has no --script flag.

Idempotent: if the Lua server is already answering on port 54321 it skips
launching/loading and only drives whatever screen the game is on.
"""

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "mGBA"))
from mgba_client import MGBAClient, MGBAError  # noqa: E402

LUA_SCRIPT = os.path.join(REPO, "mGBA", "mgba_server.lua")
PORT = 54321

with open(os.path.join(REPO, "mGBA", "screens.json")) as f:
    SCREENS = json.load(f)

# how each screen on the way in is dismissed
INTRO_BUTTON = {
    "?intro_movie_1": "A", "?intro_movie_2": "A", "?intro_movie_3": "A",
    "?title_or_copyright": "A",
    "?main_menu_init": "A", "?main_menu": "A", "?main_menu_return": "A",
}


def lua_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            return True
    except OSError:
        return False


def find_rom() -> str:
    roms = sorted(glob.glob(os.path.join(REPO, "*Leaf Green*.gba")))
    if not roms:
        sys.exit("no LeafGreen ROM found in the repo directory")
    return roms[0]


def load_lua_script():
    """Drive mGBA's menus to load mgba_server.lua (needs Accessibility)."""
    script = f'''
    tell application "mGBA" to activate
    delay 1
    tell application "System Events"
      tell process "mGBA"
        -- open the Scripting window if the main window is frontmost;
        -- if the Scripting window is already up, its File menu is active
        if exists menu "Tools" of menu bar 1 then
          click (first menu item of menu "Tools" of menu bar 1 whose name begins with "Scripting")
          delay 1.5
        end if
        set fileMenu to menu "File" of menu bar 1
        set recentMatch to missing value
        try
          set recentItem to (first menu item of fileMenu whose name begins with "Load recent")
          set recentMatch to (first menu item of menu 1 of recentItem whose name ends with "mgba_server.lua")
        end try
        if recentMatch is not missing value then
          click recentMatch
        else
          click (first menu item of fileMenu whose name begins with "Load script")
          delay 1
          keystroke "g" using {{command down, shift down}}
          delay 0.7
          keystroke "{LUA_SCRIPT}"
          delay 0.3
          keystroke return
          delay 0.7
          keystroke return
        end if
      end tell
    end tell
    '''
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        if "assistive access" in r.stderr:
            sys.exit(
                "macOS blocked the menu automation. Grant Accessibility to the "
                "app running this script (System Settings → Privacy & Security "
                "→ Accessibility), then rerun."
            )
        sys.exit(f"couldn't load the Lua script via the menus:\n{r.stderr}")


def wait_for_lua(timeout=30) -> bool:
    for _ in range(timeout * 2):
        if lua_up():
            return True
        time.sleep(0.5)
    return False


def screen_name(c: MGBAClient) -> str:
    cb = c.screen().get("callback2", "")
    return SCREENS.get(cb, cb)


def drive_to_overworld(c: MGBAClient):
    """Press through intro movie → title → CONTINUE → recap → overworld."""
    for _ in range(90):
        name = screen_name(c)
        if name == "overworld" or name.startswith("battle"):
            break
        c.tap(INTRO_BUTTON.get(name, "A"), 12)
        time.sleep(0.7)
    else:
        sys.exit(f"never reached the overworld (stuck on: {screen_name(c)})")
    # the "previously on your quest" recap runs as a task under the overworld
    # callback; B closes it and is a no-op once it's gone
    for _ in range(4):
        c.tap("B", 12)
        time.sleep(0.5)
    pos = c.position()
    print(f"in the overworld: {pos.get('map_name', pos)}")


def ensure_viewer():
    check = subprocess.run(["pgrep", "-f", "viewer/server.py"], capture_output=True)
    if check.returncode != 0:
        subprocess.Popen(
            [os.path.join(REPO, ".venv/bin/python"), os.path.join(REPO, "viewer/server.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("viewer started on port 8777")


def next_run_log() -> str:
    nums = [int(m.group(1)) for p in glob.glob(os.path.join(REPO, "runs", "run-*.jsonl"))
            if (m := re.search(r"run-(\d+)\.jsonl$", p))]
    return os.path.join(REPO, "runs", f"run-{max(nums, default=0) + 1}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="start player_ai after loading in")
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--ollama-host", default="https://ollama.synik4l.net")
    args = ap.parse_args()

    ensure_viewer()

    if lua_up():
        print("Lua server already running — skipping launch")
    else:
        subprocess.run(["open", "-a", "mGBA", find_rom()], check=True)
        time.sleep(3)
        print("mGBA launched, loading the Lua script…")
        load_lua_script()
        if not wait_for_lua():
            sys.exit("the Lua server never came up — load mGBA/mgba_server.lua "
                     "by hand (Tools → Scripting → File → Load script)")
        print("Lua server up")

    with MGBAClient() as c:
        try:
            drive_to_overworld(c)
        except MGBAError as e:
            sys.exit(f"emulator stopped responding: {e}")

    if args.play:
        log = next_run_log()
        print(f"starting player: {os.path.basename(log)}")
        sys.stdout.flush()
        os.execv(os.path.join(REPO, ".venv/bin/python"), [
            "python", os.path.join(REPO, "player_ai.py"), "play",
            "--ollama-host", args.ollama_host, "--model", args.model,
            "--turns", str(args.turns), "--log", log,
            "--screenshot", os.path.join(REPO, "runs", "last-frame.png"),
        ])


if __name__ == "__main__":
    main()
