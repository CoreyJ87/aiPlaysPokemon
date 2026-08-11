#!/usr/bin/env python
"""save_game.py - park the game safely so you can take a break.

Flees the current battle if there is one (wild only - a trainer battle can't
be fled, so finish that first), clears any open dialog, opens the START menu,
finds SAVE by reading the cursor off the screen, and saves the game. Run it
any time; it does nothing destructive and re-saving is always safe.

    .venv/bin/python save_game.py            # default host/port
    .venv/bin/python save_game.py --host 127.0.0.1 --port 54321

Stop player_ai first (Ctrl-C it or kill the process) - two drivers pressing
buttons at once will fight over the cursor.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "mGBA"))
sys.path.insert(0, str(HERE))

import cv2                      # noqa: E402
import numpy as np              # noqa: E402
from mgba_client import MGBAClient  # noqa: E402
from screen_state import measure    # noqa: E402

# gBattleTypeFlags (FRLG, same address in FR 1.0 and LG 1.1 per pret symbols).
BATTLE_TYPE_FLAGS_ADDR = 0x02022B4C
BATTLE_TYPE_TRAINER = 0x08

# START-menu geometry on the 240x160 frame, measured empirically: the cursor
# arrow renders at x 177-182, item rows are 16px tall starting at y=6.
ARROW_X = slice(177, 183)
TEXT_X = slice(185, 236)
ROW_H = 16
ROW_Y0 = 6
# The menu tops out at 7 entries (POKEDEX..EXIT); the row-8 band would read
# the description bar at the bottom of the screen as an item.
MAX_ROWS = 7


def tap(c, button, settle=0.5):
    c.tap(button, 12)
    time.sleep(settle)


def frame(c):
    png = c.screenshot()
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def menu_rows(img):
    """(cursor_index, item_count) read off the screen, or (None, 0)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cursor, count = None, 0
    for i in range(MAX_ROWS):
        band = slice(ROW_Y0 + i * ROW_H, ROW_Y0 + i * ROW_H + ROW_H - 4)
        if (gray[band, TEXT_X] < 100).sum() >= 8:      # a word of dark text
            count = i + 1
            if (gray[band, ARROW_X] < 100).any():      # the arrow glyph
                cursor = i
    return cursor, count


def flee_battle(c):
    """Get out of a wild battle; refuse politely on a trainer one."""
    flags = int.from_bytes(c.peek(BATTLE_TYPE_FLAGS_ADDR, 4), "little")
    if flags & BATTLE_TYPE_TRAINER:
        sys.exit("In a TRAINER battle - there is no running from those. "
                 "Finish it (or let player_ai finish it), then rerun this.")
    print("In a wild battle - fleeing ...")
    for attempt in range(10):
        tap(c, "b", 0.8)            # back out of any move/bag submenu
        tap(c, "down", 0.4)         # 2x2 action menu: RUN is bottom-right
        tap(c, "right", 0.4)
        tap(c, "a", 2.0)
        tap(c, "a", 1.5)            # clear "Got away safely!"
        if not c.game_state().get("in_battle"):
            print("  fled.")
            return
    sys.exit("Could not flee after 10 tries - is a move animation stuck? "
             "Check the emulator window.")


def clear_dialog(c):
    for _ in range(14):
        if not measure(frame(c))["open"]:
            return
        tap(c, "a", 0.6)
    print("  (a text box would not close; continuing anyway)")


def save(c):
    for attempt in range(3):
        tap(c, "start", 1.0)
        cursor, count = menu_rows(frame(c))
        if cursor is not None:
            break
        tap(c, "b", 0.5)            # whatever opened, it wasn't the menu
    else:
        sys.exit("Could not open the START menu - check the emulator window.")

    save_idx = count - 3            # SAVE sits above OPTION and EXIT, always
    for _ in range((save_idx - cursor) % count):
        tap(c, "down", 0.4)

    cursor, _ = menu_rows(frame(c))
    if cursor != save_idx:
        sys.exit(f"Cursor ended on row {cursor}, expected {save_idx} - "
                 f"menu layout surprise. Save manually in the emulator.")

    stale = (c.screen().get("dialog_text") or "").lower()
    tap(c, "a", 1.2)                # SAVE -> "Would you like to save?"

    # Drive whatever prompts appear rather than counting presses: a fresh
    # save asks once, an overwrite asks twice, and the pauses in between
    # aren't fixed. YES is always the default cursor position. gStringVar4
    # is sticky, so "saved the game" only counts once we have seen (and
    # answered) a prompt this run - the stale copy from a previous save
    # would otherwise pass for success instantly.
    answered = False
    for _ in range(30):
        text = (c.screen().get("dialog_text") or "").lower()
        if "saved the game" in text and (answered or text != stale):
            tap(c, "a", 0.5)        # dismiss the confirmation
            tap(c, "b", 0.3)        # close the menu
            return True
        if "like to save" in text or "overwrite" in text:
            tap(c, "a", 0.8)        # YES
            answered = True
            continue
        time.sleep(0.7)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=54321)
    args = ap.parse_args()

    c = MGBAClient(host=args.host, port=args.port)
    try:
        c.connect()
    except OSError as exc:
        sys.exit(f"Could not reach mGBA on {args.host}:{args.port} - {exc}. "
                 f"Is mgba_server.lua loaded?")

    if c.game_state().get("in_battle"):
        flee_battle(c)
    clear_dialog(c)
    ok = save(c)

    state = c.game_state()
    p = state["player"]
    party = ", ".join(
        f"{q.get('nickname') or q.get('species_name')} Lv{q.get('level')} "
        f"{q.get('hp')}/{q.get('max_hp')}" for q in state.get("party", []))
    print(f"{'Saved.' if ok else 'Save NOT confirmed - check the emulator!'}"
          f"  {p['name']} @ map ({p['map_bank']},{p['map_number']}) "
          f"${p['money']}  [{party}]")
    print("Safe to close mGBA. To resume: load the ROM (it loads the .sav "
          "automatically), load mGBA/mgba_server.lua, rerun player_ai.py.")
    c.close()


if __name__ == "__main__":
    main()
