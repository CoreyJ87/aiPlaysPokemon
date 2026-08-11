"""
player_ai.py - the harness that lets a local LLM actually play Pokemon Leaf Green.

This is the top of the stack. Everything below it already works and is left
alone; this file's whole job is to turn the three tool modules into something a
small local model can drive:

    mGBA/mgba_client.py        buttons, screenshots, GAME_STATE
    locationTracking/          where am I, and how do I walk to X
    battle/                    what will each of my moves actually do
    objectives.py              what am I supposed to be doing, and how far in

One turn of the loop is:

    screenshot -> game state -> location fix -> (battle: damage table)
        -> advance the walkthrough if the game says an objective is done
        -> render it all as a compact report -> ask the model
        -> parse one command out of the reply -> execute it -> remember it

Three design notes worth reading before changing anything:

* No JSON tool calling. Gemma's chat template has no tool-call slot, and the
  combination "images + tools" is the least reliable corner of every local
  runtime. So the tool surface is a one-line text grammar (`goto viridian city`,
  `use ember`, `press a 2`) and the parser is deliberately forgiving - it
  fuzzy-matches the verb, fuzzy-matches the argument against the names actually
  on offer this turn, and retries with a correction before it gives up. A 12B
  model gets a grammar right far more often than it gets a JSON schema right.

* The command list in the prompt is generated from the same table that
  dispatches it, and is filtered to what is legal right now (battle commands
  only appear in battle). Documentation that can't drift, and a smaller menu to
  choose from, which is most of the battle with a small model.

* Progress is measured, never asserted. objectives.json says what "done" means
  for the current objective as a condition over the game state, and the harness
  advances the moment RAM agrees - the model is never asked whether it has
  finished, because a model asked that says yes. What the model does own is
  memories.json, which it writes with `note`, and which is what stops it
  rediscovering the same locked door every twenty turns.

Usage:
    python player_ai.py                 # play: loop until Ctrl-C
    python player_ai.py once            # a single turn, then stop
    python player_ai.py manual          # drive the tool layer by hand, no LLM
    python player_ai.py dry-run         # ask the model, print, execute nothing
    python player_ai.py gui             # play, with a pause/request console
                                         # (operator_gui.py) alongside the terminal
    python player_ai.py --goal "get to Pewter City and beat Brock"
    python player_ai.py --no-objectives # play with no walkthrough at all
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE / "mGBA", HERE / "battle", HERE / "locationTracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ollama  # noqa: E402

from mgba_client import (BUTTONS, MGBAClient, MGBAError,  # noqa: E402
                         gen3_decode, gen3_decode_text)
from navigator import Navigator  # noqa: E402
from pathfinder import BLOCKED, RETURN_TARGET  # noqa: E402
from damage_calc import GameData  # noqa: E402
from live_calc import (  # noqa: E402
    Session,
    Snapshot,
    build_rows,
    effective_speed,
    ko_text,
    move_names,
    print_matchup,
)
from matchup import (  # noqa: E402
    Roster,
    assess,
    describe as describeReadiness,
    levelsNeeded,
    summarize as summarizeReadiness,
)
from objectives import (  # noqa: E402
    Memory,
    ObjectiveBook,
    renderObjective,
)
from screen_state import measure as measureScreen  # noqa: E402
from naming_screen import Keyboard, NamingError, currentName  # noqa: E402
from naming_screen import isOpen as namingScreenOpen  # noqa: E402

# Raw taps into a menu need a beat to land: the tap returns when its frames
# elapse, but the menu redraws a few frames later, and a second tap fired into
# that window is eaten. Walking doesn't need this (navigator verifies each step
# against RAM), menus do.
MENU_TAP_FRAMES = 12
MENU_TAP_DELAY = 0.15

# gBattleTypeFlags (FRLG). Bit 3 is the trainer flag - clear means a wild
# battle, which is the only kind `run` works in. Read over PEEK rather than
# guessed from party sizes, because plenty of early trainers carry one mon.
# Verified empirically against LeafGreen v1.1 (wild battle read 0x4:
# IS_MASTER set, TRAINER clear).
BATTLE_TYPE_FLAGS_ADDR = 0x02022B4C
BATTLE_TYPE_TRAINER = 0x08

# callback2 values (see mGBA/screens.json) that mean the battle proper is on
# screen. gMain.inBattle stays true while the party menu, a summary page or
# the bag is open on top of the fight - and a move table rendered for those
# screens sends `use` taps into whatever menu is actually drawn. Anything
# in-battle but not in this set is such an overlay.
BATTLE_SCREEN_CALLBACKS = {
    "080565BD",   # battle_start_from_field
    "0800FDB1",   # battle_init
    "0801051D",   # battle_intro
    "08011115",   # battle_main
    "080777FD",   # battle_reshow_after_menu
}

# Battle menus are 2x2 grids whose cursor persists between turns, so "press A
# twice to attack" is not reliable - you might attack with whatever you picked
# last turn. The cursor movement is clamped rather than wrapping (DPAD_LEFT is
# ignored in the left column, DPAD_UP in the top row), which gives us a free
# reset: LEFT then UP lands on slot 0 from anywhere. Every battle action below
# normalises that way first and then walks to the slot it wants.
#
#   action menu:  FIGHT 0  BAG 1        move menu:  slot0  slot1
#                 POKEMON 2 RUN 3                   slot2  slot3
ACTION_FIGHT, ACTION_BAG, ACTION_POKEMON, ACTION_RUN = 0, 1, 2, 3

# The battle cursors CAN be read: gActionSelectionCursor / gMoveSelectionCursor
# (same addresses in FR 1.0 and LG 1.1 per pret's symbol files). One byte per
# battler, battler 0 is the player in singles. Reading them upgrades the blind
# LEFT+UP reset to a verified walk - a tap eaten by an animation gets noticed
# and retried instead of silently attacking with the wrong move.
ACTION_CURSOR_ADDR = 0x02023FF8
MOVE_CURSOR_ADDR = 0x02023FFC

# gBagMenuState (include/item_menu.h): +0x06 u16 pocket (0 items, 1 key
# items, 2 poke balls), +0x08 u16 itemsAbove[3], +0x0E u16 cursorPos[3].
# The battle BAG is the regular bag, so one struct serves both.
BAG_STATE_ADDR = 0x0203ACFC
BALL_POCKET = 2

# sShopData (src/shop.c in pret/pokefirered; same address FR 1.0 / LG 1.1):
#   +0x04 const u16 *itemList   +0x0C u16 selectedRow  +0x0E u16 scrollOffset
#   +0x10 u16 itemCount         +0x14 u16 maxQuantity
# selectedRow+scrollOffset is the highlighted item's index in itemList, which
# makes the whole buy flow verifiable instead of counted-out blind.
SHOP_DATA_ADDR = 0x02039934
# Item data table in ROM: 44-byte entries, name (gen3 text, 14 bytes) first,
# price as u16 at +16. The address is per-ROM; add entries as games appear.
ITEM_TABLE_BY_GAME = {
    "LeafGreen v1.1": 0x083DAED4,
}
ITEM_ENTRY_SIZE = 44

# Move learning at four known moves. gMoveToLearn holds the pending move id
# while the "trying to learn" prompts run, and gDisplayedStringBattle is the
# text actually drawn in battle - gStringVar4 goes stale there, this doesn't.
# Move names live in ROM at 13 bytes per entry (same table the Lua server
# reads); addresses are per-ROM like the item table above.
DISPLAYED_STRING_ADDR = 0x0202298C
MOVE_TO_LEARN_ADDR = 0x02024022
MOVE_TABLE_BY_GAME = {
    "LeafGreen v1.1": 0x082470E0,
}
MOVE_NAME_SIZE = 13
LEARN_PROMPT_MARKERS = ("trying to learn", "make room for",
                        "which move should be forgotten", "stop learning")

DIRECTIONS = ("Up", "Down", "Left", "Right")
DIR_ALIASES = {"u": "Up", "up": "Up", "n": "Up", "north": "Up",
               "d": "Down", "down": "Down", "s": "Down", "south": "Down",
               "l": "Left", "left": "Left", "w": "Left", "west": "Left",
               "r": "Right", "right": "Right", "e": "Right", "east": "Right"}

# How many taps one command may fire. A model that answers "press a 50" is
# usually confused, and 50 blind A presses in the overworld can sell your
# Pokemon to a trade NPC.
MAX_PRESSES = 8

# Map exits offered as destinations, so "leave town and go north" is something
# the player can ask for by name. Only the current map's exits and those one
# hop away: that is enough to get out of a building and out of a town, and it
# keeps the survey to a handful of extra route plans per turn.
EXIT_HOPS = 1
EXIT_CATEGORY = "exit"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Everything tunable in one place (the old TODO's yaml file, in Python)."""

    model: str = "gemma4:12b"
    ollamaHost: str | None = None       # None = ollama's own default
    mgbaHost: str = "127.0.0.1"
    mgbaPort: int = 54321

    screenshotPath: Path = HERE / "screenshot.png"
    sendImage: bool = True

    temperature: float = 0.6
    numPredict: int = 200
    keepAlive: str = "30m"              # keep the model resident between turns
    # Gemma 4 is a thinking model, and left to itself it spends the whole token
    # budget reasoning and returns an empty message - the answer never arrives.
    # Thinking off is both the reliable setting and the honest one for a local
    # model on a laptop: a turn should cost seconds, not a minute of monologue.
    think: bool = False

    historyLength: int = 6              # recent action/result pairs in the prompt
    destinationLimit: int = 10          # walkable places listed per turn
    showDestinations: bool = True
    noteLimit: int = 6                  # remembered notes shown per turn

    objectivesPath: Path = HERE / "objectives.json"
    memoriesPath: Path = HERE / "memories.json"
    useObjectives: bool = True
    # Doing the same thing this many times with the same result is the loop the
    # objectives are meant to break; say so in the prompt when it happens.
    repeatAlert: int = 3

    # Text-box handling. Detection is a pixel heuristic, so it gets a trust
    # window: after this many turns of claiming a box is open with nothing in
    # the game changing, the harness stops acting on it and hands the walking
    # commands back. A misread would otherwise wedge the player pressing A at
    # scenery. The window is short because pressing A at a real box always
    # changes something - it turns the page, or it closes the box and the player
    # can move again - so several presses that change nothing at all is already
    # strong evidence there is no box, not a conversation being slow.
    detectDialog: bool = True
    dialogTrustTurns: int = 3
    moveBudget: int = 300               # navigator step budget per goal
    parseRetries: int = 2               # re-asks before falling back
    turnDelay: float = 0.4              # pause between turns, seconds

    goal: str = ""
    logPath: Path | None = None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for matching."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def _bestMatch(needle: str, candidates: list, cutoff: float = 0.55):
    """Fuzzy-pick one of `candidates` (list of (label, payload)).

    Tried in order of how much we trust it: exact, prefix/substring, then
    difflib. The model rarely reproduces a name exactly ("pokemon center" for
    "Pokemon Center 1F", "ember!" for "Ember"), and refusing those would waste a
    whole turn on a typo.
    """
    want = _norm(needle)
    if not want or not candidates:
        return None
    table = [(_norm(label), label, payload) for label, payload in candidates]

    for key, label, payload in table:
        if key == want:
            return label, payload
    hits = [(key, label, payload) for key, label, payload in table
            if key.startswith(want) or want in key or key in want]
    if hits:
        hits.sort(key=lambda h: abs(len(h[0]) - len(want)))
        return hits[0][1], hits[0][2]

    close = difflib.get_close_matches(want, [k for k, _l, _p in table], n=1,
                                      cutoff=cutoff)
    if close:
        for key, label, payload in table:
            if key == close[0]:
                return label, payload
    return None


def friendlyMapName(mapName: str) -> str:
    """'3-19-Route1' -> 'Route 1'. Map ids are for the tools, not the player."""
    name = re.sub(r"^\d+-\d+-", "", str(mapName))
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)      # PalletTown -> Pallet Town
    name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", name)      # Route1 -> Route 1
    return " ".join(name.split())


class _CachedState:
    """Hands live_calc's Snapshot the GAME_STATE we already fetched this turn.

    Snapshot.capture() wants a client so it can call game_state(); giving it one
    that just replays a dict keeps the battle report free of a second round trip
    without forking any of live_calc's reshaping logic.
    """

    def __init__(self, raw: dict):
        self._raw = raw

    def game_state(self) -> dict:
        return self._raw


def _captureText(fn, *args, **kwargs) -> str:
    """Run a print-based renderer and return what it printed.

    live_calc's print_matchup() is the exact battle table a human reads at the
    REPL, tuned over a lot of real fights. Reimplementing it to return a string
    would mean maintaining two copies that slowly disagree.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        fn(*args, **kwargs)
    return sink.getvalue().rstrip()


# --------------------------------------------------------------------------
# The tool surface
# --------------------------------------------------------------------------


class ActionError(RuntimeError):
    """The command was understood but can't be run (bad argument, wrong screen)."""


@dataclass
class Command:
    name: str
    usage: str
    help: str
    aliases: tuple = ()
    context: str = "any"          # 'any' | 'battle' | 'overworld'


COMMANDS = (
    Command("press", "press <button> [times]",
            "Tap a button: a, b, up, down, left, right, start, select. "
            "Use this for dialog, menus and anything the other commands can't do.",
            aliases=("tap", "button", "hit")),
    Command("move", "move <up|down|left|right> [steps]",
            "Walk that way, one verified tile at a time. Stops early if blocked.",
            aliases=("walk", "step", "go"), context="overworld"),
    Command("goto", "goto <place>",
            "Walk all the way to one of the places listed above. Handles doors, "
            "routes and pathfinding for you.",
            aliases=("travel", "goto_place", "navigate"), context="overworld"),
    Command("heal", "heal",
            "Walk to the nearest Pokemon Center and step up to the nurse.",
            aliases=("pc", "center"), context="overworld"),
    Command("catch", "catch <species>",
            "Walk to grass where that species lives and pace until one appears.",
            context="overworld"),
    Command("collect", "collect <item>",
            "Walk to the nearest uncollected item ball of that name and pick it up.",
            aliases=("pickup", "grab"), context="overworld"),
    Command("buy", "buy <item> [quantity]",
            "Buy from the shop clerk directly in front of you (stand facing "
            "them first). Works the mart menus for you, e.g. `buy potion 5`.",
            aliases=("shop", "purchase"), context="overworld"),
    Command("use", "use <move name>",
            "Attack with one of your moves. Name the move, e.g. `use ember`.",
            aliases=("attack", "fight", "move_use"), context="battle"),
    Command("switch", "switch <pokemon>",
            "Send out a different Pokemon from your party.",
            aliases=("swap", "sub"), context="battle"),
    Command("bag", "bag",
            "Open the bag in battle, then use `press` to pick an item.",
            aliases=("item", "items"), context="battle"),
    Command("run", "run",
            "Try to flee the battle. Only works against wild Pokemon.",
            aliases=("flee", "escape"), context="battle"),
    Command("throw", "throw [ball name]",
            "Throw a Poke Ball at the wild Pokemon to catch it. Works best "
            "when it is weakened or asleep. Wild battles only.",
            aliases=("ball", "catch_ball"), context="battle"),
    Command("learn", "learn [skip]",
            "Answer the 'trying to learn a move' prompt: makes room by "
            "forgetting the weakest overlapping move, or `learn skip` to "
            "refuse. Only when that prompt is on screen.",
            aliases=("forget", "teach"), context="battle"),
    Command("note", "note <something worth remembering>",
            "Write one short fact into your memory so you still have it in a "
            "hundred turns: where a door was, what an NPC told you, what did "
            "not work.",
            aliases=("remember", "write")),
    Command("check", "check <trainer>",
            "Ask the damage calculator whether your party could beat a trainer "
            "you have already met, and what to train.",
            aliases=("assess", "readiness", "scout")),
    Command("name", "name <a short name>",
            "Type a name on the keyboard screen and confirm it. Give a name of "
            "1-10 letters; the harness presses the keys.",
            aliases=("nickname", "call", "type"), context="naming"),
    Command("wait", "wait [seconds]",
            "Do nothing and let an animation or cutscene finish.",
            aliases=("idle", "nothing", "pass")),
)

VERB_LOOKUP = {}
for _c in COMMANDS:
    VERB_LOOKUP[_c.name] = _c.name
    for _a in _c.aliases:
        VERB_LOOKUP[_a] = _c.name


class Actions:
    """Executes one parsed command against the game.

    Every method returns a short string describing what happened; that string is
    what the model sees next turn, so it is written for the model, not for a log
    file - it says whether the thing worked and what changed.
    """

    def __init__(self, nav: Navigator, cfg: Config, memory=None, roster=None,
                 data=None, battle=None):
        self.nav = nav
        self.cfg = cfg
        self.client = nav.client
        self.memory = memory        # objectives.Memory, for `note`
        self.roster = roster        # matchup.Roster, for `check`
        self.data = data            # damage_calc.GameData, for `check`
        self.battle = battle        # live_calc.Session, for `learn`
        # Set by PlayerAI each turn so battle commands can name real moves.
        self.observation: "Observation | None" = None
        self.turn = 0               # for stamping notes

    # ---- primitives -------------------------------------------------------

    def _menuTap(self, button: str, times: int = 1):
        for _ in range(times):
            self.client.tap(button.upper(), MENU_TAP_FRAMES)
            time.sleep(MENU_TAP_DELAY)

    def _resetCursor(self):
        """Park a 2x2 battle cursor on slot 0 (see the note at the top)."""
        self._menuTap("LEFT")
        self._menuTap("UP")

    def _cursorTo(self, slot: int):
        """Walk from slot 0 to `slot` in a 2x2 grid."""
        if slot & 1:
            self._menuTap("RIGHT")
        if slot & 2:
            self._menuTap("DOWN")

    def _readCursor(self, addr: int):
        """One byte of cursor RAM, or None when the read isn't available."""
        try:
            value = self.client.peek(addr, 1)[0]
        except Exception:
            return None
        return value if value <= 3 else None

    def _cursorSeek(self, addr: int, slot: int, label: str):
        """Walk a 2x2 battle cursor to `slot`, verified against RAM.

        The blind reset works until an animation eats a tap; the cursor RAM
        turns "should be on FIGHT now" into "is on FIGHT". Falls back to the
        blind walk when the read fails (wrong game hash, server without PEEK).
        """
        for _ in range(4):
            cur = self._readCursor(addr)
            if cur is None:
                self._resetCursor()
                self._cursorTo(slot)
                return
            if cur == slot:
                return
            if (cur & 1) != (slot & 1):
                self._menuTap("RIGHT" if slot & 1 else "LEFT")
            if (cur & 2) != (slot & 2):
                self._menuTap("DOWN" if slot & 2 else "UP")
        raise ActionError(f"the {label} cursor is not responding - an "
                          f"animation is probably still playing. `wait 1` "
                          f"and try again")

    def _chooseAction(self, slot: int):
        self._cursorSeek(ACTION_CURSOR_ADDR, slot, "action menu")
        self._menuTap("A")

    def _requireBattle(self):
        obs = self.observation
        if obs is None or not obs.inBattle:
            raise ActionError("that only works in battle")
        return obs

    def _requireNoDialog(self):
        """Refuse to walk while text is on screen, and say why.

        The game would ignore the input anyway; the difference is that a
        refusal explains itself in one turn, where being ignored looks exactly
        like a blocked tile and gets retried until the step budget runs out.
        """
        obs = self.observation
        if obs is not None and obs.namingOpen:
            raise ActionError("a naming keyboard is on screen - answer with "
                              "`name <what to call it>` first")
        if obs is not None and obs.dialogOpen and not obs.dialogDoubted:
            raise ActionError("there is a text box on screen - the game ignores "
                              "movement until it is cleared. Press A to advance "
                              "dialog, or B to back out if a menu is open")

    # ---- overworld --------------------------------------------------------

    def press(self, args: list) -> str:
        if not args:
            raise ActionError("press needs a button, e.g. `press a`")
        button = DIR_ALIASES.get(args[0].lower(), args[0]).upper()
        if button not in BUTTONS:
            raise ActionError(f"{args[0]!r} is not a button; use one of "
                              f"{', '.join(b.lower() for b in BUTTONS)}")
        times = 1
        if len(args) > 1:
            match = re.search(r"\d+", args[1])
            if match:
                times = max(1, min(MAX_PRESSES, int(match.group())))

        before = self._worldMark()
        self._menuTap(button, times)

        # navigator caches which way we're facing to save a tap per step. A raw
        # tap always ends facing that direction; an A/B press may have been a
        # cutscene that spun us, so drop the cache rather than trust it.
        title = button.title()
        self.nav.facing = title if title in DIRECTIONS else None
        label = f"pressed {button}" + (f" x{times}" if times > 1 else "")
        return label + self._pressEffect(before)

    def _worldMark(self) -> dict | None:
        """The two things a button press can change: where you are, and whether
        there is a box on screen."""
        try:
            state = self.client.game_state()
        except (MGBAError, ValueError):
            return None
        player = state.get("player", {}) or {}
        dialog = False
        if self.cfg.detectDialog and not state.get("in_battle"):
            try:
                dialog = bool(measureScreen(self.client.screenshot())["open"])
            except (MGBAError, ValueError):
                dialog = False
        return {"battle": bool(state.get("in_battle")),
                "dialog": dialog,
                "ram": (player.get("map_bank"), player.get("map_number"),
                        player.get("x"), player.get("y"))}

    def _pressEffect(self, before: dict | None) -> str:
        """Say what the press actually did.

        Every other command here reports its outcome - `move` says where you
        ended up, `goto` says whether it arrived. `press` used to answer
        "pressed A" whether it had opened a conversation or tapped thin air,
        which is the one case where the model cannot tell the difference for
        itself: a 240x160 screenshot is not enough to be sure a box is there,
        and six lines of "press a -> pressed A" in the history read exactly
        like a long conversation going well. That is how a player ends up
        pressing A at an empty gym floor for twenty turns, each turn more
        convinced by its own record. So report the effect, and be blunt when
        there wasn't one.
        """
        after = self._worldMark()
        if before is None or after is None:
            return ""
        if after["battle"] and not before["battle"]:
            return " - a battle started"
        if before["battle"] and not after["battle"]:
            return " - the battle is over; you are back in the world"
        if after["battle"]:
            # Battle text boxes live outside where the pixel check looks, so
            # "nothing responded" would be a guess dressed up as a fact. In a
            # battle a press that looks like nothing is usually an animation
            # still playing.
            return ""
        if after["dialog"]:
            # A conversation advances and eventually ends; a menu (PC, shop,
            # options) cycles under A forever. After enough presses with a box
            # still up, say so - B backs out of menus and merely speeds up
            # dialog, so it is safe advice either way.
            streak = getattr(self, "_boxStreak", 0) + 1
            self._boxStreak = streak
            if streak >= 4:
                return (f" - a box is STILL on screen after {streak} presses. "
                        "If the text is not changing, this is a MENU, not "
                        "dialog: A goes deeper in, B backs out. Press b "
                        "until you can walk again")
            return " - there is a text box on screen now; keep pressing A"
        self._boxStreak = 0
        if before["dialog"]:
            return " - the text box is gone; you can walk again"
        if after["ram"] != before["ram"]:
            return " - you moved"
        return (" - nothing responded. There is no text box on screen and "
                "nothing in front of you to talk to. Pressing A again will do "
                "the same nothing; walk somewhere else instead")

    def move(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("move needs a direction, e.g. `move up 3`")
        direction = DIR_ALIASES.get(args[0].lower())
        if direction is None:
            raise ActionError(f"{args[0]!r} is not a direction (up/down/left/right)")
        steps = 1
        if len(args) > 1:
            match = re.search(r"\d+", args[1])
            if match:
                steps = max(1, min(20, int(match.group())))

        walked, outcome = 0, "moved"
        for _ in range(steps):
            outcome = self.nav._step(direction)
            if outcome == "blocked":
                break
            walked += 1

        where = self._whereNow()
        if outcome == "blocked":
            return (f"walked {walked} of {steps} tile(s) {direction}, then hit "
                    f"something solid. {where}")
        return f"walked {walked} tile(s) {direction}. {where}"

    def _whereNow(self) -> str:
        fix = self.nav.locate()
        if fix is None:
            return "Location unknown now (battle, dialog or a fade)."
        return f"Now on {fix['mapName']} at tile {tuple(fix['tile'])}."

    def goto(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("goto needs a place, e.g. `goto pokemon center`")
        name = " ".join(args)
        obs = self.observation

        # The places listed in this turn's prompt come first: matching what the
        # model was actually shown beats matching the full index, where three
        # different maps own a "Pokemon Center".
        listed = [(e["name"], e) for e in (obs.destinations if obs else [])]
        hit = _bestMatch(name, listed)
        if hit is not None:
            label, entry = hit
            if not entry["found"]:
                raise ActionError(f"{label} is known but not reachable from here "
                                  f"({entry['reason']})")
            result = self.nav.goToTile(entry["map"], entry["tile"],
                                       interact=entry["interact"],
                                       label=f"go to {label}",
                                       maxSteps=self.cfg.moveBudget)
            return self._describeRun(result)

        # Asking for something the objective took off the menu. The name is
        # still in the full index, and the fallback below would happily walk
        # there - which is the exact loop the filtering exists to break - so
        # refuse it here, with the reason the objective gave.
        if obs is not None and obs.hiddenPlaces:
            for pattern in obs.hiddenPlaces:
                if _bestMatch(name, [(pattern, pattern)], cutoff=0.75):
                    raise ActionError(
                        obs.hiddenReason or
                        f"{name} is not somewhere to go for this objective")

        # Not on this turn's list: fall back to the landmark / object index, so
        # somewhere the model remembers from earlier still works.
        landmarks = [(k, k) for k in self.nav.pf.getAvailableLandmarks()]
        objects = [(entryName, entryName)
                   for entries in self.nav.pf.objectIndex.values()
                   for (_m, _c, _r, entryName) in entries]
        hit = _bestMatch(name, landmarks + objects)
        if hit is None:
            raise ActionError(f"I don't know a place called {name!r}. Pick one of "
                              f"the places listed in the report.")
        result = self.nav.goTo(hit[1], maxSteps=self.cfg.moveBudget)
        return self._describeRun(result)

    def heal(self, args: list) -> str:
        self._requireNoDialog()
        # A full-health heal is not harmless: it walks the whole way back to
        # the Center, and an objective hint that says "heal first" reads as
        # eternally applicable, so gym trips turn into laps. Refuse instead.
        try:
            party = [p for p in (self.client.game_state().get("party") or [])
                     if p.get("max_hp")]
        except (MGBAError, ValueError):
            party = []
        BAD = ("psn", "poison", "par", "slp", "sleep", "brn", "burn",
               "frz", "freeze")
        if party and all(p.get("hp", 0) >= p["max_hp"]
                         and not any(k in str(p.get("status") or "").lower()
                                     for k in BAD)
                         for p in party):
            raise ActionError("your party is already at full health - you do "
                              "not need to heal. Carry on with your objective "
                              "instead of walking back to the Center")
        return self._describeRun(self.nav.goHeal(maxSteps=self.cfg.moveBudget))

    def catch(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("catch needs a species, e.g. `catch pidgey`")
        known = [(s, s) for s in self.nav.species()]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no grass is tagged with {' '.join(args)!r}. "
                              f"Known species: {', '.join(s for s, _ in known[:20])}")
        return self._describeRun(self.nav.goCatch(hit[1],
                                                  maxSteps=self.cfg.moveBudget))

    def collect(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("collect needs an item name")
        known = [(k, k) for k in self.nav.pf.itemIndex]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no item ball called {' '.join(args)!r} is mapped")
        return self._describeRun(self.nav.collect(hit[1],
                                                  maxSteps=self.cfg.moveBudget))

    def _describeRun(self, result: dict) -> str:
        """Turn a navigator result dict into one line the model can act on."""
        return (f"{result['goal']}: {result['status']} after "
                f"{result['steps']} step(s) - {result['reason']}")

    # ---- shopping ---------------------------------------------------------

    def _peekU16(self, addr: int) -> int:
        return int.from_bytes(self.client.peek(addr, 2), "little")

    def _peekU32(self, addr: int) -> int:
        return int.from_bytes(self.client.peek(addr, 4), "little")

    def _itemName(self, table: int, itemId: int) -> str:
        return gen3_decode(self.client.peek(table + itemId * ITEM_ENTRY_SIZE,
                                            14)).strip()

    def _dialogText(self) -> str:
        try:
            return (self.client.screen().get("dialog_text") or "").lower()
        except MGBAError:
            return ""

    def buy(self, args: list) -> str:
        """Drive the mart: clerk -> BUY -> item -> quantity -> confirm -> out.

        Every hop is verified against sShopData rather than counted out in
        button presses, so an eaten tap surfaces as an honest error instead
        of a mystery purchase.
        """
        self._requireNoDialog()
        if not args:
            raise ActionError("buy needs an item, e.g. `buy potion 5`")
        qty = 1
        if args[-1].isdigit():
            qty = max(1, min(99, int(args[-1])))
            args = args[:-1]
        wanted = " ".join(args)

        state = self.client.game_state()
        table = ITEM_TABLE_BY_GAME.get(state.get("game", ""))
        if table is None:
            raise ActionError(f"no item table known for "
                              f"{state.get('game', 'this ROM')!r} - add it to "
                              f"ITEM_TABLE_BY_GAME")
        moneyBefore = (state.get("player") or {}).get("money", 0)

        # Clerk greeting -> BUY. The BUY/SELL/QUIT menu is created fresh each
        # time with its cursor on BUY, so A presses walk straight in. Text
        # polling (gStringVar4) is too stale to sequence this; the reliable
        # signal is callback2 - the buy list is its own screen, so the
        # callback leaving the overworld means the list is open.
        overworld = str((self.observation.screen or {}).get("callback2", ""))
        opened = False
        for _ in range(6):
            self._menuTap("A")
            time.sleep(0.9)
            try:
                cb = str(self.client.screen().get("callback2", ""))
            except MGBAError:
                cb = overworld
            if cb and cb != overworld:
                opened = True
                break
        if not opened:
            raise ActionError("no shop opened - stand directly in front of "
                              "the counter, facing the clerk, and try again")
        time.sleep(1.0)              # list fade-in

        # What is for sale, straight from the shop's item list in ROM.
        listPtr = self._peekU32(SHOP_DATA_ADDR + 0x04)
        count = self._peekU16(SHOP_DATA_ADDR + 0x10)
        if not 0x08000000 <= listPtr < 0x0A000000 or not 0 < count <= 32:
            raise ActionError("the buy menu did not open (this may not be a "
                              "shop counter). Press b a few times to back out")
        stock = []
        for i in range(count):
            itemId = self._peekU16(listPtr + i * 2)
            if itemId == 0:
                break
            stock.append((self._itemName(table, itemId), i))
        hit = _bestMatch(wanted, stock)
        if hit is None:
            self._menuTap("B", 3)    # list -> menu -> goodbye
            self._menuTap("A")
            raise ActionError(f"{wanted!r} is not sold here. Stock: "
                              f"{', '.join(n for n, _ in stock)}")
        name, index = hit

        # Walk the list, verified: highlighted index = selectedRow + scroll.
        for _ in range(count + 4):
            at = (self._peekU16(SHOP_DATA_ADDR + 0x0C)
                  + self._peekU16(SHOP_DATA_ADDR + 0x0E))
            if at == index:
                break
            self._menuTap("DOWN" if at < index else "UP")
        else:
            raise ActionError("the shop cursor is not responding - press b "
                              "to back out and try again")

        self._menuTap("A")           # select the item
        # "POTION? Certainly. How many would you like?" types out first, and
        # the spinner eats every press until it has finished - wait for it.
        for _ in range(10):
            time.sleep(0.5)
            if "how many" in self._dialogText():
                break
        time.sleep(0.4)

        maxQty = self._peekU16(SHOP_DATA_ADDR + 0x14)
        target = min(qty, maxQty) if maxQty else qty
        # sShopData.itemPrice is the running total, which makes the spinner
        # verifiable: unit price at x1, then every UP must move the total.
        unit = self._peekU32(SHOP_DATA_ADDR + 0x08)
        for _ in range(3 * target):
            if unit and self._peekU32(SHOP_DATA_ADDR + 0x08) >= unit * target:
                break
            self._menuTap("UP")
            time.sleep(0.2)
        bought = (self._peekU32(SHOP_DATA_ADDR + 0x08) // unit if unit
                  else target)

        self._menuTap("A")           # -> "and you want N. $X. Okay?"
        time.sleep(1.2)
        self._menuTap("A")           # YES
        time.sleep(2.0)
        self._menuTap("A")           # "Here you are! Thank you!"
        time.sleep(1.0)

        # Leave: list -> "anything else?" menu -> "Please come again". B all
        # the way out: B backs out of menus AND dismisses finished text, but
        # never selects - an A here can land on the reopened BUY/SELL menu
        # and start the whole conversation over.
        self._menuTap("B", 2)
        for _ in range(6):
            time.sleep(0.8)
            try:
                if not measureScreen(self.client.screenshot())["open"]:
                    break
            except (MGBAError, ValueError):
                break
            self._menuTap("B")

        # The money HUD (and the RAM behind it) settles a beat after the
        # thank-you line; don't declare failure on a race.
        spent, moneyAfter = 0, moneyBefore
        for _ in range(4):
            moneyAfter = (self.client.game_state().get("player")
                          or {}).get("money", moneyBefore)
            spent = moneyBefore - moneyAfter
            if spent > 0:
                break
            time.sleep(0.8)
        if spent <= 0:
            return (f"walked the menus but no money moved - the purchase "
                    f"did not go through. Check the screen with `press`")
        note = "" if bought == qty else f" (shop capped it at {bought})"
        return (f"bought {bought}x {name} for ${spent}{note}; "
                f"${moneyAfter} left")

    # ---- battle -----------------------------------------------------------

    def use(self, args: list) -> str:
        obs = self._requireBattle()
        if not obs.moveSlots:
            raise ActionError("the battle hasn't loaded your moves yet - "
                              "`wait 1` and look again")
        names = ", ".join(m for m, _i in obs.moveSlots)
        if not args:
            raise ActionError(f"use needs a move: {names}")
        if all(a.isdigit() for a in args):
            # The damage table is sorted by expected damage, so its row numbers
            # are not the move's slot in the game menu. Refuse rather than guess
            # and attack with the wrong move.
            raise ActionError(f"name the move rather than numbering it - "
                              f"one of: {names}")
        hit = _bestMatch(" ".join(args), obs.moveSlots)
        if hit is None:
            raise ActionError(f"{' '.join(args)!r} isn't one of your moves "
                              f"({', '.join(m for m, _i in obs.moveSlots)})")
        name, slot = hit

        self._chooseAction(ACTION_FIGHT)
        time.sleep(0.3)              # let the move menu draw before seeking
        self._cursorSeek(MOVE_CURSOR_ADDR, slot, "move menu")
        self._menuTap("A")
        return f"attacking with {name} (move slot {slot + 1})"

    def switch(self, args: list) -> str:
        obs = self._requireBattle()
        if not args:
            raise ActionError("switch needs a party member, e.g. `switch bulbasaur`")

        party = obs.state.get("party") or []
        table = []
        for i, pk in enumerate(party):
            table.append((pk.get("nickname") or pk.get("species", "?"), i))
            table.append((pk.get("species", "?"), i))
            table.append((str(i + 1), i))
        hit = _bestMatch(" ".join(args), table)
        if hit is None:
            raise ActionError("no party member by that name")
        name, index = hit
        if party[index].get("hp", 0) <= 0:
            raise ActionError(f"{name} has fainted and can't be sent out")

        self._chooseAction(ACTION_POKEMON)
        # The party screen opens on the first slot and DOWN walks the list; the
        # per-Pokemon menu that follows opens on SEND OUT / SHIFT.
        self._menuTap("DOWN", index)
        self._menuTap("A")
        self._menuTap("A")
        return (f"switching to {name} (party slot {index + 1}) - check the next "
                f"screenshot, the party menu may still be open")

    def bag(self, args: list) -> str:
        self._requireBattle()
        self._chooseAction(ACTION_BAG)
        return ("opened the bag in battle. Use `press` with up/down/a to pick an "
                "item, or `press b` to back out.")

    def run(self, args: list) -> str:
        self._requireBattle()
        self._chooseAction(ACTION_RUN)
        return "tried to run from the battle"

    def throw(self, args: list) -> str:
        """BAG -> ball pocket -> ball -> USE, every hop read back from RAM."""
        obs = self._requireBattle()
        if self._wildBattle() is False:
            raise ActionError("you can't catch a trainer's Pokemon - balls "
                              "only work on wild ones")
        balls = (obs.state.get("bag") or {}).get("poke_balls") or []
        balls = [b for b in balls if b.get("quantity", 0) > 0]
        if not balls:
            raise ActionError("no Poke Balls in the bag - buy some at a "
                              "Poke Mart first")
        index = 0
        if args:
            hit = _bestMatch(" ".join(args),
                             [(b.get("name", ""), i)
                              for i, b in enumerate(balls)])
            if hit is None:
                raise ActionError(f"no {' '.join(args)!r} in the ball "
                                  f"pocket. You have: "
                                  f"{', '.join(b['name'] for b in balls)}")
            index = hit[1]
        name = balls[index].get("name", "ball")

        overworldCb = str((obs.screen or {}).get("callback2", ""))
        self._chooseAction(ACTION_BAG)
        for _ in range(12):                 # bag fade-in
            time.sleep(0.5)
            try:
                cb = str(self.client.screen().get("callback2", ""))
            except MGBAError:
                continue
            if cb and cb != overworldCb:
                break
        else:
            raise ActionError("the bag never opened - `wait 1` and try again")
        time.sleep(0.8)

        # Pocket, then item row - both verified against gBagMenuState.
        for _ in range(6):
            pocket = self._peekU16(BAG_STATE_ADDR + 0x06)
            if pocket == BALL_POCKET:
                break
            self._menuTap("RIGHT" if pocket < BALL_POCKET else "LEFT")
            time.sleep(0.5)
        else:
            self._menuTap("B", 2)
            raise ActionError("could not reach the POKE BALLS pocket - "
                              "backed out of the bag")
        for _ in range(3 * max(1, index)):
            at = (self._peekU16(BAG_STATE_ADDR + 0x08 + 2 * BALL_POCKET)
                  + self._peekU16(BAG_STATE_ADDR + 0x0E + 2 * BALL_POCKET))
            if at == index:
                break
            self._menuTap("DOWN" if at < index else "UP")
            time.sleep(0.25)

        self._menuTap("A")                  # select the ball
        time.sleep(0.8)
        self._menuTap("A")                  # USE (default cursor position)
        return (f"threw a {name}! Watch the next screenshot: if it escapes "
                f"the ball, pick a move; if the naming keyboard appears, it "
                f"was CAUGHT - `name` it or press b to skip naming")

    # ---- move learning -----------------------------------------------------

    def _battleText(self) -> str:
        """What the battle text box is drawing right now."""
        try:
            return gen3_decode_text(
                self.client.peek(DISPLAYED_STRING_ADDR, 200))
        except (MGBAError, ValueError):
            return ""

    def _moveName(self, state: dict, moveId: int) -> str:
        table = MOVE_TABLE_BY_GAME.get(state.get("game", ""))
        if not table or not 0 < moveId < 512:
            return ""
        try:
            return gen3_decode(
                self.client.peek(table + MOVE_NAME_SIZE * moveId,
                                 MOVE_NAME_SIZE)).strip()
        except (MGBAError, ValueError):
            return ""

    def _learnPending(self, state: dict):
        """(mon, new move name) while a learn prompt is up, else None."""
        text = self._battleText()
        low = text.lower()
        if not any(k in low for k in LEARN_PROMPT_MARKERS):
            return None
        newName = self._moveName(state, self._peekU16(MOVE_TO_LEARN_ADDR))
        party = state.get("party") or []
        mon = None
        named = re.match(r"\s*([^\n]+?) is trying to learn", text)
        if named:
            nick = named.group(1).strip().lower()
            mon = next((p for p in party
                        if str(p.get("nickname") or "").lower() == nick
                        or str(p.get("species") or "").lower() == nick), None)
        if mon is None:
            # later prompts in the chain don't repeat the name; the learner is
            # the healthy party member already holding four moves
            mon = next((p for p in party
                        if len(p.get("moves") or []) >= 4
                        and p.get("hp", 0) > 0), None)
        return mon, newName

    def _forgetChoice(self, mon: dict, newName: str):
        """Which slot to sacrifice: (slot, old name, why)."""
        def info(name):
            mv = self.battle.resolve(name)
            return (getattr(mv, "power", 0) or 0,
                    str(getattr(mv, "type", "") or ""))
        moves = [str(m.get("name") or "")
                 for m in (mon.get("moves") or [])][:4]
        scored = [(i, n, *info(n)) for i, n in enumerate(moves) if n]
        if not scored:
            return None
        newPower, newType = info(newName)
        if newPower:
            outclassed = [s for s in scored
                          if s[3] == newType and 0 < s[2] < newPower]
            if outclassed:
                i, n, _, _ = min(outclassed, key=lambda s: s[2])
                return i, n, f"{newName} is a stronger {newType} attack"
        status = [s for s in scored if s[2] == 0]
        if status:
            i, n, _, _ = status[0]
            return i, n, f"{n} does no damage"
        i, n, _, _ = min(scored, key=lambda s: s[2])
        return i, n, f"{n} is the weakest attack"

    def learn(self, args: list) -> str:
        """Drive the four-moves learn prompt end to end.

        A through the yes/no chain until the move-forget screen replaces the
        battle callback, pick the sacrificial slot there, A back out through
        the poof dialog - then reread the party, which is the only answer
        that counts.
        """
        obs = self._requireBattle()
        state = obs.state
        pending = self._learnPending(state)
        if pending is None:
            raise ActionError("no move is waiting to be learned right now")
        mon, newName = pending

        if args and args[0].lower() in ("skip", "no", "refuse", "stop"):
            self._menuTap("B")
            time.sleep(0.8)
            self._menuTap("A")              # "Stop learning X?" -> YES
            time.sleep(0.8)
            self._menuTap("A")
            return f"gave up on learning {newName or 'the new move'}"

        if mon is None:
            raise ActionError("could not tell which Pokemon is learning - "
                              "answer the prompts with `press` instead")
        choice = self._forgetChoice(mon, newName)
        if choice is None:
            raise ActionError("could not read that Pokemon's moves - "
                              "answer the prompts with `press` instead")
        slot, oldName, why = choice
        monName = mon.get("nickname") or mon.get("species") or "your Pokemon"
        partyIndex = (state.get("party") or []).index(mon)

        switched = False
        for _ in range(10):
            try:
                cb = str(self.client.screen().get("callback2", "")).upper()
            except MGBAError:
                cb = ""
            if cb and cb not in BATTLE_SCREEN_CALLBACKS:
                switched = True
                break
            self._menuTap("A")
            time.sleep(0.8)
        if not switched:
            raise ActionError("the move-forget screen never opened - answer "
                              "the prompts with `press a` instead")

        time.sleep(1.5)                     # summary screen fade-in
        self._menuTap("DOWN", slot)
        time.sleep(0.3)
        self._menuTap("A")
        time.sleep(1.0)
        for _ in range(14):                 # poof + learned dialog
            try:
                cb = str(self.client.screen().get("callback2", "")).upper()
            except MGBAError:
                cb = ""
            if (cb in BATTLE_SCREEN_CALLBACKS
                    and "learned" in self._battleText().lower()):
                break
            self._menuTap("A")
            time.sleep(0.8)

        try:
            fresh = (self.client.game_state().get("party") or [])[partyIndex]
            have = [str(m.get("name") or "").lower()
                    for m in fresh.get("moves") or []]
        except (MGBAError, ValueError, IndexError):
            have = []
        if newName and newName.lower() in have:
            return (f"{monName} forgot {oldName} and learned {newName} "
                    f"({why})")
        return (f"walked through the menus to learn {newName or 'the move'} "
                f"over {oldName}, but the party does not show it yet - look "
                f"at the screen and finish any dialog with `press a`")

    # ---- memory and planning ----------------------------------------------

    def note(self, args: list) -> str:
        if self.memory is None:
            raise ActionError("memory is disabled for this run")
        text = " ".join(args).strip()
        if len(text) < 4:
            raise ActionError("write a whole sentence worth remembering, "
                              "e.g. `note the gym door is on the west side`")
        self.memory.note(text, self.turn)
        return f"wrote it down: \"{text}\""

    def check(self, args: list) -> str:
        if self.roster is None or self.data is None:
            raise ActionError("the battle calculator isn't loaded")
        known = [(t, t) for t in self.roster.ids()]
        if not known:
            raise ActionError("no trainer teams have been recorded yet")
        if not args:
            raise ActionError(f"check needs a trainer: "
                              f"{', '.join(t for t, _ in known)}")
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"I have no notes on {' '.join(args)!r}. "
                              f"Known: {', '.join(t for t, _ in known)}")
        trainerId = hit[1]
        obs = self.observation
        party = (obs.state.get("party") if obs else None) or []
        team = self.roster.team(trainerId)
        report = assess(self.data, party, team)
        entry = self.roster.get(trainerId) or {}
        levels = (levelsNeeded(self.data, party, team)
                  if report["verdict"] != "ready" else None)
        return describeReadiness(report, entry.get("name") or trainerId, levels)

    # ---- the naming keyboard ----------------------------------------------

    def name(self, args: list) -> str:
        """Hand a name to the keyboard driver, which types and confirms it."""
        obs = self.observation
        if obs is None or not obs.namingOpen:
            raise ActionError("nothing is asking for a name right now")
        wanted = " ".join(args).strip()
        if not wanted:
            raise ActionError("name needs the name to type, e.g. `name sprout`")
        try:
            result = Keyboard(self.client).enter(wanted)
        except NamingError as exc:
            raise ActionError(str(exc))
        note = f" ({result['note']})" if result["note"] else ""
        if not result["confirmed"]:
            return (f"typed {result['name']!r}{note}, but the keyboard is still "
                    f"open - check the screen and press A to accept it")
        return f"named it {result['name']!r}{note} and confirmed"

    # ---- misc -------------------------------------------------------------

    def wait(self, args: list) -> str:
        seconds = 1.0
        if args:
            match = re.search(r"\d+(\.\d+)?", args[0])
            if match:
                seconds = max(0.1, min(5.0, float(match.group())))
        time.sleep(seconds)
        return f"waited {seconds:g}s"

    # ---- dispatch ---------------------------------------------------------

    def execute(self, verb: str, args: list) -> str:
        method = getattr(self, verb, None)
        if method is None:
            raise ActionError(f"unknown command {verb!r}")
        return method(args)


# --------------------------------------------------------------------------
# Observation: everything the model gets to see this turn
# --------------------------------------------------------------------------


@dataclass
class Observation:
    turn: int
    state: dict                         # full GAME_STATE
    fix: dict | None                    # locationTracker fix, None if unknown
    screen: dict = field(default_factory=dict)
    inBattle: bool = False
    battleReport: str = ""
    recommendation: str = ""
    moveSlots: list = field(default_factory=list)   # [(moveName, slotIndex)]
    destinations: list = field(default_factory=list)
    screenshotPath: Path | None = None
    note: str = ""                      # harness-level warnings for the model
    objectiveText: str = ""             # from objectives.renderObjective
    hiddenPlaces: list = field(default_factory=list)   # patterns this objective hides
    hiddenReason: str = ""              # why, told to the model if it asks anyway
    hidden: int = 0                     # how many destinations were filtered out
    namingOpen: bool = False            # the keyboard screen is asking for a name
    namingSoFar: str = ""               # what is already typed into it
    liveRequest: str = ""               # a short-term ask from the operator GUI
    dialogOpen: bool = False            # a message box is covering the screen
    dialogText: str = ""                # what it says, if gStringVar4 is known
    dialogDoubted: bool = False         # asserted too long without anything moving
    noDialogNotice: bool = False        # pressing A at nothing; say so outright
    lostOnMap: bool = False             # our tile is one nobody could stand on

    # ---- rendering --------------------------------------------------------

    def render(self, cfg: Config, history: list) -> str:
        blocks = [f"=== TURN {self.turn} ==="]
        if cfg.goal:
            blocks.append(f"WHAT YOUR OPERATOR ASKED FOR: {cfg.goal}")
        if self.liveRequest:
            blocks.append(self._liveRequestBlock())
        if self.objectiveText:
            blocks.append(self.objectiveText)
        # Before anything else about the world: if a box or a keyboard is up,
        # none of the world matters this turn.
        if self.namingOpen:
            blocks.append(self._namingBlock())
        elif self.dialogOpen:
            blocks.append(self._dialogBlock())
        elif self.noDialogNotice:
            blocks.append(self._noDialogBlock())
        blocks.append(self._situation())
        if self.inBattle:
            blocks.append(self.battleReport)
            if self.recommendation:
                blocks.append(self.recommendation)
            # The party matters in battle too - `switch` is only a real option
            # if the model can see what else is alive and how healthy it is.
            blocks.append(self._party())
        else:
            blocks.append(self._party())
            # No point listing places to walk to in the same breath as
            # refusing to walk - that is the contradiction the model would
            # resolve by walking.
            if cfg.showDestinations and not self._walkingBlocked():
                blocks.append(self._places(cfg.destinationLimit))
        if self.note:
            blocks.append(f"NOTE: {self.note}")
        if history:
            blocks.append(self._history(history, cfg.historyLength))
        blocks.append(self._commandHelp())
        return "\n\n".join(b for b in blocks if b)

    def _liveRequestBlock(self) -> str:
        return "\n".join([
            "SOMETHING JUST CAME IN",
            f'  "{self.liveRequest}"',
            "  This is more urgent than the objective below - work on it now. "
            "If it isn't possible from where you are, say why with `note` and "
            "go back to the objective."])

    def _walkingBlocked(self) -> bool:
        return self.namingOpen or (self.dialogOpen and not self.dialogDoubted)

    def _namingBlock(self) -> str:
        lines = ["THE GAME IS ASKING YOU TO NAME SOMETHING.",
                 "  An on-screen keyboard is up. Do NOT try to walk or to press "
                 "letters one at a time - that is how a Pokemon ends up called "
                 "FFFF."]
        if self.namingSoFar:
            lines.append(f'  Typed so far (it will be cleared first): '
                         f'"{self.namingSoFar}"')
        lines.append("  Answer with `name <what to call it>`, using 1-10 "
                     "letters, and the keyboard will be typed and confirmed for "
                     "you. A short, memorable name is best.")
        return "\n".join(lines)

    def _dialogBlock(self) -> str:
        # Once the box is doubted this block has to change its story completely,
        # not soften it. Saying "a box is open, you cannot walk" and then listing
        # `goto` underneath leaves the model to pick which half to believe, and
        # it picks the prose - which is how a misread costs a dozen turns.
        if self.dialogDoubted:
            return ("\n".join([
                "THERE IS PROBABLY NO TEXT BOX ON SCREEN.",
                "  Something down there looked like one, but pressing A has "
                "changed nothing for several turns, so it was most likely "
                "scenery. Stop pressing A.",
                "  Treat this as an ordinary turn: walk, use `goto`, or explore. "
                "Everything below is real."]))

        lines = ["A TEXT BOX IS OPEN ON SCREEN."]
        if self.dialogText:
            flat = " ".join(self.dialogText.split())
            lines.append(f'  It says: "{flat[:400]}"')
            lines.append("  (That text is the last message the game wrote, which "
                         "it keeps after a box closes - so it may be older than "
                         "what is on screen.)")
        lines.append("  While text is on screen the game ignores every button "
                     "except A and B. You cannot walk anywhere, and no amount "
                     "of moving will change that.")
        lines.append("  Press A to advance the text. Long conversations take "
                     "several presses; keep going until the box is gone.")
        return "\n".join(lines)

    def _noDialogBlock(self) -> str:
        """Contradict the hallucination outright, once it has started.

        Saying nothing about dialog is not the same as saying there is none.
        The model is looking at a 240x160 screenshot with a strong prior that
        a gym contains a trainer who talks, and silence in the report leaves
        that prior unopposed - it will narrate a conversation nobody is
        having and press A at it indefinitely. Only shown once the pressing
        has actually started, so an ordinary turn is not cluttered with a
        denial of something nobody suggested.
        """
        return "\n".join([
            "THERE IS NO TEXT BOX ON SCREEN.",
            "  You pressed A and nothing answered. Whatever the screenshot "
            "looks like, nobody is talking to you: there is no conversation "
            "to advance and no menu waiting on you.",
            "  Pressing A again will do the same nothing. To talk to someone "
            "you have to be standing next to them and facing them - use "
            "`goto` to walk to them by name, which handles that for you."])

    def _situation(self) -> str:
        p = self.state.get("player", {})
        lines = ["WHERE YOU ARE"]
        if self.inBattle:
            lines.append("  You are in a BATTLE. The overworld map is not visible.")
        elif self.fix is not None:
            lines.append(f"  Map:   {self.fix['mapName']}  "
                         f"tile {tuple(self.fix['tile'])}")
        else:
            lines.append("  Map:   unknown - a dialog box, a cutscene or a "
                         "screen transition is covering the map.")
        lines.append(f"  RAM:   position ({p.get('x')}, {p.get('y')})  "
                     f"map id ({p.get('map_bank')}, {p.get('map_number')})")
        lines.append(f"  Money: ${p.get('money', 0):,}   Badges: {p.get('badges', 0)}")

        # The message text deliberately isn't repeated here: gStringVar4 keeps
        # the last thing said long after its box has closed, so quoting it
        # unconditionally would put words on screen that aren't there. When a
        # box really is up, the block above has already shown them.
        return "\n".join(lines)

    def _party(self) -> str:
        party = self.state.get("party") or []
        if not party:
            return "YOUR PARTY\n  (empty)"
        lines = ["YOUR PARTY"]
        for i, pk in enumerate(party, 1):
            status = "" if pk.get("status") == "OK" else f" [{pk.get('status')}]"
            pct = 100 * pk.get("hp", 0) / max(1, pk.get("max_hp", 1))
            types = pk.get("type1", "?")
            if pk.get("type2") and pk["type2"] != pk.get("type1"):
                types += f"/{pk['type2']}"
            moves = ", ".join(f"{m['name']}({m['pp']}pp)" for m in pk.get("moves", []))
            lines.append(f"  {i}. {pk.get('nickname')} ({pk.get('species')}) "
                         f"Lv{pk.get('level')}  HP {pk.get('hp')}/{pk.get('max_hp')} "
                         f"({pct:.0f}%)  {types}{status}")
            if moves:
                lines.append(f"     moves: {moves}")
        return "\n".join(lines)

    def _places(self, limit: int) -> str:
        # A lost fix has to be named as one. "Nowhere is reachable" is a
        # statement about the map; "I don't know where you are" is a statement
        # about the harness, and only the second one suggests the fix, which is
        # to walk a few tiles somewhere the map can be recognised from.
        if self.lostOnMap:
            return ("PLACES YOU CAN WALK TO\n"
                    "  I have lost track of exactly where you are on this map, "
                    "so I can't route you anywhere yet.\n"
                    "  Use `move` a few tiles - away from doorways and the edge "
                    "of the map - and it should sort itself out. `goto` will "
                    "work again once it does.")

        reachable = [e for e in self.destinations if e["found"]]
        if not reachable:
            body = ("PLACES YOU CAN WALK TO\n  (none reachable from here - use "
                    "`move` to explore)")
        else:
            lines = ["PLACES YOU CAN WALK TO  (command: goto <name>)"]
            for e in reachable[:limit]:
                lines.append(f"  {e['steps']:>4} steps  {e['name']}  "
                             f"[{e['category']}] on {e['map']}")
            body = "\n".join(lines)
        # Say that something was withheld and why. Silently shortening the list
        # would leave the model wondering where the lab went and arguing with
        # its own memory; one sentence turns a missing option into information.
        if self.hidden and self.hiddenReason:
            body += f"\n  Not listed right now: {self.hiddenReason}"
        return body

    def _history(self, history: list, limit: int) -> str:
        lines = ["WHAT YOU JUST DID"]
        for entry in history[-limit:]:
            # `check` answers with a whole report; the history only needs the
            # gist of it, and the full text was already shown the turn it ran.
            result = " ".join(str(entry["result"]).split())
            if len(result) > 180:
                result = result[:177] + "..."
            lines.append(f"  turn {entry['turn']}: {entry['action']} -> {result}")
        return "\n".join(lines)

    def _commandHelp(self) -> str:
        # With a box up, the only legal context is the one every command shares
        # ('any'), which leaves press/wait/note/check and takes walking off the
        # menu entirely. Telling a model not to walk is weaker than not
        # offering it - the same reason objectives hide misleading places.
        if self.namingOpen:
            context = "naming"
        elif self.dialogOpen and not self.dialogDoubted:
            context = "dialog"
        elif self.inBattle:
            context = "battle"
        else:
            context = "overworld"
        lines = ["COMMANDS YOU CAN USE RIGHT NOW"]
        for cmd in COMMANDS:
            if cmd.context in ("any", context):
                lines.append(f"  {cmd.usage:<34} {cmd.help}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Brain: the ollama side, and the parser that makes text into a command
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing Pokemon Leaf Green on a Game Boy Advance.

Each turn you get a screenshot and a written report of the game state, and you
choose exactly ONE action. Tools handle the hard parts for you: `goto` walks
whole routes, and the battle table already tells you what each move will do.
Prefer those over pressing buttons one at a time.

You are always working towards the objective at the top of the report. It is
marked done automatically when the game says so, so you never need to claim it
is finished - just work on it. If you have tried the same thing several times
and the report has not changed, that approach is not working: try a different
one, and `note` what you learned so you do not repeat it.

Answer in exactly this format and nothing else:

THINK: <one short sentence about what you are doing and why>
ACTION: <one command from the list you were given>

Examples of well-formed answers:

THINK: The nurse can heal my hurt party, so I will walk to the Pokemon Center.
ACTION: goto pokemon center

THINK: Ember is super effective and should knock it out this turn.
ACTION: use ember

THINK: There is a text box on screen, so I need to advance it.
ACTION: press a
"""

# Anchored on the ACTION: line, but tolerant of the wrappers small models add
# around it (**ACTION:**, "action -", a bullet, a trailing period).
ACTION_RE = re.compile(r"^[\s>*\-•]*action\s*[:\-]?\s*(.+)$",
                       re.IGNORECASE | re.MULTILINE)
CLEAN_RE = re.compile(r"^[\s`*_\"'\[(]+|[\s`*_\"'.!\])]+$")


def parseCommand(reply: str):
    """Pull (verb, args) out of a model reply, or return None.

    Three passes, most trustworthy first: the ACTION: line the format asks for,
    a JSON object in case the model reached for tool-calling on its own, then
    any line whose first word is a verb we know - which catches the very common
    "it just answered `press a`" case.
    """
    if not reply:
        return None

    matches = ACTION_RE.findall(reply)
    for raw in reversed(matches):
        parsed = _parseLine(raw)
        if parsed:
            return parsed

    for blob in re.findall(r"\{[^{}]*\}", reply):
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        verb = data.get("action") or data.get("command") or data.get("name")
        if verb:
            argument = data.get("argument") or data.get("arg") or data.get("target") or ""
            parsed = _parseLine(f"{verb} {argument}")
            if parsed:
                return parsed

    for line in reversed([ln for ln in reply.splitlines() if ln.strip()]):
        parsed = _parseLine(line)
        if parsed:
            return parsed
    return None


def _parseLine(line: str):
    """One candidate line -> (verb, args), if it starts with a command we know."""
    text = CLEAN_RE.sub("", line.strip())
    text = re.sub(r"^(?:think|reasoning|thought)\s*[:\-].*$", "", text,
                  flags=re.IGNORECASE).strip()
    if not text:
        return None
    tokens = [t for t in re.split(r"[\s,]+", text) if t]
    if not tokens:
        return None

    head = _norm(tokens[0])
    verb = VERB_LOOKUP.get(head)

    # A bare button or direction is an action even without a verb, and models
    # answer that way constantly.
    if verb is None and head in DIR_ALIASES:
        return "move", [DIR_ALIASES[head]] + tokens[1:]
    if verb is None and head.upper() in BUTTONS:
        return "press", tokens
    if verb is None:
        close = difflib.get_close_matches(head, list(VERB_LOOKUP), n=1, cutoff=0.8)
        verb = VERB_LOOKUP[close[0]] if close else None
    if verb is None:
        return None

    args = [CLEAN_RE.sub("", t) for t in tokens[1:]]
    args = [a for a in args if a and _norm(a) not in ("to", "the", "at", "with")]

    # `move`/`go` reads as `goto` when what follows is a place, not a direction.
    if verb == "move" and args and _norm(args[0]) not in DIR_ALIASES:
        verb = "goto"
    return verb, args


class Brain:
    """One model call per turn, plus the retries that get a parseable answer."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = ollama.Client(host=cfg.ollamaHost) if cfg.ollamaHost else ollama
        self.lastReply = ""
        # Cleared the first time the server rejects `think` - not every model
        # accepts the parameter, and the harness is meant to be model-agnostic.
        self._supportsThink = True

    def preload(self):
        """Pay the model's load time once, at startup, not on turn one."""
        print(f"Loading {self.cfg.model} ...")
        started = time.time()
        reply = self._chat([{"role": "user", "content":
                             "Reply with exactly: ready"}])
        print(f"  model says: {reply.strip()[:60]}  ({time.time() - started:.1f}s)")

    def decide(self, report: str, imagePath: Path | None):
        """Ask for a command. Returns (verb, args, rawReply) - verb may be None."""
        user = {"role": "user",
                "content": report + "\n\nWhat is your next action?"}
        if imagePath is not None and self.cfg.sendImage and imagePath.exists():
            user["images"] = [str(imagePath)]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, user]
        for attempt in range(self.cfg.parseRetries + 1):
            reply = self._chat(messages)
            self.lastReply = reply
            parsed = parseCommand(reply)
            if parsed is not None:
                return parsed[0], parsed[1], reply
            if attempt < self.cfg.parseRetries:
                print("  (reply had no usable ACTION line, re-asking)")
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                     "That reply had no usable ACTION line. Answer with only "
                     "one line, in the form:\nACTION: <command>"},
                ]
        return None, [], self.lastReply

    def _chat(self, messages: list) -> str:
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            keep_alive=self.cfg.keepAlive,
            options={"temperature": self.cfg.temperature,
                     "num_predict": self.cfg.numPredict},
        )
        if self._supportsThink:
            kwargs["think"] = self.cfg.think

        try:
            message = self.client.chat(**kwargs)["message"]
        except (ollama.ResponseError, TypeError) as exc:
            if not self._supportsThink or "think" not in str(exc).lower():
                raise
            print("  (this model doesn't take a `think` setting; continuing "
                  "without it)")
            self._supportsThink = False
            kwargs.pop("think")
            message = self.client.chat(**kwargs)["message"]

        content = (message.get("content") or "").strip()
        # A thinking model that runs out of budget mid-thought answers with an
        # empty content field. The command is usually sitting in the reasoning
        # anyway, so read that rather than throw the turn away.
        return content or (message.get("thinking") or "")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class PlayerAI:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"Connecting to mGBA at {cfg.mgbaHost}:{cfg.mgbaPort} ...")
        self.client = MGBAClient(host=cfg.mgbaHost, port=cfg.mgbaPort)
        self.client.ping()

        # One client, shared by everything: the navigator's step verification,
        # the battle snapshot and our own screenshots all talk over the same
        # socket, so nothing can observe a different frame than anything else.
        self.nav = Navigator(client=self.client,
                             screenshotPath=str(cfg.screenshotPath))
        self.battle = Session(data=GameData.load())
        self.roster = Roster.load()
        self.brain = Brain(cfg)

        self.book = (ObjectiveBook.load(cfg.objectivesPath)
                     if cfg.useObjectives else ObjectiveBook())
        self.memory = Memory.load(cfg.memoriesPath)
        self.actions = Actions(self.nav, cfg, memory=self.memory,
                               roster=self.roster, data=self.battle.data,
                               battle=self.battle)

        # Turns continue across runs, so "you have been on this objective for
        # 40 turns" survives a restart - which is exactly when it matters.
        self.turn = int(self.memory.data.get("turns_played") or 0)
        self.history = []
        self._readinessCache = {}
        self._dialogStreak = 0
        self._dialogMarker = None
        # Set after a `use`, checked on the next observation: the battle cursor
        # trick is the one assumption in here the game could still surprise us
        # on, so it gets verified against the PP that actually moved.
        self._pendingMove = None
        self._loadAddresses()

        # Set by main() when running in `gui` mode: an operator_inbox.OperatorInbox
        # shared with the Tkinter console. None everywhere else, so run()/observe()
        # behave exactly as before when nothing is watching.
        self.inbox = None
        # The last report rendered in step(), so the feasibility worker can
        # judge a request against real game state without touching the mGBA
        # socket from its own thread (that socket belongs to this one).
        self.lastReport = ""

    def close(self):
        self.nav.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, exc, tb):
        self.close()
        return False

    def _loadAddresses(self):
        """Register gStringVar4/gTasks if discover.py has found them before.

        Optional: without them SCREEN just omits dialog_text, and the model
        falls back to reading the text box off the screenshot.
        """
        path = HERE / "mGBA" / "addresses.json"
        if not path.exists():
            print("No mGBA/addresses.json - dialog text won't be in the report. "
                  "Run `python mGBA/discover.py` to add it.")
            return
        try:
            self.client.load_addrs(json.loads(path.read_text(encoding="utf-8")))
            print(f"Registered saved addresses from {path.name}")
        except (ValueError, MGBAError, OSError) as exc:
            print(f"Could not register saved addresses: {exc}")

    # ---- observation ------------------------------------------------------

    def observe(self) -> Observation:
        """Screenshot, game state, location, and the battle table if we're in one."""
        self.client.screenshot(str(self.cfg.screenshotPath))
        state = self.client.game_state()

        screen = {}
        try:
            screen = self.client.screen()
        except MGBAError:
            pass   # SCREEN needs a GBA game loaded; not worth failing a turn over

        inBattle = bool(state.get("in_battle"))
        obs = Observation(turn=self.turn, state=state, fix=None, screen=screen,
                          inBattle=inBattle,
                          screenshotPath=self.cfg.screenshotPath)
        # The keyboard screen has its own callback2, so unlike a dialog it can
        # be recognised outright rather than inferred from pixels.
        obs.namingOpen = namingScreenOpen(screen)
        if obs.namingOpen:
            obs.namingSoFar = currentName(self.client) or ""
        self._checkDialog(obs, state, screen, inBattle)

        if self.inbox is not None:
            request = self.inbox.request
            obs.liveRequest = request.text if request is not None else ""

        # Objectives first: the current one decides which destinations are worth
        # showing, so it has to be settled before the survey is filtered.
        obs.objectiveText = self._objectiveBlock(state)
        objective = (self.book.current(self.memory)
                     if self.cfg.useObjectives else None)

        if inBattle:
            self._fillBattle(obs, state)
        else:
            # observe() is the navigator's own fix: RAM map id first, template
            # match only when that map isn't registered yet.
            obs.fix, _pos = self.nav.observe()
            obs.lostOnMap = self._lostOnMap(obs.fix)
            if (obs.fix is not None and self.cfg.showDestinations
                    and not obs._walkingBlocked()):
                found = self.nav.nearby(fix=obs.fix, gameState=state)
                found += self._exits(obs.fix, state, found)
                found.sort(key=lambda e: (not e["found"],
                                          e["steps"] if e["found"] else 0,
                                          e["name"].lower()))
                if objective is not None:
                    kept = [e for e in found
                            if not objective.hides(e["name"], e["category"])]
                    obs.hidden = len(found) - len(kept)
                    found = kept
                obs.destinations = found

        if objective is not None:
            obs.hiddenPlaces = objective.hidden
            obs.hiddenReason = objective.hiddenReason

        obs.note = " ".join(n for n in (self._checkPendingMove(state, inBattle),
                                        self._lowHpAlert(state, inBattle),
                                        self._repeatAlert()) if n)
        return obs

    def _lowHpAlert(self, state: dict, inBattle: bool) -> str:
        """Warn when the party can't take a hit, before grass proves it.

        The battle advisor can say `run`, but by then the encounter already
        happened. The losable moment is earlier - walking encounter terrain
        with a party one Tackle from a blackout - and nothing else in the
        report calls that out.
        """
        if inBattle:
            return ""
        party = [p for p in (state.get("party") or []) if p.get("max_hp")]
        if not party:
            return ""
        healthy = [p for p in party if p.get("hp", 0) > 0]
        if not healthy:
            return ""
        poisoned = [p for p in healthy
                    if "poison" in str(p.get("status", "")).lower()
                    or "psn" in str(p.get("status", "")).lower()]
        strongest = max(healthy, key=lambda p: p["hp"] / p["max_hp"])
        share = strongest["hp"] / strongest["max_hp"]
        if poisoned:
            name = (poisoned[0].get("nickname")
                    or poisoned[0].get("species_name") or "Your Pokemon")
            return (f"{name} is POISONED and loses HP every few steps you "
                    f"walk - it can collapse in the overworld. `heal` NOW "
                    f"(a Pokemon Center or your mom cures it), take the "
                    f"shortest path, and do not detour through grass.")
        if share > 0.25:
            return ""
        name = strongest.get("nickname") or strongest.get("species_name") or "your Pokemon"
        alert = (f"{name} is at {strongest['hp']}/{strongest['max_hp']} HP. One "
                 f"more wild battle could end in a blackout - `heal` before "
                 f"walking through grass, and stay out of it on the way.")
        money = (state.get("player") or {}).get("money", 0)
        if not self._healingItems(state) and money >= 300:
            alert += (" You also carry NO healing items - next time you are "
                      "at a Poke Mart, buy a few Potions (300 each).")
        return alert

    @staticmethod
    def _healingItems(state: dict) -> list:
        """HP-restoring items in the bag, best first, as display names."""
        healing = ("MAX POTION", "HYPER POTION", "SUPER POTION", "POTION",
                   "FULL RESTORE", "FRESH WATER", "SODA POP", "LEMONADE",
                   "MOOMOO MILK", "ORAN BERRY", "SITRUS BERRY")
        bag = (state.get("bag") or {}).get("items") or []
        found = {i.get("name", "").upper(): i for i in bag}
        return [n.title() for n in healing if n in found]

    # ---- what is on screen ------------------------------------------------

    def _checkDialog(self, obs: Observation, state: dict, screen: dict,
                     inBattle: bool):
        """Decide whether a message box is up, and how much to trust that.

        Two sources, and neither is sufficient alone. The pixels say whether a
        box is *drawn*; gStringVar4 says what the last message *was*, and keeps
        saying it long after the box has closed - so the text is only reported
        when the picture agrees there is something to read.

        Battles are skipped: their own report already describes the screen, and
        the split message/menu row down there doesn't look like a plain box.
        """
        if inBattle or obs.namingOpen or not self.cfg.detectDialog:
            self._dialogStreak = 0
            return

        reading = measureScreen(str(self.cfg.screenshotPath))
        obs.dialogOpen = reading["open"]
        if not obs.dialogOpen:
            self._dialogStreak = 0
            # No box, and the last thing we did was press A at one anyway. The
            # model is arguing with the screenshot, so answer it directly.
            obs.noDialogNotice = self._lastActionWasAdvance()
            return

        obs.dialogText = (screen or {}).get("dialog_text", "") or ""

        # The trust window. A conversation that is going somewhere redraws its
        # box on every press, and ends by letting the player move again. If
        # neither the box nor the player has changed after several presses, the
        # harness is probably arguing with a wall - so it stops insisting.
        #
        # Only turns that actually pressed A or B count towards that. A box that
        # sits unchanged while the model writes notes has not failed to advance;
        # it has not been asked to, and holding it against the box would doubt a
        # real conversation for the crime of being talked over.
        marker = (reading["fingerprint"], self._ramSignature(state))
        if marker != getattr(self, "_dialogMarker", None):
            self._dialogStreak = 0
            self._dialogMarker = marker
        elif self._lastActionWasAdvance():
            self._dialogStreak += 1
        obs.dialogDoubted = self._dialogStreak > self.cfg.dialogTrustTurns

    def _lastActionWasAdvance(self) -> bool:
        """Did last turn press one of the two buttons a text box listens to?"""
        if not self.history:
            return False
        parts = str(self.history[-1].get("action", "")).lower().split()
        return len(parts) >= 2 and parts[0] == "press" and parts[1] in ("a", "b")

    def _lostOnMap(self, fix) -> bool:
        """Is the tile we think we're on one the player could not be standing on?

        Off the edge of the map rip, or inside a wall. Both mean the same thing:
        the RAM->image offset for this map is wrong, so every tile the planner
        reasons about is displaced. Routing quietly stops working - `nearby`
        finds nothing walkable, `goto` has nowhere to start - and from the
        outside that is indistinguishable from standing in a dead end, which is
        a thing a model will happily accept and keep pressing buttons about.
        """
        if fix is None:
            return False
        info = self.nav.pf.tileData.get(fix["mapName"])
        if not info:
            return False
        col, row = fix["tile"]
        if not (0 <= row < info["heightTiles"] and 0 <= col < info["widthTiles"]):
            return True
        return info["tiles"][row][col] == BLOCKED

    @staticmethod
    def _ramSignature(state: dict) -> tuple:
        p = state.get("player", {}) or {}
        return (p.get("map_bank"), p.get("map_number"), p.get("x"), p.get("y"))

    # ---- destinations -----------------------------------------------------

    def _exits(self, fix: dict, state: dict, existing: list) -> list:
        """The ways out of here, named after where they lead.

        nearby() ranks things that live *on* a map - people, items, doors worth
        interacting with. What it cannot offer is "leave", because the edge of
        Pallet Town is not an object. That gap is why a player told to go north
        to Route 1 walks to Professor Oak's lab instead: the lab was the only
        thing on the menu with a familiar name. Reading the exits out of the
        connection graph puts "Route 1" on the menu, one hop at a time.
        """
        pf = self.nav.pf
        curMap, curTile = fix["mapName"], tuple(fix["tile"])
        caps = self.nav.inferCapabilities(state)

        # Owner map -> the exits it owns. The current map first, then the maps
        # it opens onto, so a bedroom can still offer the town outside.
        owners, frontier = [curMap], [curMap]
        for _hop in range(EXIT_HOPS):
            nextHop = []
            for mapName in frontier:
                for conn in pf.connections.get(mapName, []):
                    target = conn.get("toMap")
                    if target and target != RETURN_TARGET and target not in owners:
                        owners.append(target)
                        nextHop.append(target)
            frontier = nextHop

        taken = {_norm(e["name"]) for e in existing}
        candidates = {}
        for owner in owners:
            for conn in pf.connections.get(owner, []):
                toMap = conn.get("toMap")
                landing = conn.get("toTile")
                if not toMap or not landing or toMap in (RETURN_TARGET, curMap):
                    continue
                name = friendlyMapName(toMap)
                if toMap in candidates or _norm(name) in taken:
                    continue
                # Aim at where the door comes *out*, not at the door itself.
                # A goal on this side of a threshold is reached by standing on
                # the mat, which is not what "go to Pallet Town" means and
                # leaves the walk loop with nothing left to do; a goal on the
                # far side makes the crossing part of the route.
                candidates[toMap] = (toMap, tuple(landing), name)

        results = []
        sink = io.StringIO()
        for toMap, (target, tile, name) in candidates.items():
            with contextlib.redirect_stdout(sink):
                plan = pf.planToTile(curMap, curTile, target, tile,
                                     capabilities=caps,
                                     warpStack=self.nav.warpStack)
            results.append({
                "kind": "exit", "category": EXIT_CATEGORY, "name": name,
                "map": target, "tile": tile, "interact": False,
                "found": plan["found"],
                "steps": len(plan["directions"]) if plan["found"] else None,
                "reason": plan["reason"],
            })
        return results

    # ---- objectives -------------------------------------------------------

    def _objectiveBlock(self, state: dict) -> str:
        """Advance the walkthrough if it's earned, then render where we are."""
        if not self.cfg.useObjectives or not len(self.book):
            return ""
        self.book.sync(state, self.memory, turn=self.turn,
                       trainerReady=lambda tid: self._ready(state, tid))
        objective = self.book.current(self.memory)

        readiness = ""
        if objective is not None and objective.trainer:
            readiness = self._readinessLine(state, objective.trainer)
        return renderObjective(self.book, self.memory, self.turn,
                               readiness=readiness,
                               noteLimit=self.cfg.noteLimit)

    def _readiness(self, state: dict, trainerId: str):
        """assess() for a trainer, cached until the party actually changes.

        Cheap enough to run every turn, but it is a few hundred damage rolls
        and the answer cannot change while you stand still.
        """
        team = self.roster.team(trainerId)
        if not team:
            return None
        party = state.get("party") or []
        signature = (trainerId, tuple(
            (p.get("species"), p.get("level"), p.get("max_hp"),
             tuple(m.get("name") for m in p.get("moves", [])))
            for p in party))
        if signature not in self._readinessCache:
            report = assess(self.battle.data, party, team)
            levels = (levelsNeeded(self.battle.data, party, team)
                      if report["verdict"] != "ready" else None)
            self._readinessCache = {signature: (report, levels)}   # one entry
        return self._readinessCache[signature]

    def _readinessLine(self, state: dict, trainerId: str) -> str:
        cached = self._readiness(state, trainerId)
        if cached is None:
            return (f"(no team recorded for {trainerId} yet - fight them once, "
                    f"then run `python battle/matchup.py capture {trainerId}`)")
        report, levels = cached
        entry = self.roster.get(trainerId) or {}
        return summarizeReadiness(report, entry.get("name") or trainerId, levels)

    def _ready(self, state: dict, trainerId: str) -> bool:
        cached = self._readiness(state, trainerId)
        return bool(cached and cached[0]["verdict"] == "ready")

    def _repeatAlert(self) -> str:
        """Call out a loop, since the model can't feel itself repeating."""
        recent = self.history[-self.cfg.repeatAlert:]
        if len(recent) < self.cfg.repeatAlert:
            return ""
        first = recent[0]
        if all(e["action"] == first["action"] and e["result"] == first["result"]
               for e in recent):
            return (f"you have done `{first['action']}` {len(recent)} times in a "
                    f"row and nothing changed. Do something different - a "
                    f"different direction, a different command, or `note` what "
                    f"is blocking you.")
        return ""

    def _fillBattle(self, obs: Observation, state: dict):
        # gMain.inBattle stays true while the party menu, a summary page or
        # the bag sits on top of the fight. Rendering the move table there
        # sends `use` taps into whatever menu is really drawn - so name the
        # situation instead and hand the model the way back out.
        # A pending move-learn freezes the battle on a prompt chain that no
        # other tool understands - `use` fails, `wait` changes nothing. Catch
        # it before anything else and point at the one command built for it.
        pending = self.actions._learnPending(state)
        if pending is not None:
            mon, newName = pending
            monName = ((mon or {}).get("nickname")
                       or (mon or {}).get("species") or "Your Pokemon")
            choice = self.actions._forgetChoice(mon, newName) if mon else None
            plan = (f"forget {choice[1]} ({choice[2]}) and learn "
                    f"{newName or 'the new move'}"
                    if choice else "pick a move to forget")
            obs.battleReport = (
                f"MOVE LEARNING: {monName} wants to learn "
                f"{newName or 'a new move'} but already knows four moves. "
                f"The game is PAUSED on this prompt - moves and menus will "
                f"not respond until it is answered.\n"
                f"  `learn`      - make room: {plan}\n"
                f"  `learn skip` - refuse and keep the current moves")
            obs.recommendation = ("RECOMMENDED: `learn` - a new move is "
                                  "almost always worth a slot")
            return

        cb = str((obs.screen or {}).get("callback2", "")).upper()
        if cb and cb not in BATTLE_SCREEN_CALLBACKS:
            obs.battleReport = (
                "You are in a battle, but a menu (party list, summary page or "
                "bag) is open on top of it, so the move table is unavailable "
                "and `use` will NOT work. Press b to back out to the battle "
                "menu - possibly more than once - or finish what you opened "
                "with `press`.")
            return

        snap = Snapshot.capture(_CachedState(state))
        obs.battleReport = _captureText(print_matchup, self.battle, snap)
        if not snap.ready:
            return

        names = move_names(snap.you_raw)
        obs.moveSlots = [(name, i) for i, name in enumerate(names)]

        rows = build_rows(self.battle, snap.you, snap.foe, names, snap.you_raw)
        best = next((r for r in rows if r.is_damaging), None)
        outlook = self._survivalOutlook(snap, state, best)
        if outlook is not None:
            obs.recommendation = outlook
        elif best is not None:
            obs.recommendation = (
                f"CALCULATOR'S PICK: {best.move.name} - highest expected damage "
                f"({best.expected:.0f} per turn, {ko_text(best, snap.foe)}, "
                f"{best.accuracy:.0%} accurate). You do not have to take this "
                f"advice, but you need a reason not to.")
        elif rows:
            obs.recommendation = ("CALCULATOR'S PICK: none of your moves damage "
                                  "this foe. Consider switching or running.")
        note = self._catchOpportunity(snap, state)
        if note:
            obs.recommendation = ((obs.recommendation + "\n")
                                  if obs.recommendation else "") + note

    def _catchOpportunity(self, snap, state: dict) -> str:
        """One line when throwing a ball beats attacking, else empty.

        The model has a `throw` command but nothing in its world ever
        suggests using it - the same gap that kept `run` unused. Catching
        is how a one-Squirtle run becomes a team, so the moment is worth
        calling out: wild, weakened, a ball in the bag, room in the party.
        """
        if snap.you is None or snap.foe is None or snap.foe.max_hp <= 0:
            return ""
        party = state.get("party") or []
        balls = [b for b in (state.get("bag") or {}).get("poke_balls") or []
                 if b.get("quantity", 0) > 0]
        share = snap.foe.current_hp / snap.foe.max_hp
        if (not balls or len(party) >= 6 or share > 0.5
                or self._wildBattle() is False):
            return ""
        species = snap.foe.species
        owned = {(p.get("species_name") or "").upper() for p in party}
        dup = " (you already have one)" if species.upper() in owned else ""
        return (f"CATCH CHANCE: this {species} is at {share:.0%} HP and you "
                f"have {balls[0]['quantity']}x {balls[0]['name']}{dup}. "
                f"`throw` catches it and grows your team - status like sleep "
                f"or paralysis helps.")

    def _wildBattle(self):
        """True for a wild battle, False for a trainer one, None if unreadable.

        `run` only works on wild Pokemon, so advice to flee has to know which
        kind this is. The flag comes straight from RAM; if the read fails the
        advice hedges rather than asserting something it doesn't know.
        """
        try:
            raw = self.client.peek(BATTLE_TYPE_FLAGS_ADDR, 4)
            return not (int.from_bytes(raw, "little") & BATTLE_TYPE_TRAINER)
        except Exception:
            return None

    def _survivalOutlook(self, snap, state: dict, best):
        """A losing exchange spelled out - or None while the fight is fine.

        The pick always names a move and the prompt says the model needs a
        reason to disobey it, so a model that is losing keeps attacking right
        up to the faint - nothing in its world ever says the word "run". When
        the same damage math behind the pick projects that we faint first,
        that projection IS the recommendation, and it has to come from the
        calculator with the same authority as the move picks do.
        """
        healthy = [p for p in (state.get("party") or []) if p.get("hp", 0) > 0]
        backup = len(healthy) > 1

        # Fainted is its own screen, and "run" is not on it. Say what the game
        # is actually waiting for instead of advice that can't be followed.
        if snap.you.current_hp <= 0:
            if backup:
                name = healthy[0].get("nickname") or healthy[0].get(
                    "species_name") or "your next Pokemon"
                return (f"CALCULATOR'S PICK: switch {name} - your active "
                        f"Pokemon has fainted; the game is waiting for the "
                        f"next one.")
            return ("CALCULATOR'S PICK: none - your last Pokemon has fainted "
                    "and this battle is lost. Press a to click through; you "
                    "will wake up healed at the last place you rested.")

        foeRows = [r for r in build_rows(self.battle, snap.foe, snap.you,
                                         move_names(snap.foe_raw), snap.foe_raw)
                   if r.is_damaging and r.expected > 0]
        if not foeRows:
            return None                       # the foe cannot hurt us
        foeBest = foeRows[0]

        # ceil() without floats: turns each side needs at the expected rate.
        theirTurns = -(-snap.you.current_hp // max(1, round(foeBest.expected)))
        ourTurns = (-(-snap.foe.current_hp // max(1, round(best.expected)))
                    if best is not None and best.expected > 0 else 99)
        weMoveFirst = effective_speed(snap.you) > effective_speed(snap.foe)

        # Two ways to be losing. The race: they KO us in fewer turns than we
        # KO them. The gamble: their best hit can end us THIS turn and we
        # cannot guarantee ending them first - a 95% move or a low damage
        # roll is a coin toss with a faint on the wrong face, and expected
        # values don't faint, Pokemon do.
        foeMaxHit = max(r.result.max_damage for r in foeRows)
        dangerNow = foeMaxHit >= snap.you.current_hp
        finishNow = (weMoveFirst and best is not None
                     and best.accuracy >= 0.999
                     and best.result.min_damage >= snap.foe.current_hp)
        losing = (theirTurns < ourTurns
                  or (theirTurns == ourTurns and not weMoveFirst)
                  or (dangerNow and not finishNow))
        if not losing:
            return None

        wild = self._wildBattle()

        if dangerNow and not finishNow:
            math = (f"the foe's next hit can do {foeMaxHit} and you have "
                    f"{snap.you.current_hp} HP - one unlucky turn ends you")
        else:
            math = (f"the foe KOs you in ~{theirTurns} turn(s) but you need "
                    f"~{ourTurns if ourTurns < 99 else '?'} to KO it"
                    + ("" if weMoveFirst else ", and it moves first"))

        if wild is False:                     # trainer: fleeing is not legal
            if backup:
                return (f"CALCULATOR'S PICK: switch - {math}. You cannot flee "
                        f"a trainer battle, but a `switch` to a healthier "
                        f"Pokemon changes the math.")
            if best is None:
                return None
            heals = self._healingItems(state)
            if heals:
                return (f"CALCULATOR'S PICK: {best.move.name}, but WARNING: "
                        f"{math}, and you have nothing to switch to. A "
                        f"{heals[0]} from the `bag` is the only way to change "
                        f"this outcome.")
            return (f"CALCULATOR'S PICK: {best.move.name}, but WARNING: "
                    f"{math}, you have nothing to switch to and NO healing "
                    f"items in the bag. Do your best damage - and after this, "
                    f"buy Potions at a Poke Mart before fighting anyone else.")

        verb = "run" if wild else "run (if this is a wild Pokemon)"
        if backup:
            return (f"CALCULATOR'S PICK: {verb} or switch - {math}. This "
                    f"fight is not worth a faint.")
        return (f"CALCULATOR'S PICK: {verb} - {math}, and this is your LAST "
                f"healthy Pokemon. Fainting here means blacking out and "
                f"losing money. Live to fight something smaller.")

    def _checkPendingMove(self, state: dict, inBattle: bool) -> str:
        """Confirm the move we selected is the move whose PP went down.

        Selecting a move means navigating a cursor we can't read, so this is the
        cheapest honest check available: if a different move's PP dropped, the
        cursor was somewhere we didn't expect and the model deserves to be told
        rather than left wondering why it never used Ember.
        """
        pending, self._pendingMove = self._pendingMove, None
        if not pending or not inBattle:
            return ""

        battle = state.get("battle") or {}
        active = battle.get("player_active") or {}
        if active.get("species") != pending["species"]:
            return ""
        current = {m["name"]: m["pp"] for m in active.get("moves", [])}
        if pending["name"] not in current:
            return ""

        spent = [name for name, pp in current.items()
                 if pp < pending["pp"].get(name, pp)]
        if not spent:
            return ""     # the turn hasn't resolved yet; nothing to judge
        if pending["name"] in spent:
            return ""
        return (f"the game used {spent[0]}, not {pending['name']} - the battle "
                f"menu cursor was not where the harness expected. Re-check the "
                f"move list before attacking again.")

    # ---- one turn ---------------------------------------------------------

    def step(self) -> dict:
        self.turn += 1
        self.actions.turn = self.turn
        obs = self.observe()
        self.actions.observation = obs

        report = obs.render(self.cfg, self.history)
        self.lastReport = report
        print(f"\n{'=' * 72}")
        print(report)
        print("-" * 72)

        verb, args, reply = self.brain.decide(report, obs.screenshotPath)
        think = self._extractThought(reply)
        if think:
            print(f"MODEL: {think}")

        if verb is None:
            verb, args = self._fallback(obs)
            print(f"MODEL gave no usable command; falling back to `{verb}"
                  f"{' ' + ' '.join(args) if args else ''}`")
            print(f"  raw reply: {reply.strip()[:200]}")

        command = f"{verb} {' '.join(args)}".strip()
        print(f"ACTION: {command}")

        result = self._perform(verb, args, obs)
        print(f"RESULT: {result}")

        entry = {"turn": self.turn, "action": command, "result": result,
                 "think": think}
        self.history.append(entry)
        self.memory.data["turns_played"] = self.turn
        self.memory.data["last_action"] = command
        self.memory.save()
        self._log(obs, report, reply, entry)
        return entry

    def _perform(self, verb: str, args: list, obs: Observation) -> str:
        try:
            if verb == "use" and obs.inBattle:
                # Snapshot the PP before the taps so _checkPendingMove has a
                # baseline to compare against next turn.
                active = (obs.state.get("battle") or {}).get("player_active") or {}
                hit = _bestMatch(" ".join(args), obs.moveSlots) if args else None
                if hit is not None:
                    self._pendingMove = {
                        "name": hit[0],
                        "species": active.get("species"),
                        "pp": {m["name"]: m["pp"] for m in active.get("moves", [])},
                    }
            return self.actions.execute(verb, args)
        except ActionError as exc:
            self._pendingMove = None
            return f"that didn't work: {exc}"
        except (MGBAError, ValueError) as exc:
            self._pendingMove = None
            return f"the emulator refused that: {exc}"

    def _fallback(self, obs: Observation):
        """What to do when the model produced nothing we could parse.

        With the map covered (dialog, cutscene) the only useful move is to
        advance the text; anywhere else, do nothing rather than press a button
        we didn't choose.
        """
        if obs.namingOpen:
            # Never a blind A here: on the keyboard screen the cursor may be
            # sitting on OK, and pressing it accepts whatever is typed.
            return "wait", ["1"]
        if not obs.inBattle and obs.fix is None:
            return "press", ["a"]
        return "wait", ["1"]

    @staticmethod
    def _extractThought(reply: str) -> str:
        match = re.search(r"^[\s>*\-]*think\s*[:\-]\s*(.+)$", reply or "",
                          re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip().strip("*") if match else ""

    def _log(self, obs, report, reply, entry):
        if not self.cfg.logPath:
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "turn": entry["turn"], "in_battle": obs.inBattle,
            "map": obs.fix["mapName"] if obs.fix else None,
            "tile": list(obs.fix["tile"]) if obs.fix else None,
            "report": report, "reply": reply,
            "action": entry["action"], "result": entry["result"],
        }
        with self.cfg.logPath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ---- run modes --------------------------------------------------------

    def run(self, maxTurns: int | None = None):
        print("\nPlaying. Ctrl-C to stop.")
        # Counted for this run, not since the save began: self.turn continues
        # across restarts so the objective clock survives them, which would
        # make `--turns 3` mean "stop at turn 3, i.e. immediately".
        taken = 0
        wasPaused = False
        try:
            while maxTurns is None or taken < maxTurns:
                if self.inbox is not None and self.inbox.stopped:
                    break
                if self.inbox is not None and self.inbox.paused:
                    if not wasPaused:
                        print("\n-- paused by operator --")
                        wasPaused = True
                    time.sleep(0.2)
                    continue
                if wasPaused:
                    print("-- resumed --")
                    wasPaused = False
                try:
                    self.step()
                    taken += 1
                except (MGBAError, ValueError) as exc:
                    print(f"Turn failed: {exc}")
                except ConnectionError as exc:
                    print(f"Lost the emulator: {exc}")
                    break
                except ollama.ResponseError as exc:
                    print(f"Ollama refused the request: {exc}")
                    break
                time.sleep(self.cfg.turnDelay)
        except KeyboardInterrupt:
            print("\nStopped.")
        print(f"Played {taken} turn(s) this run ({self.turn} on this save).")

    def dryRun(self):
        """One turn that asks the model and prints its choice, executing nothing."""
        self.turn += 1
        obs = self.observe()
        self.actions.observation = obs
        report = obs.render(self.cfg, self.history)
        print(report)
        print("-" * 72)
        verb, args, reply = self.brain.decide(report, obs.screenshotPath)
        print(f"RAW REPLY:\n{reply}")
        print(f"\nPARSED: {verb} {args}  (not executed)")


# --------------------------------------------------------------------------
# Manual console — the same tool layer, driven by a human
# --------------------------------------------------------------------------

MANUAL_HELP = """Commands (the same grammar the model uses):
  <any game command>   press a / move up 3 / goto pokemon center / use ember ...
                       note <text> / check brock
  obs                  build and print this turn's report
  ask                  run one full model turn
  raw                  print the model's last raw reply
  memory               print memories.json as the harness sees it
  forget <text|all>    drop remembered notes matching that text
  help / quit
"""


def manual(player: PlayerAI):
    print("=== player_ai manual console ===")
    print(MANUAL_HELP)
    player.actions.turn = player.turn
    obs = player.observe()
    player.actions.observation = obs
    print(obs.render(player.cfg, player.history))

    while True:
        try:
            line = input("\nplay> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        low = line.lower()

        try:
            if low in ("q", "quit", "exit"):
                break
            if low in ("help", "?"):
                print(MANUAL_HELP)
                continue
            if low in ("obs", "state", "report"):
                obs = player.observe()
                player.actions.observation = obs
                print(obs.render(player.cfg, player.history))
                continue
            if low == "ask":
                player.step()
                continue
            if low == "raw":
                print(player.brain.lastReply or "(nothing asked yet)")
                continue
            if low == "memory":
                print(json.dumps(player.memory.data, indent=2))
                continue
            if low.startswith("forget"):
                target = line[len("forget"):].strip()
                if not target:
                    print("Usage: forget <text|all>")
                else:
                    print(f"dropped {player.memory.forget(target)} note(s)")
                continue

            parsed = parseCommand(line)
            if parsed is None:
                print(f"Not a command I know: {line!r}")
                continue
            obs = player.observe()
            player.actions.observation = obs
            print(player._perform(parsed[0], parsed[1], obs))
        except ConnectionError as exc:
            print(f"Lost the emulator: {exc}")
            break

    print("Disconnected.")


# --------------------------------------------------------------------------
# GUI mode — an operator console next to the terminal
# --------------------------------------------------------------------------


def runGui(player: PlayerAI, maxTurns: int | None):
    """Play with a Tkinter console alongside it: pause, resume, and hand the
    model a short-term request. All the usual THINK/ACTION/RESULT printing
    still happens in the terminal - the window only owns pause and input.

    Every submitted request is judged before the model ever sees it: a free
    instant filter, then a separate model call that reviews it against the
    current game state (see feasibility.py). That runs on its own thread too,
    so a slow judgment never stalls a turn or freezes the window.

    Tkinter needs the main thread, so the play loop runs on a background
    thread instead; every side only ever touches the shared OperatorInbox.
    """
    from feasibility import FeasibilityWorker, Referee
    from operator_gui import OperatorGui
    from operator_inbox import OperatorInbox

    inbox = OperatorInbox()
    player.inbox = inbox

    playThread = threading.Thread(target=lambda: player.run(maxTurns=maxTurns),
                                  daemon=True)
    playThread.start()

    referee = Referee(player.cfg.model, ollamaHost=player.cfg.ollamaHost)
    worker = FeasibilityWorker(inbox, referee,
                               getSituationReport=lambda: player.lastReport)
    worker.start()

    OperatorGui(inbox).run()   # blocks until the window is closed

    inbox.stop()
    playThread.join(timeout=5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def buildConfig(args) -> Config:
    cfg = Config(model=args.model, ollamaHost=args.ollama_host,
                 mgbaHost=args.host, mgbaPort=args.port,
                 goal=args.goal or "", logPath=args.log)
    cfg.sendImage = not args.no_image
    cfg.showDestinations = not args.no_places
    cfg.temperature = args.temperature
    cfg.turnDelay = args.delay
    cfg.objectivesPath = Path(args.objectives).resolve()
    cfg.memoriesPath = Path(args.memories).resolve()
    cfg.useObjectives = not args.no_objectives
    cfg.think = args.think
    if args.think:
        cfg.numPredict = max(cfg.numPredict, 1024)   # room to think AND answer
    if args.screenshot:
        cfg.screenshotPath = Path(args.screenshot).resolve()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("mode", nargs="?", default="play",
                        choices=("play", "once", "manual", "dry-run", "gui"))
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--ollama-host", default=None,
                        help="e.g. http://192.168.1.20:11434")
    parser.add_argument("--host", default=Config.mgbaHost)
    parser.add_argument("--port", type=int, default=Config.mgbaPort)
    parser.add_argument("--goal", default=None,
                        help="one line of intent shown to the model every turn")
    parser.add_argument("--turns", type=int, default=None,
                        help="stop after this many turns")
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--delay", type=float, default=Config.turnDelay,
                        help="seconds between turns")
    parser.add_argument("--screenshot", default=None,
                        help="where to write the frame the model sees")
    parser.add_argument("--think", action="store_true",
                        help="let a thinking model reason before answering "
                             "(slower, and it may never reach the ACTION line)")
    parser.add_argument("--no-image", action="store_true",
                        help="text-only prompts (for models without vision)")
    parser.add_argument("--no-places", action="store_true",
                        help="skip the walkable-destination survey each turn")
    parser.add_argument("--objectives", type=Path, default=Config.objectivesPath,
                        help="walkthrough to follow (objectives.json)")
    parser.add_argument("--memories", type=Path, default=Config.memoriesPath,
                        help="where progress and notes are stored")
    parser.add_argument("--no-objectives", action="store_true",
                        help="play with no walkthrough (notes still work)")
    parser.add_argument("--log", type=Path, default=None,
                        help="append every turn to this JSONL file")
    args = parser.parse_args()

    cfg = buildConfig(args)

    try:
        player = PlayerAI(cfg)
    except OSError as exc:
        print(f"Could not connect to mGBA on {cfg.mgbaHost}:{cfg.mgbaPort} - {exc}")
        print("Load mGBA/mgba_server.lua in mGBA (Tools > Scripting) and retry.")
        return 1

    with player:
        if args.mode != "manual":
            player.brain.preload()
        if args.mode == "manual":
            manual(player)
        elif args.mode == "dry-run":
            player.dryRun()
        elif args.mode == "once":
            player.step()
        elif args.mode == "gui":
            runGui(player, args.turns)
        else:
            player.run(maxTurns=args.turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
