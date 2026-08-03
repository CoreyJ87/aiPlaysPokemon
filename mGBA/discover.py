"""
discover.py — locate FR/LG memory symbols empirically, instead of trusting
addresses copied off a forum post.

Everything in mgba_server.lua's GAME_STATE is anchored to addresses that are
either published constants or derived from a known struct layout. The symbols
this tool hunts (gStringVar4, gTasks, gPaletteFade) are the ones worth having
but easy to get subtly wrong — a near-miss address reads plausible garbage
rather than failing loudly. So: find them by observation on YOUR ROM, confirm
them, then save them.

Discovered addresses are written to addresses.json next to this file and
re-registered on later runs via MGBAClient.load_addrs().

Usage:
    python discover.py watch            # log screens as you move around
    python discover.py label            # name the screen showing right now
    python discover.py stringvar "<text on screen right now>"
    python discover.py tasks            # scan IWRAM for the gTasks array
    python discover.py diff             # snapshot/compare memory between states
    python discover.py verify           # re-check saved addresses still work
    python discover.py show             # print saved addresses

Typical first session:
    1. python discover.py watch
       Walk into a battle, open the bag, open the party menu, name a Pokemon.
       Then `label` to name the ones you care about.
    2. Stand in a dialog box, then:
       python discover.py stringvar "the exact words on screen"
    3. python discover.py tasks
    4. python discover.py verify

A note on what callback2 can and cannot do: it identifies SCREENS. Dialog
boxes and the START menu are tasks running under the overworld callback, so
they do not change it. Steps 2 and 3 are what cover those, and they are the
steps that matter most for an AI player.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mgba_client import (  # noqa: E402
    MGBAClient, MGBAError, gen3_decode, gen3_encode,
    EWRAM_START, EWRAM_SIZE, IWRAM_START, IWRAM_SIZE, ROM_START,
    _format_hexdump,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ADDRESS_FILE = os.path.join(HERE, "addresses.json")
SCREEN_FILE = os.path.join(HERE, "screens.json")
SNAPSHOT_DIR = os.path.join(HERE, ".snapshots")

TASK_STRUCT_SIZE = 0x28
TASK_COUNT = 16
ROM_END = 0x09FFFFFF


# -------------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------------

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"  -> saved {os.path.relpath(path, HERE)}")


def save_address(name: str, addr: int):
    addrs = load_json(ADDRESS_FILE, {})
    addrs[name] = f"{addr:08X}"
    save_json(ADDRESS_FILE, addrs)


def connect() -> MGBAClient:
    """Connect and re-register any previously discovered addresses."""
    client = MGBAClient()
    saved = load_json(ADDRESS_FILE, {})
    if saved:
        client.load_addrs(saved)
        print(f"Registered {len(saved)} saved address(es): {', '.join(sorted(saved))}")
    return client


# -------------------------------------------------------------------------
# watch — build the callback2 -> screen name table
# -------------------------------------------------------------------------

DEFAULT_MIN_DWELL = 0.4     # seconds a callback2 must hold to count as a screen


def cmd_watch(client: MGBAClient, args):
    """Poll SCREEN and report screens as they settle.

    callback2 needs no discovery — gMain's layout pins it — but it needs
    LABELS, and only you know what was on screen when it changed.

    Most callback2 values you pass through are transitions: fades, map loads,
    battle-intro stages. They hold for a frame or two and scroll the real
    screens off the top. So a value is only printed once it has HELD for
    min-dwell seconds. Pass --all to see every change including transitions.
    """
    labels = load_json(SCREEN_FILE, {})
    show_all = "--all" in args
    min_dwell = DEFAULT_MIN_DWELL
    if "--min-dwell" in args:
        min_dwell = float(args[args.index("--min-dwell") + 1])

    print("Watching. Move between menus/battles/dialogs. Ctrl-C to stop.")
    print(f"Reporting states held >= {min_dwell}s"
          + (" (--all: showing transitions too)" if show_all else "")
          + ".\n")
    print(f"{'held':>7}  {'callback2':>9}  {'saved_cb':>9}  {'st':>3}  bat  label")
    print("-" * 72)

    current, since, printed = None, time.time(), False
    stats = {}          # cb2 -> {"n": count, "total": secs, "max": secs}

    def record(cb2, held):
        e = stats.setdefault(cb2, {"n": 0, "total": 0.0, "max": 0.0})
        e["n"] += 1
        e["total"] += held
        e["max"] = max(e["max"], held)

    try:
        while True:
            s = client.screen()
            key = (s["callback2"], s["saved_callback"])
            now = time.time()

            if key != current:
                if current is not None:
                    record(current[0], now - since)
                current, since, printed = key, now, False
                if show_all:
                    _print_watch_row(0.0, s, labels)
                    printed = True
            elif not printed and (now - since) >= min_dwell:
                _print_watch_row(now - since, s, labels)
                printed = True

            time.sleep(0.05)
    except KeyboardInterrupt:
        if current is not None:
            record(current[0], time.time() - since)

    _print_watch_summary(stats, labels, min_dwell)


def _print_watch_row(held, s, labels):
    cb2 = s["callback2"]
    label = labels.get(cb2) or "<unlabelled>"
    print(f"{held:>6.1f}s  {cb2:>9}  {s['saved_callback']:>9}  "
          f"{s['main_state']:>3}  {'yes' if s['in_battle'] else ' no':>3}  {label}")


def _print_watch_summary(stats, labels, min_dwell):
    if not stats:
        print("\nNothing observed.")
        return

    # A screen is something you sit in; a transition is something you pass
    # through. Max dwell separates them far better than visit count does.
    screens = {k: v for k, v in stats.items() if v["max"] >= min_dwell}
    transitions = {k: v for k, v in stats.items() if v["max"] < min_dwell}

    print(f"\n{'=' * 72}\nSaw {len(stats)} distinct callback2 values "
          f"({len(screens)} screens, {len(transitions)} transitions)\n")

    def dump(title, group):
        if not group:
            return
        print(f"{title}:")
        print(f"  {'callback2':>9}  {'visits':>6}  {'max held':>8}  label")
        for cb2, e in sorted(group.items(), key=lambda kv: -kv[1]["max"]):
            print(f"  {cb2:>9}  {e['n']:>6}  {e['max']:>7.1f}s  "
                  f"{labels.get(cb2) or '<unlabelled>'}")
        print()

    dump("SCREENS (worth labelling)", screens)
    dump("TRANSITIONS (usually not worth labelling)", transitions)

    new = [cb for cb in stats if cb not in labels]
    if new:
        for cb in new:
            labels.setdefault(cb, "")
        save_json(SCREEN_FILE, labels)
        print(f"Added {len(new)} new value(s) to screens.json to fill in.")
    print("Tip: `python discover.py label` walks you through naming these one "
          "at a time, which is far easier than matching them up after the fact.")


# -------------------------------------------------------------------------
# label — guided, one screen at a time
# -------------------------------------------------------------------------

def cmd_label(client: MGBAClient, args):
    """Interactively name the screen the game is currently showing.

    Reading a scrolling trace after the fact means reconstructing what you
    were doing from memory. This inverts it: put the game where you want it,
    press Enter, name it.
    """
    labels = load_json(SCREEN_FILE, {})
    print("Put the game in a state, then press Enter to capture it.")
    print("Empty name skips. Ctrl-C or 'q' to finish.\n")

    try:
        while True:
            input("  [Enter to capture] ")

            # Sample briefly so we capture the settled state, not a fade frame
            samples = []
            for _ in range(6):
                samples.append(client.screen()["callback2"])
                time.sleep(0.05)
            if len(set(samples)) > 1:
                print(f"    Screen is still changing ({', '.join(sorted(set(samples)))})"
                      " — wait for it to settle and try again.")
                continue

            cb2 = samples[0]
            s = client.screen()
            existing = labels.get(cb2)
            print(f"    callback2={cb2}  saved_cb={s['saved_callback']}  "
                  f"state={s['main_state']}  in_battle={s['in_battle']}")
            if existing:
                print(f"    currently labelled: {existing!r}")

            name = input("    name: ").strip()
            if name.lower() == "q":
                break
            if not name:
                print("    skipped.")
                continue
            labels[cb2] = name
            save_json(SCREEN_FILE, labels)
    except (KeyboardInterrupt, EOFError):
        print()

    labelled = sum(1 for v in labels.values() if v and not v.startswith("?"))
    print(f"\n{labelled}/{len(labels)} values have confirmed labels.")


# -------------------------------------------------------------------------
# stringvar — locate gStringVar1..4
# -------------------------------------------------------------------------

def cmd_stringvar(client: MGBAClient, args):
    """Find gStringVar4 by searching EWRAM for text currently on screen.

    gStringVar4 holds the fully-expanded dialog string, so a phrase you can
    read in the message box is sitting in it verbatim, Gen 3 encoded. Any
    other hits are the ROM-side template or a copy buffer; the EWRAM hit that
    tracks the message box as it advances is the one you want.
    """
    if not args:
        print('Usage: python discover.py stringvar "text currently on screen"')
        print("Tip: use a distinctive phrase, and omit trailing punctuation.")
        return

    needle = " ".join(args)
    encoded = gen3_encode(needle)
    print(f"Searching EWRAM for: {needle!r}")
    print(f"  Gen 3 encoding: {encoded.hex().upper()}")

    if 0x00 in encoded:
        unmapped = [c for c in needle if gen3_encode(c) == b"\x00" and c != " "]
        if unmapped:
            print(f"  WARNING: no Gen 3 code for {set(unmapped)} — "
                  f"these matched as spaces, so hits may be loose.")

    hits = client.find_text(needle, EWRAM_START, EWRAM_SIZE)
    if not hits:
        print("\nNo match. Things to check:")
        print("  - Is the dialog actually on screen right now?")
        print("  - Retype the phrase exactly, including capitalisation.")
        print("  - Try a shorter fragment (5-10 characters).")
        return

    print(f"\n{len(hits)} match(es):")
    for addr in hits:
        # Walk back to the start of the buffer: the match may be mid-string.
        context_start = max(EWRAM_START, addr - 0x40)
        data = client.peek(context_start, 0xC0)
        print(f"\n  --- {addr:08X} ---")
        print(_format_hexdump(context_start, data))
        full = gen3_decode(client.peek(addr, 200))
        print(f"  reads as: {full!r}")

    print("\nTo confirm which hit is gStringVar4, advance the dialog (tap A) "
          "and re-run this with the NEW text. The address that changes to "
          "follow the message box is gStringVar4; ROM hits will not move.")
    print("\nThen register it:")
    print(f"  python discover.py set gStringVar4 {hits[0]:08X}")


# -------------------------------------------------------------------------
# tasks — locate the gTasks array
# -------------------------------------------------------------------------

def _score_task_array(data: bytes, base_offset: int) -> int:
    """Score a candidate gTasks base. Higher is more task-array-like.

    A real gTasks has 16 x 0x28-byte slots where isActive is strictly 0 or 1,
    and every active slot's func field points into ROM. Random IWRAM almost
    never satisfies both across all 16 slots.
    """
    active = 0
    for i in range(TASK_COUNT):
        slot = base_offset + i * TASK_STRUCT_SIZE
        if slot + TASK_STRUCT_SIZE > len(data):
            return -1
        func = int.from_bytes(data[slot:slot + 4], "little")
        is_active = data[slot + 4]
        if is_active > 1:
            return -1                      # isActive is a bool8
        if is_active == 1:
            if not (ROM_START <= func <= ROM_END):
                return -1                  # active task must have a ROM func
            active += 1
        elif func != 0 and not (ROM_START <= func <= ROM_END):
            return -1                      # stale funcs are 0 or a ROM ptr
    return active


def cmd_tasks(client: MGBAClient, args):
    """Scan IWRAM for the gTasks array by its structural signature."""
    print("Reading IWRAM (32 KB)...")
    iwram = client.read_range(IWRAM_START, IWRAM_SIZE)

    print("Scanning for a 16 x 0x28 task array...")
    candidates = []
    for offset in range(0, IWRAM_SIZE - TASK_COUNT * TASK_STRUCT_SIZE, 4):
        score = _score_task_array(iwram, offset)
        if score >= 2:                     # at least 2 tasks always run
            candidates.append((score, IWRAM_START + offset))

    if not candidates:
        print("No candidate found. Is the game past the title screen?")
        return

    candidates.sort(key=lambda c: (-c[0], c[1]))
    print(f"\n{len(candidates)} candidate(s), best first:\n")
    for score, addr in candidates[:8]:
        print(f"  {addr:08X}  ({score} active tasks)")
        for i in range(TASK_COUNT):
            slot = addr - IWRAM_START + i * TASK_STRUCT_SIZE
            func = int.from_bytes(iwram[slot:slot + 4], "little")
            if iwram[slot + 4] == 1:
                print(f"      slot {i:2d}: func={func:08X}")

    best = candidates[0][1]
    print(f"\nMost likely gTasks base: {best:08X}")
    print("Confirm it: open the START menu, re-run this, and check that an "
          "extra task appears at the same base. Then register it:")
    print(f"  python discover.py set gTasks {best:08X}")


# -------------------------------------------------------------------------
# diff — generic "what changed between these two states" search
# -------------------------------------------------------------------------

def _snapshot_path(name: str) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    return os.path.join(SNAPSHOT_DIR, f"{name}.bin")


def cmd_snap(client: MGBAClient, args):
    """Dump EWRAM+IWRAM to a named snapshot file."""
    name = args[0] if args else "snap"
    print(f"Snapshotting EWRAM + IWRAM as {name!r}...")
    data = client.read_range(EWRAM_START, EWRAM_SIZE) + \
        client.read_range(IWRAM_START, IWRAM_SIZE)
    with open(_snapshot_path(name), "wb") as f:
        f.write(data)
    print(f"  {len(data)} bytes -> {_snapshot_path(name)}")


def cmd_diff(client: MGBAClient, args):
    """Compare two snapshots and report changed regions.

    Use this for symbols with no searchable text — gPaletteFade, menu cursor
    indices, script-context flags. Snapshot in state A, change one thing,
    snapshot as B, and diff. The fewer things you change between snapshots,
    the shorter the candidate list.
    """
    if len(args) < 2:
        print("Usage: python discover.py diff <snapA> <snapB>")
        print("  Take snapshots first: python discover.py snap <name>")
        print(f"  Existing: {', '.join(sorted(_list_snapshots())) or '(none)'}")
        return

    a_path, b_path = _snapshot_path(args[0]), _snapshot_path(args[1])
    for p in (a_path, b_path):
        if not os.path.exists(p):
            print(f"Missing snapshot: {p}")
            return

    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    if len(a) != len(b):
        print("Snapshots differ in size — retake them.")
        return

    # Group consecutive changed bytes into runs
    runs, run_start = [], None
    for i in range(len(a)):
        if a[i] != b[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            runs.append((run_start, i))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(a)))

    def to_addr(offset):
        if offset < EWRAM_SIZE:
            return EWRAM_START + offset
        return IWRAM_START + (offset - EWRAM_SIZE)

    changed = sum(end - start for start, end in runs)
    print(f"{len(runs)} changed region(s), {changed} bytes total "
          f"({changed / len(a) * 100:.2f}% of memory)\n")

    # Long runs are usually sprite/tile buffers; short ones are flags & vars
    short = [r for r in runs if r[1] - r[0] <= 16]
    print(f"Showing the {min(len(short), 40)} shortest runs "
          f"(flags and counters look like this; big runs are graphics):\n")
    for start, end in short[:40]:
        addr = to_addr(start)
        old = a[start:end].hex().upper()
        new = b[start:end].hex().upper()
        print(f"  {addr:08X}  {old:>16} -> {new:<16}")


def _list_snapshots():
    if not os.path.isdir(SNAPSHOT_DIR):
        return []
    return [f[:-4] for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".bin")]


# -------------------------------------------------------------------------
# set / show / verify
# -------------------------------------------------------------------------

def cmd_set(client: MGBAClient, args):
    """Register and persist a symbol address."""
    if len(args) < 2:
        print("Usage: python discover.py set <symbol> <hex_addr>")
        return
    name, addr = args[0], int(args[1], 16)
    client.set_addr(name, addr)            # raises if the name is unknown
    save_address(name, addr)
    print(f"Registered {name} = {addr:08X}")


def cmd_show(client: MGBAClient, args):
    print(json.dumps(client.addrs(), indent=2))
    labels = load_json(SCREEN_FILE, {})
    if labels:
        print("\nScreen labels (screens.json):")
        for cb2, name in sorted(labels.items()):
            print(f"  {cb2}  {name or '<unlabelled>'}")


def cmd_verify(client: MGBAClient, args):
    """Sanity-check every saved address against live memory."""
    saved = load_json(ADDRESS_FILE, {})
    if not saved:
        print("No saved addresses yet.")
        return

    print("Checking saved addresses against live memory...\n")
    ok = True

    if "gStringVar4" in saved:
        try:
            d = client.dialog()
            text = d.get("text", "")
            printable = all(ch.isprintable() for ch in text)
            verdict = "OK" if (text and printable) else "SUSPECT"
            ok &= verdict == "OK"
            print(f"  gStringVar4 {saved['gStringVar4']}  [{verdict}]")
            print(f"    reads: {text[:70]!r}")
            if not text:
                print("    (empty — open a dialog box and re-run to confirm)")
        except MGBAError as e:
            ok = False
            print(f"  gStringVar4  [FAIL] {e}")

    if "gTasks" in saved:
        try:
            t = client.tasks()
            verdict = "OK" if 0 < t["count"] <= TASK_COUNT else "SUSPECT"
            ok &= verdict == "OK"
            print(f"  gTasks {saved['gTasks']}  [{verdict}] "
                  f"{t['count']} active task(s)")
            for task in t["tasks"]:
                print(f"    slot {task['slot']:2d}: func={task['func']}")
        except MGBAError as e:
            ok = False
            print(f"  gTasks  [FAIL] {e}")

    print("\nAll checks passed." if ok else
          "\nSomething looks wrong — re-run discovery for the SUSPECT symbols.")


# -------------------------------------------------------------------------

COMMANDS = {
    "watch": cmd_watch,
    "label": cmd_label,
    "stringvar": cmd_stringvar,
    "tasks": cmd_tasks,
    "snap": cmd_snap,
    "diff": cmd_diff,
    "set": cmd_set,
    "show": cmd_show,
    "verify": cmd_verify,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 1

    try:
        client = connect()
    except (ConnectionError, OSError) as e:
        print(f"Could not connect to mGBA on 127.0.0.1:54321 — is the script loaded? ({e})")
        return 1

    try:
        COMMANDS[sys.argv[1]](client, sys.argv[2:])
    except MGBAError as e:
        print(f"Server error: {e}")
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
