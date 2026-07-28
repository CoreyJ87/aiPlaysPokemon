"""
mgba_client.py — Python client for mgba_server.lua

As a library:
    from mgba_client import MGBAClient

    with MGBAClient() as gba:
        gba.ping()
        gba.tap("A")
        gba.tap_sequence(["DOWN", "DOWN", "A"], frames=8)
        state = gba.game_state()
        gba.screenshot("shot.png")

    # or without the context manager
    gba = MGBAClient(host="127.0.0.1", port=54321)
    gba.connect()
    ...
    gba.close()

As a script:
    python mgba_client.py              # interactive mode
    python mgba_client.py tap A        # tap A button (default hold)
    python mgba_client.py tap A 16     # tap A button for 16 frames
    python mgba_client.py screenshot   # save screenshot to screenshot.png
    python mgba_client.py game_state   # print full game state
    python mgba_client.py ping         # health check

Multiple clients can connect simultaneously.
"""

import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 54321

BUTTONS = ("A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R")


class MGBAError(RuntimeError):
    """Server replied with an ERR|... response."""


class MGBAClient:
    """Connection to mgba_server.lua running inside mGBA.

    Every method raises MGBAError if the server returns an error response, and
    ConnectionError if the socket drops mid-exchange.
    """

    def __init__(self, host: str = HOST, port: int = PORT,
                 timeout: float = 10.0, auto_connect: bool = True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        if auto_connect:
            self.connect()

    # ---- connection management -------------------------------------------

    def connect(self):
        """Open the socket. Safe to call again after close()."""
        if self.sock is not None:
            return self
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.sock = sock
        return self

    def close(self):
        """Close the socket if open."""
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def reconnect(self):
        """Drop and re-open the connection (e.g. after mGBA restarts)."""
        self.close()
        return self.connect()

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---- low-level protocol ----------------------------------------------

    def _require_sock(self) -> socket.socket:
        if self.sock is None:
            raise ConnectionError("Not connected — call connect() first")
        return self.sock

    def _recv_line(self, buf: bytes = b"") -> tuple:
        """Read until a newline. Returns (header_str, leftover_bytes)."""
        sock = self._require_sock()
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            buf += chunk
        nl = buf.index(b"\n")
        return buf[:nl].decode("utf-8"), buf[nl + 1:]

    def send_raw(self, command: str) -> tuple:
        """Send a command and return its raw (header, leftover_bytes).

        No error checking — use this for commands this class doesn't wrap yet.
        """
        self._require_sock().sendall((command + "\n").encode("utf-8"))
        return self._recv_line()

    def send(self, command: str) -> tuple:
        """Send a command, raising MGBAError on an ERR response."""
        header, remainder = self.send_raw(command)
        if header.startswith("ERR"):
            raise MGBAError(header.split("|", 1)[-1] if "|" in header else header)
        return header, remainder

    # ---- commands ---------------------------------------------------------

    def ping(self) -> bool:
        """Health check. Returns True, or raises on failure."""
        self.send("PING")
        return True

    def tap(self, button: str, frames: int = None):
        """Press a button for `frames` frames (server default if omitted)."""
        button = button.upper()
        if button not in BUTTONS:
            raise ValueError(f"Unknown button {button!r}; expected one of {', '.join(BUTTONS)}")
        cmd = f"TAP|{button}" + (f"|{frames}" if frames is not None else "")
        self.send(cmd)

    def tap_sequence(self, buttons, frames: int = None, delay: float = 0.25):
        """Tap several buttons in order, sleeping `delay` seconds between each."""
        for i, button in enumerate(buttons):
            if i:
                time.sleep(delay)
            self.tap(button, frames)

    def screenshot(self, output_path: str = None) -> bytes:
        """Capture a screenshot. Returns the PNG bytes, saving them if a path is given."""
        self._require_sock().sendall(b"SCREENSHOT\n")
        header, remainder = self._recv_line()
        if header.startswith("ERR"):
            raise MGBAError(header.split("|", 1)[-1])

        byte_length = int(header.split("|")[1])
        png_data = remainder
        while len(png_data) < byte_length:
            chunk = self.sock.recv(min(65536, byte_length - len(png_data)))
            if not chunk:
                raise ConnectionError("Server closed connection during transfer")
            png_data += chunk
        png_data = png_data[:byte_length]

        if output_path:
            with open(output_path, "wb") as f:
                f.write(png_data)
        return png_data

    def game_state(self) -> dict:
        """Return the full game state as a dict."""
        header, _ = self.send("GAME_STATE")
        parts = header.split("|", 1)
        if len(parts) < 2:
            raise MGBAError(f"Unexpected GAME_STATE response: {header}")
        return json.loads(parts[1])

    def position(self) -> dict:
        """Return just {map_bank, map_number, x, y, in_battle}.

        Cheap enough to poll after every button press, unlike game_state(),
        which serializes the party, the whole bag and any live battle.
        """
        header, _ = self.send("POSITION")
        parts = header.split("|", 1)
        if len(parts) < 2:
            raise MGBAError(f"Unexpected POSITION response: {header}")
        return json.loads(parts[1])

    # ---- convenience accessors -------------------------------------------

    def player(self) -> dict:
        """Player block from the game state (name, money, badges, map, x/y)."""
        return self.game_state().get("player", {})

    def party(self) -> list:
        """The player's party as a list of Pokemon dicts."""
        return self.game_state().get("party", [])

    def bag(self) -> dict:
        """Bag contents keyed by pocket name."""
        return self.game_state().get("bag", {})

    def in_battle(self) -> bool:
        return bool(self.game_state().get("in_battle"))

    def print_game_state(self):
        """Pretty-print the current game state to stdout."""
        print_game_state(self.game_state())


# -------------------------------------------------------------------------
# Backwards-compatible module-level helpers (take a raw socket)
# -------------------------------------------------------------------------

def _wrap(sock: socket.socket) -> MGBAClient:
    client = MGBAClient(auto_connect=False)
    client.sock = sock
    return client


def send_command(sock: socket.socket, command: str) -> tuple:
    """Send a newline-terminated command and return (header, leftover_bytes)."""
    return _wrap(sock).send_raw(command)


def tap(sock: socket.socket, button: str, frames: int = None):
    try:
        _wrap(sock).tap(button, frames)
        print(f"TAP {button}: OK")
    except MGBAError as e:
        print(f"TAP {button}: ERR|{e}")


def screenshot(sock: socket.socket, output_path: str = "screenshot.png"):
    try:
        data = _wrap(sock).screenshot(output_path)
        print(f"Screenshot saved to {output_path} ({len(data)} bytes)")
    except MGBAError as e:
        print(f"Screenshot failed: {e}")


def ping(sock: socket.socket):
    try:
        _wrap(sock).ping()
        print("PING: OK")
    except MGBAError as e:
        print(f"PING: ERR|{e}")


def game_state(sock: socket.socket):
    """Request the full game state and pretty-print it."""
    try:
        print_game_state(_wrap(sock).game_state())
    except MGBAError as e:
        print(f"GAME_STATE failed: {e}")


# -------------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------------

def _format_types(pk: dict) -> str:
    types = pk.get("type1", "?")
    if pk.get("type2") and pk["type2"] != pk.get("type1"):
        types += f"/{pk['type2']}"
    return types


def _print_pokemon(pk: dict, index: int = None, indent: str = "  ",
                   show_details: bool = False):
    """Print one Pokemon from a party list (player or enemy)."""
    prefix = f"{indent}{index}. " if index is not None else indent
    status = f" [{pk['status']}]" if pk.get("status") != "OK" else ""
    item = f"  @{pk['held_item']}" if pk.get("held_item") not in (None, "None") else ""
    # Battle mons from gBattleMons don't carry a nature field
    nature = f"  {pk['nature']}" if pk.get("nature") else ""
    print(f"{prefix}{pk['nickname']} ({pk['species']}) Lv{pk['level']}  "
          f"HP:{pk['hp']}/{pk['max_hp']}  {_format_types(pk)}{nature}  "
          f"Ability:{pk['ability']}{item}{status}")
    print(f"{indent}   Stats: ATK:{pk['attack']}  DEF:{pk['defense']}  "
          f"SpATK:{pk['sp_attack']}  SpDEF:{pk['sp_defense']}  SPD:{pk['speed']}")
    moves = [f"{m['name']} (PP:{m['pp']})" for m in pk.get("moves", [])]
    print(f"{indent}   Moves: {', '.join(moves)}")
    if show_details:
        evs = pk.get("evs")
        if evs:
            print(f"{indent}   EVs:   HP:{evs['hp']}  ATK:{evs['attack']}  "
                  f"DEF:{evs['defense']}  SpATK:{evs['sp_atk']}  "
                  f"SpDEF:{evs['sp_def']}  SPD:{evs['speed']}")
        ivs = pk.get("ivs")
        if ivs:
            print(f"{indent}   IVs:   HP:{ivs['hp']}  ATK:{ivs['attack']}  "
                  f"DEF:{ivs['defense']}  SpATK:{ivs['sp_atk']}  "
                  f"SpDEF:{ivs['sp_def']}  SPD:{ivs['speed']}")


def _print_battle_mon(label: str, mon: dict):
    """Print a live battler from gBattleMons, including stat stages."""
    if not mon:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    _print_pokemon(mon, indent="    ")
    stages = mon.get("stat_stages", {})
    changed = {k: v for k, v in stages.items() if v != 0}
    if changed:
        stage_str = "  ".join(
            f"{k.replace('_', ' ').title()}:{v:+d}" for k, v in changed.items())
        print(f"       Stages: {stage_str}")
    else:
        print(f"       Stages: (all neutral)")


def print_game_state(state: dict):
    """Pretty-print a GAME_STATE dict."""
    p = state.get("player", {})
    print(f"Game:    {state.get('game', '?')}")
    print(f"Player:  {p.get('name')}  (ID: {p.get('trainer_id')})")
    print(f"Money:   ${p.get('money', 0):,}")
    print(f"Badges:  {p.get('badges')}")
    print(f"Map:     bank={p.get('map_bank')} num={p.get('map_number')}  pos=({p.get('x')}, {p.get('y')})")
    print()

    party = state.get("party", [])
    print(f"Party ({len(party)}):")
    for i, pk in enumerate(party):
        _print_pokemon(pk, index=i + 1)

    if state.get("in_battle"):
        print("\n=== IN BATTLE ===")

        battle = state.get("battle") or {}
        _print_battle_mon("Your active", battle.get("player_active"))
        _print_battle_mon("Enemy active", battle.get("enemy_active"))

        enemy_party = state.get("enemy_party") or []
        print(f"\nEnemy Party ({len(enemy_party)}):")
        for i, pk in enumerate(enemy_party):
            _print_pokemon(pk, index=i + 1, show_details=True)

    bag = state.get("bag", {})
    for pocket_name, items in bag.items():
        if items:
            print(f"\n{pocket_name.replace('_', ' ').title()} ({len(items)}):")
            for it in items:
                print(f"  {it['name']} x{it['quantity']}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def interactive(client: MGBAClient):
    """Simple interactive REPL."""
    print("Connected to mGBA server. Commands:")
    print("  tap <button> [frames]   - e.g. tap A, tap START 16")
    print("  screenshot [filename]   - save screenshot")
    print("  game_state              - print full game state")
    print("  ping                    - health check")
    print("  quit                    - exit")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "tap":
                if len(parts) < 2:
                    print("Usage: tap <button> [frames]")
                    continue
                frames = int(parts[2]) if len(parts) > 2 else None
                client.tap(parts[1], frames)
                print(f"TAP {parts[1].upper()}: OK")
            elif cmd == "screenshot":
                path = parts[1] if len(parts) > 1 else "screenshot.png"
                data = client.screenshot(path)
                print(f"Screenshot saved to {path} ({len(data)} bytes)")
            elif cmd == "ping":
                client.ping()
                print("PING: OK")
            elif cmd == "game_state":
                client.print_game_state()
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print(f"Unknown command: {cmd}")
        except (MGBAError, ValueError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Disconnected.")


def main():
    with MGBAClient() as client:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            try:
                if cmd == "tap":
                    button = sys.argv[2] if len(sys.argv) > 2 else "A"
                    frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
                    client.tap(button, frames)
                    print(f"TAP {button.upper()}: OK")
                elif cmd == "screenshot":
                    path = sys.argv[2] if len(sys.argv) > 2 else "screenshot.png"
                    data = client.screenshot(path)
                    print(f"Screenshot saved to {path} ({len(data)} bytes)")
                elif cmd == "ping":
                    client.ping()
                    print("PING: OK")
                elif cmd == "game_state":
                    client.print_game_state()
                else:
                    print(f"Unknown command: {cmd}")
            except (MGBAError, ValueError) as e:
                print(f"Error: {e}")
        else:
            interactive(client)


if __name__ == "__main__":
    main()
