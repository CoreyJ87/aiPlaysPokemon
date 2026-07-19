"""
mgba_client.py — Example Python client for mgba_server.lua

Usage:
    python mgba_client.py              # interactive mode
    python mgba_client.py tap A        # tap A button (default 8 frames)
    python mgba_client.py tap A 16     # tap A button for 16 frames
    python mgba_client.py screenshot   # save screenshot to screenshot.png
    python mgba_client.py game_state   # print full game state as JSON
    python mgba_client.py ping         # health check

Multiple clients can connect simultaneously.
"""

import json
import socket
import sys
import struct

HOST = "127.0.0.1"
PORT = 54321


def send_command(sock: socket.socket, command: str) -> str:
    """Send a newline-terminated command string and return the response line."""
    sock.sendall((command + "\n").encode("utf-8"))
    # Read until we get a newline (the response header)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed connection")
        buf += chunk
    nl = buf.index(b"\n")
    header = buf[:nl].decode("utf-8")
    remainder = buf[nl + 1:]
    return header, remainder


def tap(sock: socket.socket, button: str, frames: int = None):
    """Send a button tap command."""
    cmd = f"TAP|{button}"
    if frames is not None:
        cmd += f"|{frames}"
    header, _ = send_command(sock, cmd)
    print(f"TAP {button}: {header}")


def screenshot(sock: socket.socket, output_path: str = "screenshot.png"):
    """Request a screenshot and save it to a file."""
    sock.sendall(b"SCREENSHOT\n")

    # Read response header: OK|<byte_length>\n
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed connection")
        buf += chunk

    nl = buf.index(b"\n")
    header = buf[:nl].decode("utf-8")
    remainder = buf[nl + 1:]

    if header.startswith("ERR"):
        print(f"Screenshot failed: {header}")
        return

    # Parse byte length
    parts = header.split("|")
    byte_length = int(parts[1])

    # Read the PNG data
    png_data = remainder
    while len(png_data) < byte_length:
        chunk = sock.recv(min(65536, byte_length - len(png_data)))
        if not chunk:
            raise ConnectionError("Server closed connection during transfer")
        png_data += chunk

    with open(output_path, "wb") as f:
        f.write(png_data[:byte_length])
    print(f"Screenshot saved to {output_path} ({byte_length} bytes)")


def ping(sock: socket.socket):
    """Send a ping command."""
    header, _ = send_command(sock, "PING")
    print(f"PING: {header}")


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


def game_state(sock: socket.socket):
    """Request the full game state and pretty-print it."""
    header, _ = send_command(sock, "GAME_STATE")
    if header.startswith("ERR"):
        print(f"GAME_STATE failed: {header}")
        return
    parts = header.split("|", 1)
    if len(parts) < 2:
        print(f"GAME_STATE: unexpected response: {header}")
        return
    state = json.loads(parts[1])

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


def interactive(sock: socket.socket):
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
                tap(sock, parts[1], frames)
            elif cmd == "screenshot":
                path = parts[1] if len(parts) > 1 else "screenshot.png"
                screenshot(sock, path)
            elif cmd == "ping":
                ping(sock)
            elif cmd == "game_state":
                game_state(sock)
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Disconnected.")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    try:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            if cmd == "tap":
                button = sys.argv[2] if len(sys.argv) > 2 else "A"
                frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
                tap(sock, button, frames)
            elif cmd == "screenshot":
                path = sys.argv[2] if len(sys.argv) > 2 else "screenshot.png"
                screenshot(sock, path)
            elif cmd == "ping":
                ping(sock)
            elif cmd == "game_state":
                game_state(sock)
            else:
                print(f"Unknown command: {cmd}")
        else:
            interactive(sock)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
