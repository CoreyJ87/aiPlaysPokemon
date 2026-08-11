"""Live viewer for player_ai JSONL logs: pretty-prints each turn as it lands.

    python runs/watch.py                    # newest runs/*.jsonl automatically
    python runs/watch.py runs/run-2.jsonl   # or a specific one
"""
import json
import re
import sys
import time
from pathlib import Path

RUNS_DIR = Path(__file__).parent


def newest_log() -> Path:
    """The highest-numbered run-N.jsonl, falling back to newest mtime."""
    logs = sorted(RUNS_DIR.glob("*.jsonl"))
    if not logs:
        sys.exit(f"no .jsonl logs in {RUNS_DIR}")

    def run_number(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    numbered = [p for p in logs if run_number(p) >= 0]
    if numbered:
        return max(numbered, key=run_number)
    return max(logs, key=lambda p: p.stat().st_mtime)


path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_log()
print(f"watching {path}\n")

with path.open() as f:
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.5)
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        print("=" * 72)
        print(f"TURN {t['turn']}   map {t.get('map')} tile {t.get('tile')}"
              f"{'   IN BATTLE' if t.get('in_battle') else ''}")
        print("-" * 72)
        print(t.get("report", "").rstrip())
        print("-" * 72)
        print("REPLY:", t.get("reply", "").strip())
        print(f"ACTION: {t.get('action')}   ->   {t.get('result')}")
        print()
