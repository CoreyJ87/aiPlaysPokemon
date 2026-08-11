#!/usr/bin/env python
"""viewer/server.py - serve the run log as a live web UI for the whole LAN.

    .venv/bin/python viewer/server.py            # http://<this-mac>:8777
    .venv/bin/python viewer/server.py --port 9000

Watches the newest runs/*.jsonl (switching automatically when a new run
starts) and serves:
    /                  the React viewer page
    /api/turns?since=N turns N.. of the current log, plus which file it is
    /frame.png         the frame the model saw last (no-cache)
"""
import argparse
import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"
FRAME = RUNS / "last-frame.png"

_lock = threading.Lock()
_cache = {"file": None, "mtime": 0, "size": 0, "turns": []}


def newest_log():
    logs = list(RUNS.glob("*.jsonl"))
    if not logs:
        return None

    def run_number(p):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    numbered = [p for p in logs if run_number(p) >= 0]
    return (max(numbered, key=run_number) if numbered
            else max(logs, key=lambda p: p.stat().st_mtime))


def load_turns():
    """Parsed turns of the newest log, incrementally re-read on growth."""
    path = newest_log()
    if path is None:
        return None, []
    with _lock:
        stat = path.stat()
        if (str(path) != _cache["file"] or stat.st_size < _cache["size"]):
            _cache.update(file=str(path), size=0, turns=[])
        if stat.st_size > _cache["size"]:
            turns = []
            with path.open() as f:        # jsonl: cheap to re-read whole
                for line in f:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass              # a line mid-write; next poll gets it
            _cache.update(size=stat.st_size, turns=turns)
        return path.name, _cache["turns"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):             # quiet
        pass

    def _send(self, code, body, ctype, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = (HERE / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        elif url.path == "/api/turns":
            since = int((parse_qs(url.query).get("since") or ["0"])[0])
            name, turns = load_turns()
            payload = {"file": name, "total": len(turns),
                       "turns": turns[since:]}
            self._send(200, json.dumps(payload).encode(),
                       "application/json")
        elif url.path == "/frame.png":
            try:
                self._send(200, FRAME.read_bytes(), "image/png")
            except OSError:
                self._send(404, b"no frame yet", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"viewer up:  http://{lan_ip()}:{args.port}  (and localhost)")
    server.serve_forever()


if __name__ == "__main__":
    main()
