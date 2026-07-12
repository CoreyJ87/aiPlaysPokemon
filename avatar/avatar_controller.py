"""
avatar_controller.py
====================
Runs a WebSocket server that the OBS browser source (avatar.html) connects to,
and gives you a simple, synchronous Python API for controlling the avatar.

Requires:  pip install websockets

Quick start
-----------
    from avatar_controller import Avatar

    avatar = Avatar()          # starts the server on ws://localhost:8765
    avatar.set_mood("happy")
    avatar.look(0.5, -0.3)     # x, y each in -1..1 (right/down positive)
    avatar.set_eye_color("#ff5555")
    avatar.set_float_speed(2.0)
    avatar.stop_float()

The server keeps the current state, so if OBS (re)connects at any time it
immediately receives the latest state — you never have to worry about
ordering between starting your script and starting OBS.
"""

import asyncio
import json
import threading

import websockets

MOODS = {"neutral", "happy", "sad", "afk", "stressed"}


class Avatar:
    """Controls the avatar overlay. Thread-safe; safe to call from sync code."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host = host
        self._port = port
        self._clients = set()
        self._state = {
            "mood": "neutral",
            "eye_color": "#ffffff",
            "look": {"x": 0.0, "y": 0.0},
            "float_speed": 1.0,
        }
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_mood(self, mood: str):
        """One of: neutral, happy, sad, afk, stressed."""
        if mood not in MOODS:
            raise ValueError(f"mood must be one of {sorted(MOODS)}")
        self._update(mood=mood)

    def set_eye_color(self, hex_color: str):
        """e.g. '#ff5555'. The overlay fades to the new color smoothly."""
        self._update(eye_color=hex_color)

    def look(self, x: float, y: float):
        """Look direction. x: -1 (left) .. 1 (right), y: -1 (up) .. 1 (down)."""
        self._update(look={"x": float(x), "y": float(y)})

    def look_center(self):
        self.look(0, 0)

    def set_float_speed(self, speed: float):
        """0 stops the bobbing; 1 is normal; higher is faster."""
        self._update(float_speed=max(0.0, float(speed)))

    def stop_float(self):
        self.set_float_speed(0.0)

    def set_state(self, *, mood=None, eye_color=None, look=None, float_speed=None):
        """Set several properties in a single message."""
        updates = {}
        if mood is not None:
            if mood not in MOODS:
                raise ValueError(f"mood must be one of {sorted(MOODS)}")
            updates["mood"] = mood
        if eye_color is not None:
            updates["eye_color"] = eye_color
        if look is not None:
            updates["look"] = {"x": float(look[0]), "y": float(look[1])}
        if float_speed is not None:
            updates["float_speed"] = max(0.0, float(float_speed))
        if updates:
            self._update(**updates)

    # Convenience presets that pair a mood with a fitting eye color
    def happy(self):    self.set_state(mood="happy", eye_color="#9ed7f0", float_speed=1.5)
    def sad(self):      self.set_state(mood="sad", eye_color="#4169e1", float_speed=0.5)
    def angry(self):    self.set_state(mood="stressed", eye_color="#ff3b30", float_speed=2.0)
    def asleep(self):   self.set_state(mood="afk", eye_color="#ffffff", float_speed=0.0)
    def neutral(self):  self.set_state(mood="neutral", eye_color="#ffffff", float_speed=1.0)
    def off(self):      self.set_state(mood="stressed", eye_color="#ffffff", float_speed=0.0)

    @property
    def connected_clients(self) -> int:
        """How many overlay pages are currently connected."""
        return len(self._clients)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update(self, **updates):
        self._state.update(updates)
        asyncio.run_coroutine_threadsafe(self._broadcast(updates), self._loop)

    async def _broadcast(self, updates):
        if not self._clients:
            return
        message = json.dumps(updates)
        await asyncio.gather(
            *(self._safe_send(ws, message) for ws in list(self._clients)),
            return_exceptions=True,
        )

    @staticmethod
    async def _safe_send(ws, message):
        try:
            await ws.send(message)
        except websockets.ConnectionClosed:
            pass

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            # New client gets the full current state right away
            await websocket.send(json.dumps(self._state))
            async for _ in websocket:  # we don't expect messages; keep alive
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)

        async def start():
            await websockets.serve(self._handler, self._host, self._port)
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()


# ----------------------------------------------------------------------
# Demo: cycles through moods so you can watch it in OBS
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    avatar = Avatar()
    print("Avatar server running on ws://localhost:8765")
    print("Add avatar.html as a Browser Source in OBS, then watch the demo loop.")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            print("neutral, looking around...")
            avatar.neutral()
            for lx, ly in [(0.8, 0), (-0.8, 0), (0, -0.8), (0.4, 0.6), (0, 0)]:
                avatar.look(lx, ly)
                time.sleep(3)

            print("happy!")
            avatar.happy()
            #avatar.set_float_speed(2.0)
            time.sleep(3)

            print("off...")
            avatar.off()
            time.sleep(3)

            print("sad...")
            avatar.sad()
            #avatar.set_float_speed(0.5)
            avatar.look(0, 0.5)
            time.sleep(3)

            print("stressed/angry!")
            avatar.angry()
            #avatar.set_float_speed(3.0)
            time.sleep(3)

            print("AFK/asleep...")
            avatar.asleep()
            time.sleep(3)
    except KeyboardInterrupt:
        print("bye!")
