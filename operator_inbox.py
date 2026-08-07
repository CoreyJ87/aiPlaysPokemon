"""
operator_inbox.py - the mailbox between whatever is asking player_ai for
things (a manual GUI today, chat via the avatar later) and the turn loop.

One object, shared by reference, same pattern as avatar_controller.Avatar:
construct it once, hand the same instance to both sides, and only ever talk
to each other through it. The GUI thread calls submit()/pause()/resume(); the
play loop calls it once per turn to see what's active and whether to hold.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import time


@dataclass
class OperatorRequest:
    text: str
    submittedAt: float = field(default_factory=time)


class OperatorInbox:
    """Thread-safe. Call the writers from any thread, the readers from the play loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._paused = False
        self._stopped = False
        self._request: OperatorRequest | None = None

    # ---- writers (GUI thread) ----------------------------------------

    def submit(self, text: str):
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._request = OperatorRequest(text=text)

    def clear(self):
        with self._lock:
            self._request = None

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def stop(self):
        """The GUI window closed; tell the play loop to wind down."""
        with self._lock:
            self._stopped = True

    # ---- readers (play loop) -------------------------------------------

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def request(self) -> OperatorRequest | None:
        with self._lock:
            return self._request
