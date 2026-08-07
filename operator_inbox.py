"""
operator_inbox.py - the mailbox between whatever is asking player_ai for
things (a manual GUI today, chat via the avatar later) and the turn loop.

Requests flow through two stops: submit() drops a request in as `pending`;
the feasibility worker (feasibility.py) claims it, judges it, and either
promotes it to `active` (what the model sees as liveRequest) or rejects it
with a reason and, in the model's stage-2 case, a reaction line the avatar
could speak. Keeping pending and active separate means a request that's
still being judged doesn't blank out whatever the model was already working
on.

One object, shared by reference, same pattern as avatar_controller.Avatar:
construct it once, hand the same instance to every side, and only ever talk
to each other through it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import time


@dataclass
class OperatorRequest:
    text: str
    submittedAt: float = field(default_factory=time)


@dataclass
class Verdict:
    accepted: bool
    reason: str = ""      # why, for whoever asked
    reaction: str = ""    # a short in-character line - what the avatar would say
    easterEgg: bool = False   # a hidden trigger fired, not an ordinary rejection


class OperatorInbox:
    """Thread-safe. Writers: the GUI and the feasibility worker. Reader: the play loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._paused = False
        self._stopped = False
        self._pending: OperatorRequest | None = None   # awaiting a verdict
        self._active: OperatorRequest | None = None     # accepted, visible to the model
        self._verdict: Verdict | None = None             # about the most recent request

    # ---- operator side (GUI thread) ------------------------------------

    def submit(self, text: str):
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._pending = OperatorRequest(text=text)
            self._verdict = None

    def clear(self):
        with self._lock:
            self._pending = None
            self._active = None
            self._verdict = None

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def stop(self):
        """The GUI window closed; tell the play loop and workers to wind down."""
        with self._lock:
            self._stopped = True

    # ---- feasibility worker side ----------------------------------------

    def takePending(self) -> OperatorRequest | None:
        """Claim the next request awaiting a verdict; None if there isn't one."""
        with self._lock:
            req, self._pending = self._pending, None
            return req

    def accept(self, request: OperatorRequest):
        with self._lock:
            self._active = request
            self._verdict = Verdict(accepted=True)

    def reject(self, request: OperatorRequest, reason: str, reaction: str = "",
              easterEgg: bool = False):
        with self._lock:
            self._verdict = Verdict(accepted=False, reason=reason,
                                    reaction=reaction, easterEgg=easterEgg)

    # ---- play loop / GUI readers -----------------------------------------

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
        """The accepted request - what the model should see this turn, if any."""
        with self._lock:
            return self._active

    @property
    def pending(self) -> OperatorRequest | None:
        with self._lock:
            return self._pending

    @property
    def verdict(self) -> Verdict | None:
        with self._lock:
            return self._verdict
