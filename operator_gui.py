"""
operator_gui.py - manual stand-in for the avatar: a small window an operator
uses to pause player_ai and hand it a short-term request.

Tkinter wants the main thread, so this owns it; player_ai's turn loop runs on
a background thread started by player_ai.py's `gui` mode. Both sides only
ever touch the shared OperatorInbox (see operator_inbox.py), never each
other - so this window could be replaced by the avatar's chat-polling code
later without player_ai.py changing at all.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from operator_inbox import OperatorInbox


class OperatorGui:
    def __init__(self, inbox: OperatorInbox):
        self.inbox = inbox
        self.root = tk.Tk()
        self.root.title("player_ai - operator console")
        self.root.geometry("480x340")
        self.root.protocol("WM_DELETE_WINDOW", self._onClose)

        self.status = tk.StringVar(value="running")
        ttk.Label(self.root, textvariable=self.status,
                 font=("", 12, "bold")).pack(pady=(10, 4))

        self.pauseButton = ttk.Button(self.root, text="Pause",
                                      command=self._togglePause)
        self.pauseButton.pack(pady=4)

        ttk.Label(self.root, text='Request for the model, e.g. "catch a '
                                  'pidgey and name it Big Bird":'
                 ).pack(anchor="w", padx=10, pady=(12, 0))
        self.entry = tk.Text(self.root, height=4, wrap="word")
        self.entry.pack(fill="x", padx=10, pady=6)
        self.entry.bind("<Return>", self._onReturn)

        buttonRow = ttk.Frame(self.root)
        buttonRow.pack(pady=2)
        ttk.Button(buttonRow, text="Submit",
                  command=self._submit).pack(side="left", padx=4)
        ttk.Button(buttonRow, text="Clear active request",
                  command=self._clear).pack(side="left", padx=4)

        self.activeVar = tk.StringVar(value="(no active request)")
        self.activeLabel = ttk.Label(self.root, textvariable=self.activeVar,
                                     wraplength=440, foreground="#555")
        self.activeLabel.pack(padx=10, pady=(14, 0), anchor="w")

        self.reactionVar = tk.StringVar(value="")
        self.reactionLabel = ttk.Label(self.root, textvariable=self.reactionVar,
                                       wraplength=440, foreground="#a33",
                                       font=("", 9, "italic"))
        self.reactionLabel.pack(padx=10, pady=(4, 0), anchor="w")

        ttk.Label(self.root, text="The model's thoughts and actions are "
                                  "printed in the terminal, not here.",
                 foreground="#888").pack(side="bottom", pady=8)

        self._poll()

    def _onReturn(self, event):
        # Shift+Enter still inserts a newline; a bare Enter submits.
        if not (event.state & 0x0001):
            self._submit()
            return "break"

    def _togglePause(self):
        if self.inbox.paused:
            self.inbox.resume()
        else:
            self.inbox.pause()

    def _submit(self):
        text = self.entry.get("1.0", "end").strip()
        if text:
            self.inbox.submit(text)
            self.entry.delete("1.0", "end")

    def _clear(self):
        self.inbox.clear()

    def _onClose(self):
        self.inbox.stop()
        self.root.destroy()

    def _poll(self):
        self.status.set("PAUSED" if self.inbox.paused else "running")
        self.pauseButton.config(text="Resume" if self.inbox.paused else "Pause")

        if self.inbox.pending is not None:
            self.activeVar.set(f'Checking: "{self.inbox.pending.text}" ...')
            self.reactionVar.set("")
        else:
            req = self.inbox.request
            self.activeVar.set(f'Active request: "{req.text}"' if req
                               else "(no active request)")
            verdict = self.inbox.verdict
            if verdict is not None and not verdict.accepted:
                reaction = f" {verdict.reaction}" if verdict.reaction else ""
                self.reactionVar.set(f"Rejected: {verdict.reason}.{reaction}")
            else:
                self.reactionVar.set("")

        self.root.after(300, self._poll)

    def run(self):
        self.root.mainloop()
