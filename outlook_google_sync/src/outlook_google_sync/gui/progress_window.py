"""Simple progress dialog for long-running operations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ProgressWindow(tk.Toplevel):
    def __init__(self, master, title: str = "処理中"):
        super().__init__(master)
        self.title(title)
        self.geometry("520x130")
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        self.msg_var = tk.StringVar(value="処理を開始しています...")
        ttk.Label(frame, textvariable=self.msg_var, wraplength=480).pack(anchor="w", pady=(0, 8))

        self.pb = ttk.Progressbar(frame, mode="indeterminate", maximum=100)
        self.pb.pack(fill="x")
        self.pb.start(12)

    def set_indeterminate(self, message: str) -> None:
        self.msg_var.set(message)
        if str(self.pb.cget("mode")) != "indeterminate":
            self.pb.stop()
            self.pb.configure(mode="indeterminate")
            self.pb.start(12)

    def set_progress(self, current: int, total: int, message: str) -> None:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        self.msg_var.set(message)
        if str(self.pb.cget("mode")) != "determinate":
            self.pb.stop()
            self.pb.configure(mode="determinate")
        self.pb.configure(maximum=total, value=current)

    def close_safe(self) -> None:
        try:
            self.pb.stop()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
