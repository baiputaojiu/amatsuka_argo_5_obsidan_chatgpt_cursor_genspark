"""Shared ttk styles: clearer enabled vs disabled button appearance."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def apply_button_contrast_style(master: tk.Misc) -> None:
    """Make TButton text dark when enabled and gray when disabled (ttk state)."""
    style = ttk.Style(master)
    try:
        style.configure("TButton", foreground="#000000")
        style.map(
            "TButton",
            foreground=[
                ("disabled", "#808080"),
                ("!disabled", "#000000"),
            ],
        )
    except tk.TclError:
        pass
