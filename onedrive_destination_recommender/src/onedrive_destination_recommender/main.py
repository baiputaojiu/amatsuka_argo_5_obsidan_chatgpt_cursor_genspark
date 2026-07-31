import tkinter as tk
from tkinter import ttk

APP_TITLE = "OneDrive保存先レコメンダー"


def create_main_window() -> tk.Tk:
    """Create the minimal Step 0 window without starting the event loop."""
    root = tk.Tk()
    root.title(APP_TITLE)
    root.minsize(560, 220)

    content = ttk.Frame(root, padding=24)
    content.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)

    ttk.Label(content, text=APP_TITLE, font=("Yu Gothic UI", 16, "bold")).grid(
        row=0,
        column=0,
        sticky="w",
    )
    ttk.Label(
        content,
        text="Step 3：MSG読込と検索語生成を確認しました。画面操作は後続Stepで実装します。",
        wraplength=500,
    ).grid(row=1, column=0, sticky="w", pady=(20, 0))

    return root


def main() -> None:
    """Run the desktop application."""
    root = create_main_window()
    root.mainloop()
