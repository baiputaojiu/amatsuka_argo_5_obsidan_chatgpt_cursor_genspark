"""Startup setup checklist dialog."""

from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from ..config.paths import runtime_dir
from ..services.setup_checklist import SetupItem, evaluate_setup, is_setup_incomplete


class SetupChecklistDialog(tk.Toplevel):
    """Modal checklist shown when credentials or token is missing."""

    def __init__(
        self,
        master,
        state: dict,
        *,
        on_google_auth: Callable[[], None] | None = None,
        on_open_settings: Callable[[], None] | None = None,
    ):
        super().__init__(master)
        self.title("初回セットアップ")
        self.geometry("640x420")
        self.transient(master)
        self.grab_set()
        self.state = state
        self.on_google_auth = on_google_auth
        self.on_open_settings = on_open_settings
        self._row_widgets: dict[str, dict] = {}

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Google 連携に必要な準備を確認してください。",
            font=("", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            root,
            text=(
                "1) Google Cloud Console で OAuth クライアント（デスクトップ）を作成\n"
                "2) JSON を credentials.json としてランタイムフォルダへ配置\n"
                "3) Google 認証を実行（token.json が作成されます）"
            ),
            justify="left",
            wraplength=600,
        ).pack(anchor="w", pady=(8, 12))

        self.list_frame = ttk.Frame(root)
        self.list_frame.pack(fill="both", expand=True)

        bf = ttk.Frame(root)
        bf.pack(fill="x", pady=(12, 0))
        ttk.Button(bf, text="後で", command=self.destroy).pack(side="right")

        self._refresh()

    def _refresh(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._row_widgets.clear()

        for item in evaluate_setup(self.state):
            self._add_row(item)

        if not is_setup_incomplete(self.state):
            self.after(100, self.destroy)

    def _add_row(self, item: SetupItem) -> None:
        row = ttk.Frame(self.list_frame)
        row.pack(fill="x", pady=6)

        status = "済" if item.done else "未"
        ttk.Label(row, text=f"[{status}]", width=4).pack(side="left")
        mid = ttk.Frame(row)
        mid.pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Label(mid, text=item.label).pack(anchor="w")
        ttk.Label(mid, text=item.hint, foreground="gray", wraplength=420).pack(anchor="w")

        if item.id == "credentials":
            ttk.Button(row, text="フォルダを開く", command=self._open_runtime_dir).pack(side="right")
        elif item.id == "token":
            btn = ttk.Button(row, text="Google 認証", command=self._run_auth)
            btn.pack(side="right")
            if item.done:
                btn.configure(state="disabled")
        elif item.id == "calendar":
            ttk.Button(row, text="設定を開く", command=self._open_settings).pack(side="right")

        self._row_widgets[item.id] = {"frame": row}

    def _open_runtime_dir(self) -> None:
        path = runtime_dir()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("エラー", f"フォルダを開けませんでした:\n{path}\n{exc}", parent=self)
        else:
            messagebox.showinfo(
                "配置先",
                f"次のフォルダへ credentials.json を配置してください:\n{path}",
                parent=self,
            )
        self._refresh()

    def _run_auth(self) -> None:
        if not self.on_google_auth:
            messagebox.showwarning("未設定", "認証処理が接続されていません。", parent=self)
            return
        try:
            self.on_google_auth()
        except Exception as exc:  # noqa: BLE001 — surface any auth failure to the user
            messagebox.showerror("認証エラー", str(exc), parent=self)
        self._refresh()

    def _open_settings(self) -> None:
        if self.on_open_settings:
            self.on_open_settings()
        self._refresh()
