"""Ch15: Preview window with individual exclude and execution bridge."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sync.preview import PreviewSnapshot


class PreviewWindow(tk.Toplevel):
    def __init__(self, master, title: str, snapshot: PreviewSnapshot, on_execute=None):
        super().__init__(master)
        self.title(title)
        self.geometry("1000x600")
        self.snapshot = snapshot
        self.on_execute = on_execute
        self.approved_indices: set[int] = set(range(len(snapshot.actions)))
        self._execute_started = False

        self._build()

    def _build(self):
        # Summary bar
        summary_frame = ttk.Frame(self)
        summary_frame.pack(fill="x", padx=8, pady=4)
        counts = {}
        for a in self.snapshot.actions:
            counts[a.action] = counts.get(a.action, 0) + 1
        parts = [f"{k}: {v}" for k, v in counts.items()]
        ttk.Label(summary_frame, text="  |  ".join(parts)).pack(anchor="w")

        # Treeview
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        cols = ("include", "action", "start", "summary", "reason", "sync_key", "method")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("include", text="含む")
        self.tree.heading("action", text="アクション")
        self.tree.heading("start", text="開始日時")
        self.tree.heading("summary", text="件名")
        self.tree.heading("reason", text="理由")
        self.tree.heading("sync_key", text="sync_key")
        self.tree.heading("method", text="入力/エンジン")

        self.tree.column("include", width=40, anchor="center")
        self.tree.column("action", width=80)
        self.tree.column("start", width=140)
        self.tree.column("summary", width=250)
        self.tree.column("reason", width=200)
        self.tree.column("sync_key", width=140)
        self.tree.column("method", width=100)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, a in enumerate(self.snapshot.actions):
            e = a.event
            g = a.google_item or {}
            start_str = ""
            if e:
                start_str = e.start.strftime("%Y-%m-%d %H:%M") if e.start else ""
            elif g:
                s = g.get("start", {})
                start_str = s.get("dateTime", s.get("date", ""))

            summary = e.summary if e else g.get("summary", "")
            sync_key = (e.sync_key[:16] if e else
                        ((g.get("extendedProperties") or {}).get("private") or {}).get("sync_key", "")[:16])
            method = f"{e.input_method}/{e.reader_engine}" if e else ""

            self.tree.insert("", "end", iid=str(i), values=(
                "✓", a.action, start_str, summary, a.reason, sync_key, method,
            ))

        self.tree.bind("<Button-1>", self._toggle_include)

        # Buttons（本番実行中は誤操作防止で主要操作のみ無効化。閉じるは常に使える）
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        self._preview_action_btns = []
        b1 = ttk.Button(btn_frame, text="一括承認", command=self._approve_all)
        b1.pack(side="left", padx=4)
        self._preview_action_btns.append(b1)

        if self.on_execute:
            b2 = ttk.Button(btn_frame, text="本番同期実行", command=self._on_execute)
            b2.pack(side="right", padx=4)
            self._preview_action_btns.append(b2)

        ttk.Button(btn_frame, text="閉じる", command=self.destroy).pack(side="right", padx=4)

    def _toggle_include(self, event):
        region = self.tree.identify("column", event.x, event.y)
        if region != "#1":
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        idx = int(iid)
        if idx in self.approved_indices:
            self.approved_indices.discard(idx)
            vals = list(self.tree.item(iid, "values"))
            vals[0] = ""
            self.tree.item(iid, values=vals)
        else:
            self.approved_indices.add(idx)
            vals = list(self.tree.item(iid, "values"))
            vals[0] = "✓"
            self.tree.item(iid, values=vals)

    def _approve_all(self):
        self.approved_indices = set(range(len(self.snapshot.actions)))
        for i in range(len(self.snapshot.actions)):
            vals = list(self.tree.item(str(i), "values"))
            vals[0] = "✓"
            self.tree.item(str(i), values=vals)

    def _on_execute(self):
        if self._execute_started:
            return
        if self.snapshot.is_stale():
            answer = messagebox.askyesno(
                "プレビュー結果が古い可能性",
                "プレビュー結果が古くなった可能性があります。再プレビューしますか？\n"
                "「いいえ」で現在のスナップショットのまま実行します。",
            )
            if answer:
                self.destroy()
                return

        if messagebox.askyesno("確認", "本番同期を実行しますか？"):
            self._execute_started = True
            for b in self._preview_action_btns:
                b.configure(state="disabled")
            self.on_execute(self.snapshot, self.approved_indices)
            self.destroy()
