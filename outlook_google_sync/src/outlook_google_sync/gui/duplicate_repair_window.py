"""Ch27: Duplicate repair window."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox

from ..connectors.google_calendar import delete_event


class DuplicateRepairWindow(tk.Toplevel):
    def __init__(self, master, dup_map: dict[str, list[dict]], calendar_id: str):
        super().__init__(master)
        self.title("重複修復ツール")
        self.geometry("900x550")
        self.calendar_id = calendar_id
        self.dup_map = dup_map

        if not dup_map:
            ttk.Label(self, text="重複候補はありません。").pack(pady=20)
            ttk.Button(self, text="閉じる", command=self.destroy).pack()
            return

        ttk.Label(self, text=f"重複グループ: {len(dup_map)}").pack(anchor="w", padx=8, pady=4)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(
            frame,
            columns=("sync_key", "summary", "start", "event_id"),
            show="headings",
        )
        self.tree.heading("sync_key", text="sync_key")
        self.tree.heading("summary", text="件名")
        self.tree.heading("start", text="開始日時")
        self.tree.heading("event_id", text="event_id")
        self.tree.column("sync_key", width=180)
        self.tree.column("summary", width=300)
        self.tree.column("start", width=160)
        self.tree.column("event_id", width=200)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._items: dict[str, dict] = {}
        for key, items in dup_map.items():
            parent = self.tree.insert("", "end", text=key, values=(key[:16], f"({len(items)}件)", "", ""))
            for item in items:
                s = (item.get("start") or {}).get("dateTime", (item.get("start") or {}).get("date", ""))
                iid = item.get("id", "")
                self.tree.insert(parent, "end", iid=iid, values=(
                    "", item.get("summary", ""), s, iid[:20],
                ))
                self._items[iid] = item

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        self._action_btns = []
        b_del = ttk.Button(btn_frame, text="選択を残し、他を削除", command=self._delete_others)
        b_del.pack(side="left", padx=4)
        self._action_btns.append(b_del)
        b_old = ttk.Button(btn_frame, text="古い方を残す（一括）", command=self._keep_oldest)
        b_old.pack(side="left", padx=4)
        self._action_btns.append(b_old)
        ttk.Button(btn_frame, text="閉じる", command=self.destroy).pack(side="right", padx=4)

    def _delete_others(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("選択", "残すイベントを1件選択してください。")
            return

        keep_id = sel[0]
        parent = self.tree.parent(keep_id)
        if not parent:
            messagebox.showwarning("選択", "個別イベントを選択してください。")
            return

        siblings = self.tree.get_children(parent)
        to_delete = [s for s in siblings if s != keep_id and s in self._items]

        if not to_delete:
            return

        if not messagebox.askyesno("確認", f"{len(to_delete)} 件を削除します。よろしいですか？"):
            return

        for b in self._action_btns:
            b.configure(state="disabled")
        try:
            for iid in to_delete:
                try:
                    delete_event(self.calendar_id, iid)
                    self.tree.delete(iid)
                except Exception as exc:
                    messagebox.showerror("削除エラー", str(exc))
        finally:
            for b in self._action_btns:
                b.configure(state="normal")

    def _keep_oldest(self):
        """DUP-03: Keep oldest by start time."""
        to_delete: list[str] = []
        for key, items in self.dup_map.items():
            if len(items) <= 1:
                continue
            sorted_items = sorted(items, key=lambda i: (i.get("start") or {}).get("dateTime", ""))
            for item in sorted_items[1:]:
                iid = item.get("id", "")
                if iid:
                    to_delete.append(iid)

        if not to_delete:
            messagebox.showinfo("情報", "削除対象はありません。")
            return

        if not messagebox.askyesno("確認", f"各グループで最も古いものを残し、{len(to_delete)} 件を削除します。"):
            return

        for b in self._action_btns:
            b.configure(state="disabled")
        deleted = 0
        try:
            for iid in to_delete:
                try:
                    delete_event(self.calendar_id, iid)
                    try:
                        self.tree.delete(iid)
                    except Exception:
                        pass
                    deleted += 1
                except Exception as exc:
                    messagebox.showerror("削除エラー", str(exc))
        finally:
            for b in self._action_btns:
                b.configure(state="normal")

        messagebox.showinfo("完了", f"{deleted} 件削除しました。")
