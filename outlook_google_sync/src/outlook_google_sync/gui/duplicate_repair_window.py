"""Ch27: Duplicate repair window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from ..config.settings_store import save_settings
from ..connectors.google_calendar import get_event
from ..sync.duplicate_merge import execute_duplicate_merge
from ..sync.duplicate_repair import DuplicateGroup
from .dialogs import DescriptionMergeDialog
from .merge_preview_window import MergePreviewWindow


class DuplicateRepairWindow(tk.Toplevel):
    def __init__(
        self,
        master,
        groups: list[DuplicateGroup],
        calendar_id: str,
        *,
        mode: str = "sync_key",
        initial_description_mode: str = "longer",
    ):
        super().__init__(master)
        self.title("重複修復ツール")
        self.geometry("960x560")
        self.calendar_id = calendar_id
        self.groups = groups
        self.mode = mode
        self.initial_description_mode = initial_description_mode
        self._group_by_parent: dict[str, DuplicateGroup] = {}

        if not groups:
            ttk.Label(self, text="重複候補はありません。").pack(pady=20)
            ttk.Button(self, text="閉じる", command=self.destroy).pack()
            return

        hint = (
            "同一 sync_key の複数行です。"
            if mode == "sync_key"
            else "同名・同開始・同終了でまとめたグループです（ツール未管理の行を含みます）。"
        )
        ttk.Label(self, text=hint, foreground="gray", wraplength=900).pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Label(
            self,
            text="場所が異なる行があるグループはマージできません。別会議の可能性に注意してください。",
            foreground="gray",
            wraplength=900,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ttk.Label(self, text=f"グループ数: {len(groups)}").pack(anchor="w", padx=8, pady=4)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(
            frame,
            columns=("sync_key", "summary", "start", "status", "event_id"),
            show="headings",
        )
        self.tree.heading("sync_key", text="sync_key / グループ")
        self.tree.heading("summary", text="件名")
        self.tree.heading("start", text="開始")
        self.tree.heading("status", text="マージ")
        self.tree.heading("event_id", text="event_id")
        self.tree.column("sync_key", width=200)
        self.tree.column("summary", width=260)
        self.tree.column("start", width=160)
        self.tree.column("status", width=100)
        self.tree.column("event_id", width=180)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._items: dict[str, dict] = {}
        for g in groups:
            sk = g.group_id[:18] if len(g.group_id) > 18 else g.group_id
            summ0 = (g.items[0].get("summary") or "")[:36] if g.items else ""
            st_txt = "可" if g.mergeable else "不可"
            if not g.mergeable and g.blocked_reason == "location_mismatch":
                st_txt = "不可(場所)"
            parent = self.tree.insert(
                "",
                "end",
                values=(sk, f"({len(g.items)}件)", "", st_txt, ""),
            )
            self._group_by_parent[parent] = g
            for item in g.items:
                s = (item.get("start") or {}).get(
                    "dateTime",
                    (item.get("start") or {}).get("date", ""),
                )
                iid = item.get("id", "")
                private = ((item.get("extendedProperties") or {}).get("private") or {})
                sk_part = (private.get("sync_key") or "")[:16]
                self.tree.insert(
                    parent,
                    "end",
                    iid=iid,
                    values=(sk_part, item.get("summary", ""), s, "", str(iid)[:24]),
                )
                self._items[str(iid)] = item

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        self._action_btns = []
        b_merge = ttk.Button(btn_frame, text="マージ（選択を残す）…", command=self._merge_preview)
        b_merge.pack(side="left", padx=4)
        self._action_btns.append(b_merge)
        b_old = ttk.Button(btn_frame, text="古い開始を残して一括マージ", command=self._keep_oldest)
        b_old.pack(side="left", padx=4)
        self._action_btns.append(b_old)
        ttk.Button(btn_frame, text="閉じる", command=self.destroy).pack(side="right", padx=4)

    def _merge_preview(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("選択", "残すイベントを1件選択してください。")
            return
        keep_id = sel[0]
        parent = self.tree.parent(keep_id)
        if not parent:
            messagebox.showwarning("選択", "グループ内の個別イベントを選択してください。")
            return
        group = self._group_by_parent.get(parent)
        if not group:
            return
        if not group.mergeable:
            messagebox.showwarning(
                "マージ不可",
                "場所が一致しないためマージできません。",
            )
            return
        MergePreviewWindow(
            self,
            self.calendar_id,
            group,
            str(keep_id),
            initial_description_mode=self.initial_description_mode,
        )

    def _keep_oldest(self) -> None:
        ops: list[tuple[DuplicateGroup, str]] = []
        for g in self.groups:
            if not g.mergeable or len(g.items) < 2:
                continue
            sorted_items = sorted(
                g.items,
                key=lambda i: (i.get("start") or {}).get("dateTime")
                or (i.get("start") or {}).get("date")
                or "",
            )
            wid = sorted_items[0].get("id")
            if wid:
                ops.append((g, str(wid)))

        if not ops:
            messagebox.showinfo("情報", "一括マージ対象のグループはありません。")
            return
        if not messagebox.askyesno(
            "確認",
            f"{len(ops)} グループで、開始が最も早い予定を残してマージします。よろしいですか？",
        ):
            return

        ddlg = DescriptionMergeDialog(
            self,
            initial=self.initial_description_mode,
        )
        if ddlg.result is None:
            return
        batch_desc_mode = ddlg.result  # type: ignore[assignment]

        for b in self._action_btns:
            b.configure(state="disabled")
        err: list[str] = []
        try:
            for g, winner_id in ops:
                full: list[dict] = []
                try:
                    for it in g.items:
                        eid = it.get("id")
                        if eid:
                            full.append(get_event(self.calendar_id, str(eid)))
                    execute_duplicate_merge(
                        self.calendar_id,
                        winner_id,
                        full,
                        batch_desc_mode,  # type: ignore[arg-type]
                    )
                except Exception as exc:
                    err.append(f"{g.group_id[:16]}: {exc}")
        finally:
            for b in self._action_btns:
                b.configure(state="normal")

        if err:
            messagebox.showerror("一部失敗", "\n".join(err[:10]))
        else:
            messagebox.showinfo("完了", f"{len(ops)} グループをマージしました。")
            top = self.winfo_toplevel()
            if hasattr(top, "state_data") and hasattr(top, "settings"):
                top.state_data["duplicate_repair_description_mode"] = batch_desc_mode
                top.settings["runtime_state"] = {
                    **top.state_data,
                    "ics_path": top.ics_var.get().strip(),
                }
                save_settings(top.settings)
        self.destroy()
