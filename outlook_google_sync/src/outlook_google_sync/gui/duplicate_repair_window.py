"""Ch27: Duplicate repair window."""

from __future__ import annotations

import tkinter as tk
import threading
from tkinter import messagebox, ttk

from ..config.settings_store import save_settings
from ..connectors.google_calendar import get_event
from ..models.google_event import GoogleEventView
from ..sync.duplicate_merge import execute_duplicate_merge
from ..sync.duplicate_repair import DuplicateGroup
from .progress_window import ProgressWindow

_CHOICES = {
    "skip": "マージしない",
    "longer": "より長い説明文",
    "concat": "区切り付きで連結",
}
_CHOICES_INV = {v: k for k, v in _CHOICES.items()}


def _start_sort_key(item: dict) -> str:
    return GoogleEventView(item).start_value


def _fmt_attendees(item: dict) -> str:
    at = item.get("attendees")
    if not at:
        return "（なし）"
    lines = []
    for a in at:
        email = a.get("email", "")
        name = a.get("displayName", "")
        status = a.get("responseStatus", "")
        prefix = f"{name} " if name else ""
        lines.append(f"{prefix}{email} ({status})".strip())
    return "\n".join(lines) if lines else "（なし）"


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
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.calendar_id = calendar_id
        self.groups = groups
        self.mode = mode
        self.initial_description_mode = initial_description_mode
        self._group_by_parent: dict[str, DuplicateGroup] = {}
        self._group_choice: dict[str, tk.StringVar] = {}
        self._group_full_cache: dict[str, list[dict]] = {}
        self._progress_win: ProgressWindow | None = None

        if not groups:
            ttk.Label(self, text="重複候補はありません。").pack(pady=20)
            ttk.Button(self, text="閉じる", command=self.destroy).pack()
            return

        hint = (
            "同一 sync_key の複数行です。"
            if mode == "sync_key"
            else "同名・同開始・同終了でまとめたグループです（ツール未管理の行を含みます）。"
        )
        ttk.Label(self, text=hint, foreground="gray", wraplength=1100).pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Label(
            self,
            text="場所が異なる行があるグループはマージできません。別会議の可能性に注意してください。",
            foreground="gray",
            wraplength=1100,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(self, text=f"グループ数: {len(groups)}").pack(anchor="w", padx=8, pady=4)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        self._build_tree(left)
        self._build_choice_list(left)
        self._build_preview(right)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        self._action_btns: list[ttk.Button] = []
        b_run = ttk.Button(btn_frame, text="選択内容で一括マージを実行", command=self._run_batch)
        b_run.pack(side="left", padx=4)
        self._action_btns.append(b_run)
        ttk.Button(btn_frame, text="閉じる", command=self.destroy).pack(side="right", padx=4)

    def _build_tree(self, parent: ttk.Frame) -> None:
        tree_wrap = ttk.LabelFrame(parent, text="重複グループ")
        tree_wrap.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            tree_wrap,
            columns=("sync_key", "summary", "start", "status", "event_id"),
            show="headings",
        )
        self.tree.heading("sync_key", text="sync_key / グループ")
        self.tree.heading("summary", text="件名")
        self.tree.heading("start", text="開始")
        self.tree.heading("status", text="マージ")
        self.tree.heading("event_id", text="event_id")
        self.tree.column("sync_key", width=190)
        self.tree.column("summary", width=220)
        self.tree.column("start", width=150)
        self.tree.column("status", width=90)
        self.tree.column("event_id", width=160)
        y = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y.set)
        self.tree.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        for g in self.groups:
            gid = g.group_id
            sk = gid[:18] if len(gid) > 18 else gid
            st_txt = "可" if g.mergeable else "不可(場所)"
            parent_iid = self.tree.insert("", "end", iid=gid, values=(sk, f"({len(g.items)}件)", "", st_txt, ""))
            self._group_by_parent[parent_iid] = g
            default = self.initial_description_mode if g.mergeable else "skip"
            self._group_choice[gid] = tk.StringVar(value=default)
            for item in g.items:
                view = GoogleEventView(item)
                iid = view.id
                sk_part = view.sync_key[:16]
                self.tree.insert(
                    parent_iid,
                    "end",
                    iid=f"{gid}:{iid}",
                    values=(sk_part, view.summary, view.start_value, "", iid[:24]),
                )

    def _build_choice_list(self, parent: ttk.Frame) -> None:
        wrap = ttk.LabelFrame(parent, text="グループ別マージ方法")
        wrap.pack(fill="x", pady=(8, 0))
        canvas = tk.Canvas(wrap, height=200)
        y = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=y.set)
        canvas.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")

        for g in self.groups:
            row = ttk.Frame(inner)
            row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=f"{g.group_id[:14]}… ({len(g.items)}件)", width=26).pack(side="left")
            state = "readonly" if g.mergeable else "disabled"
            cb = ttk.Combobox(
                row,
                width=18,
                state=state,
                values=[_CHOICES["skip"], _CHOICES["longer"], _CHOICES["concat"]],
            )
            cb.pack(side="left", padx=(4, 0))
            cb.set(_CHOICES[self._group_choice[g.group_id].get()])
            cb.bind("<<ComboboxSelected>>", lambda _e, gid=g.group_id, w=cb: self._on_choice_changed(gid, w.get()))

    def _build_preview(self, parent: ttk.Frame) -> None:
        wrap = ttk.LabelFrame(parent, text="選択グループのプレビュー（件名/開始/終了/場所/説明/参加者）")
        wrap.pack(fill="both", expand=True)
        self.preview = tk.Text(wrap, wrap="none")
        y = ttk.Scrollbar(wrap, orient="vertical", command=self.preview.yview)
        x = ttk.Scrollbar(wrap, orient="horizontal", command=self.preview.xview)
        self.preview.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self.preview.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.preview.insert("1.0", "左側でグループを選択すると詳細が表示されます。")
        self.preview.configure(state="disabled")

    def _on_choice_changed(self, group_id: str, label: str) -> None:
        self._group_choice[group_id].set(_CHOICES_INV.get(label, "skip"))

    def _selected_group(self) -> DuplicateGroup | None:
        sel = self.tree.selection()
        if not sel:
            return None
        iid = sel[0]
        # Group IDs can contain ":" (e.g. "content:..."), so avoid string split.
        if iid in self._group_by_parent:
            return self._group_by_parent.get(iid)
        parent = self.tree.parent(iid)
        if parent and parent in self._group_by_parent:
            return self._group_by_parent.get(parent)
        return None

    def _load_group_full(self, group: DuplicateGroup) -> list[dict]:
        if group.group_id in self._group_full_cache:
            return self._group_full_cache[group.group_id]
        full: list[dict] = []
        for it in group.items:
            eid = it.get("id")
            if eid:
                full.append(get_event(self.calendar_id, str(eid)))
        self._group_full_cache[group.group_id] = full
        return full

    def _on_tree_select(self, _event=None) -> None:
        group = self._selected_group()
        if not group:
            return
        try:
            full = self._load_group_full(group)
        except Exception as exc:
            self._set_preview_text(f"取得エラー: {exc}")
            return
        lines = []
        lines.append(f"グループ: {group.group_id}")
        lines.append(f"マージ可否: {'可' if group.mergeable else '不可'}")
        lines.append("-" * 88)
        for i, ev in enumerate(full, start=1):
            view = GoogleEventView(ev)
            lines.append(f"[{i}] id={view.id}")
            lines.append(f"件名: {view.summary}")
            lines.append(f"開始: {view.start_value}")
            lines.append(f"終了: {view.end_value}")
            lines.append(f"場所: {view.location.strip() or '（なし）'}")
            lines.append("説明:")
            lines.append(view.description.strip() or "（なし）")
            lines.append("参加者:")
            lines.append(_fmt_attendees(ev))
            lines.append("-" * 88)
        self._set_preview_text("\n".join(lines))

    def _set_preview_text(self, text: str) -> None:
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _persist_last_mode(self, mode: str) -> None:
        top = self.winfo_toplevel()
        if not hasattr(top, "state_data") or not hasattr(top, "settings") or not hasattr(top, "ics_var"):
            return
        top.state_data["duplicate_repair_description_mode"] = mode
        top.settings["runtime_state"] = {
            **top.state_data,
            "ics_path": top.ics_var.get().strip(),
        }
        save_settings(top.settings)

    def _open_progress(self, message: str) -> None:
        self._close_progress()
        self._progress_win = ProgressWindow(self, title="重複修復中")
        self._progress_win.set_indeterminate(message)

    def _close_progress(self) -> None:
        if self._progress_win is not None:
            self._progress_win.close_safe()
            self._progress_win = None

    def _set_progress(self, current: int, total: int, message: str) -> None:
        if self._progress_win is None:
            return
        self._progress_win.set_progress(current, total, message)

    def _run_batch(self) -> None:
        ops: list[tuple[DuplicateGroup, str]] = []
        blocked: list[str] = []
        last_mode = None
        for g in self.groups:
            choice = self._group_choice[g.group_id].get()
            if choice == "skip":
                continue
            if not g.mergeable:
                blocked.append(g.group_id[:16])
                continue
            sorted_items = sorted(g.items, key=_start_sort_key)
            winner = str(sorted_items[0].get("id", "")) if sorted_items else ""
            if not winner:
                continue
            ops.append((g, choice))
            last_mode = choice

        if not ops and not blocked:
            messagebox.showinfo("情報", "マージ対象がありません。各グループのプルダウンを設定してください。")
            return
        if not messagebox.askyesno("確認", f"{len(ops)} グループを一括マージします。よろしいですか？"):
            return

        for b in self._action_btns:
            b.configure(state="disabled")
        self._open_progress("一括マージを開始しています...")

        def run_batch():
            merged_count = 0
            err: list[str] = []
            total = len(ops)
            for idx, (g, mode) in enumerate(ops, start=1):
                self.after(0, lambda i=idx, t=total, gid=g.group_id: self._set_progress(i - 1, t, f"処理中 {i}/{t}: {gid[:18]}"))
                try:
                    full = self._load_group_full(g)
                    sorted_full = sorted(full, key=_start_sort_key)
                    winner_id = str(sorted_full[0].get("id", "")) if sorted_full else ""
                    if not winner_id:
                        continue
                    execute_duplicate_merge(self.calendar_id, winner_id, full, mode)  # type: ignore[arg-type]
                    merged_count += 1
                except Exception as exc:
                    err.append(f"{g.group_id[:16]}: {exc}")
                self.after(0, lambda i=idx, t=total, gid=g.group_id: self._set_progress(i, t, f"処理中 {i}/{t}: {gid[:18]}"))

            def done():
                self._close_progress()
                for b in self._action_btns:
                    b.configure(state="normal")
                if last_mode in ("longer", "concat"):
                    self._persist_last_mode(last_mode)
                if blocked:
                    err.append(f"場所不一致で除外: {len(blocked)} グループ")
                if err:
                    messagebox.showerror("一部失敗", "\n".join(err[:10]))
                messagebox.showinfo("完了", f"一括マージ完了: {merged_count} グループ")
                self.destroy()

            self.after(0, done)

        threading.Thread(target=run_batch, daemon=True).start()
