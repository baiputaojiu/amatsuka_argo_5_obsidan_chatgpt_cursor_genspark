"""Ch27: Preview before duplicate merge."""

from __future__ import annotations

import tkinter as tk
from typing import Literal

from tkinter import ttk, messagebox

from ..config.settings_store import save_settings
from ..connectors.google_calendar import get_event
from ..sync.duplicate_merge import build_merged_body, execute_duplicate_merge, preview_merged_location
from ..sync.duplicate_repair import DuplicateGroup


def _fmt_se_time(ev: dict) -> tuple[str, str]:
    st = ev.get("start") or {}
    en = ev.get("end") or {}
    s = st.get("dateTime") or st.get("date") or ""
    e = en.get("dateTime") or en.get("date") or ""
    return (str(s), str(e))


def _fmt_attendees(ev: dict) -> str:
    at = ev.get("attendees")
    if not at:
        return "（なし）"
    lines = []
    for a in at:
        em = a.get("email", "")
        resp = a.get("responseStatus", "")
        nm = a.get("displayName", "")
        extra = f"{nm} " if nm else ""
        lines.append(f"{extra}{em} ({resp})".strip())
    return "\n".join(lines) if lines else "（なし）"


class MergePreviewWindow(tk.Toplevel):
    def __init__(
        self,
        master,
        calendar_id: str,
        group: DuplicateGroup,
        winner_id: str,
        *,
        initial_description_mode: str = "longer",
    ):
        super().__init__(master)
        self.title("マージ内容のプレビュー")
        self.geometry("780x680")
        self.transient(master)
        self.grab_set()
        self._calendar_id = calendar_id
        self._group = group
        self._winner_id = winner_id
        self._full: list[dict] = []

        try:
            for it in group.items:
                eid = it.get("id")
                if not eid:
                    continue
                self._full.append(get_event(calendar_id, str(eid)))
        except Exception as exc:
            messagebox.showerror("取得エラー", str(exc), parent=master)
            self.destroy()
            return

        if not self._full:
            messagebox.showwarning("プレビュー", "イベントがありません。", parent=master)
            self.destroy()
            return

        by_id = {str(x.get("id")): x for x in self._full}
        self._winner = by_id.get(winner_id)
        if not self._winner:
            messagebox.showwarning("プレビュー", "代表イベントを取得できません。", parent=master)
            self.destroy()
            return

        self.var_desc = tk.StringVar(value=initial_description_mode)

        canvas = tk.Canvas(self)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(
            inner,
            text="同名・同時刻でも別会議の可能性があります。内容を確認してください。",
            foreground="gray",
            wraplength=720,
        ).pack(anchor="w", padx=8, pady=(8, 4))

        for i, ev in enumerate(self._full):
            eid = str(ev.get("id", ""))
            lf = ttk.LabelFrame(inner, text=f"イベント {i + 1}  ({eid[:16]}…)")
            lf.pack(fill="x", padx=8, pady=6)
            s1, s2 = _fmt_se_time(ev)
            ttk.Label(lf, text=f"件名: {ev.get('summary', '')}").pack(anchor="w", padx=6, pady=2)
            ttk.Label(lf, text=f"開始: {s1}").pack(anchor="w", padx=6)
            ttk.Label(lf, text=f"終了: {s2}").pack(anchor="w", padx=6)
            loc = (ev.get("location") or "").strip() or "（なし）"
            ttk.Label(lf, text=f"場所: {loc}").pack(anchor="w", padx=6, pady=(4, 0))
            ttk.Label(lf, text="説明:").pack(anchor="w", padx=6, pady=(4, 0))
            td = tk.Text(lf, height=4, wrap="word", width=86)
            td.pack(fill="x", padx=6, pady=2)
            td.insert("1.0", (ev.get("description") or "").strip())
            td.configure(state="disabled")
            ttk.Label(lf, text="参加者:").pack(anchor="w", padx=6, pady=(2, 0))
            ta = tk.Text(lf, height=5, wrap="word", width=86)
            ta.pack(fill="x", padx=6, pady=2)
            ta.insert("1.0", _fmt_attendees(ev))
            ta.configure(state="disabled")
            if eid == winner_id:
                ttk.Label(lf, text="← この予定を残してマージします", foreground="navy").pack(anchor="w", padx=6)

        mode_fr = ttk.LabelFrame(inner, text="このグループの説明文のマージ")
        mode_fr.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Radiobutton(
            mode_fr,
            text="より長い説明文を採用",
            variable=self.var_desc,
            value="longer",
            command=self._refresh_merged,
        ).pack(anchor="w", padx=8, pady=4)
        ttk.Radiobutton(
            mode_fr,
            text="区切り（---）付きで連結",
            variable=self.var_desc,
            value="concat",
            command=self._refresh_merged,
        ).pack(anchor="w", padx=8, pady=(0, 6))

        self._merged = ttk.LabelFrame(inner, text="マージ結果（適用後の代表イベント）")
        self._merged.pack(fill="x", padx=8, pady=10)

        self._refresh_merged()

        bf = ttk.Frame(inner)
        bf.pack(fill="x", padx=8, pady=12)
        ttk.Button(bf, text="マージを実行", command=self._run).pack(side="right", padx=4)
        ttk.Button(bf, text="戻る", command=self.destroy).pack(side="right", padx=4)

    def _refresh_merged(self) -> None:
        dm: Literal["longer", "concat"] = (
            "concat" if self.var_desc.get() == "concat" else "longer"
        )
        body = build_merged_body(
            self._winner,
            description_mode=dm,
            group_items=self._full,
        )
        for w in self._merged.winfo_children():
            w.destroy()

        self._lbl_summary = ttk.Label(self._merged, text=f"件名: {body.get('summary', '')}")
        self._lbl_summary.pack(anchor="w", padx=6, pady=2)
        st = body.get("start") or {}
        en = body.get("end") or {}
        self._lbl_start = ttk.Label(
            self._merged,
            text=f"開始: {st.get('dateTime') or st.get('date', '')}",
        )
        self._lbl_start.pack(anchor="w", padx=6)
        self._lbl_end = ttk.Label(
            self._merged,
            text=f"終了: {en.get('dateTime') or en.get('date', '')}",
        )
        self._lbl_end.pack(anchor="w", padx=6)
        self._lbl_loc = ttk.Label(
            self._merged,
            text=f"場所: {preview_merged_location(self._winner, self._full) or '（なし）'}",
        )
        self._lbl_loc.pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(self._merged, text="説明:").pack(anchor="w", padx=6, pady=(4, 0))
        self._tm = tk.Text(self._merged, height=5, wrap="word", width=86)
        self._tm.pack(fill="x", padx=6, pady=2)
        self._tm.insert("1.0", (body.get("description") or "").strip())
        self._tm.configure(state="disabled")

        ttk.Label(self._merged, text="参加者（代表の予定のまま）:").pack(anchor="w", padx=6, pady=(4, 0))
        ta2 = tk.Text(self._merged, height=4, wrap="word", width=86)
        ta2.pack(fill="x", padx=6, pady=2)
        ta2.insert("1.0", _fmt_attendees(self._winner))
        ta2.configure(state="disabled")

    def _persist_description_mode(self, mode: str) -> None:
        top = self.winfo_toplevel()
        if not hasattr(top, "state_data") or not hasattr(top, "settings"):
            return
        top.state_data["duplicate_repair_description_mode"] = mode
        ics_path = top.ics_var.get().strip() if hasattr(top, "ics_var") else ""
        top.settings["runtime_state"] = {**top.state_data, "ics_path": ics_path}
        save_settings(top.settings)

    def _run(self) -> None:
        if not messagebox.askyesno(
            "確認",
            "代表以外の予定を削除し、代表に内容をマージします。よろしいですか？",
            parent=self,
        ):
            return
        dm: Literal["longer", "concat"] = (
            "concat" if self.var_desc.get() == "concat" else "longer"
        )
        self._persist_description_mode(dm)
        try:
            execute_duplicate_merge(
                self._calendar_id,
                self._winner_id,
                self._full,
                dm,
            )
            messagebox.showinfo("完了", "マージしました。", parent=self)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("エラー", str(exc), parent=self)
