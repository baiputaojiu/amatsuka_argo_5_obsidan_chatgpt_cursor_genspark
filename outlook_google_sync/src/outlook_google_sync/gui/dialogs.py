"""Shared GUI dialog utilities."""

import tkinter as tk
from tkinter import ttk, messagebox


def ask_yes_no(title: str, message: str) -> bool:
    return messagebox.askyesno(title, message)


class DuplicateRepairOptionsDialog(tk.Toplevel):
    """Choose duplicate detection mode."""

    def __init__(
        self,
        master,
        *,
        initial_mode: str = "sync_key",
    ):
        super().__init__(master)
        self.title("重複修復の設定")
        self.geometry("520x200")
        self.transient(master)
        self.grab_set()
        self.result: str | None = None

        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="検出する重複の種類").pack(anchor="w")
        self.var_mode = tk.StringVar(value=initial_mode)
        ttk.Radiobutton(
            f,
            text="同一 sync_key のみ（ツール管理で同じキーが複数行）",
            variable=self.var_mode,
            value="sync_key",
        ).pack(anchor="w", pady=(6, 0))
        ttk.Radiobutton(
            f,
            text="同名・同開始・同終了（ツール未管理の行も含む）",
            variable=self.var_mode,
            value="content",
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            f,
            text="説明文のマージ方法は、マージ前プレビュー画面でグループごとに選べます。",
            foreground="gray",
            wraplength=480,
        ).pack(anchor="w", pady=(14, 0))

        bf = ttk.Frame(f)
        bf.pack(fill="x", pady=(16, 0))
        ttk.Button(bf, text="続行", command=self._ok).pack(side="right", padx=(6, 0))
        ttk.Button(bf, text="キャンセル", command=self._cancel).pack(side="right")

        self.wait_window()

    def _ok(self) -> None:
        self.result = self.var_mode.get()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class DescriptionMergeDialog(tk.Toplevel):
    """Choose description merge mode for batch duplicate merge."""

    def __init__(self, master, *, initial: str = "longer"):
        super().__init__(master)
        self.title("説明文のマージ")
        self.geometry("420x180")
        self.transient(master)
        self.grab_set()
        self.result: str | None = None

        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="一括マージ時の説明文の扱いを選んでください。").pack(anchor="w")
        self.var_desc = tk.StringVar(value=initial)
        ttk.Radiobutton(
            f,
            text="より長い説明文を採用",
            variable=self.var_desc,
            value="longer",
        ).pack(anchor="w", pady=(10, 0))
        ttk.Radiobutton(
            f,
            text="区切り（---）付きで連結",
            variable=self.var_desc,
            value="concat",
        ).pack(anchor="w", pady=(4, 0))

        bf = ttk.Frame(f)
        bf.pack(fill="x", pady=(16, 0))
        ttk.Button(bf, text="OK", command=self._ok).pack(side="right", padx=(6, 0))
        ttk.Button(bf, text="キャンセル", command=self._cancel).pack(side="right")

        self.wait_window()

    def _ok(self) -> None:
        self.result = self.var_desc.get()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class DeleteConfirmDialog(tk.Toplevel):
    """Ch26.3: Modal dialog for delete candidate confirmation."""

    def __init__(self, master, candidates: list[dict]):
        super().__init__(master)
        self.title("削除候補の確認")
        self.geometry("800x500")
        self.transient(master)
        self.grab_set()

        self.candidates = candidates
        self.approved: list[dict] = []

        if not candidates:
            ttk.Label(self, text="削除候補はありません。").pack(pady=20)
            ttk.Button(self, text="閉じる", command=self.destroy).pack()
            return

        ttk.Label(self, text=f"削除候補: {len(candidates)} 件").pack(anchor="w", padx=8, pady=4)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(
            frame,
            columns=("check", "start", "summary", "sync_key"),
            show="headings",
            selectmode="extended",
        )
        self.tree.heading("check", text="許可")
        self.tree.heading("start", text="開始日時")
        self.tree.heading("summary", text="件名")
        self.tree.heading("sync_key", text="sync_key")
        self.tree.column("check", width=50, anchor="center")
        self.tree.column("start", width=160)
        self.tree.column("summary", width=300)
        self.tree.column("sync_key", width=200)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.checks: dict[str, tk.BooleanVar] = {}
        for item in candidates:
            iid = item.get("id", "")
            private = ((item.get("extendedProperties") or {}).get("private") or {})
            start_val = (item.get("start") or {}).get("dateTime", (item.get("start") or {}).get("date", ""))
            summary = item.get("summary", "(no subject)")
            sync_key = private.get("sync_key", "")[:16]
            self.checks[iid] = tk.BooleanVar(value=True)
            self.tree.insert("", "end", iid=iid, values=("✓", start_val, summary, sync_key))

        self.tree.bind("<Button-1>", self._toggle_check)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        ttk.Button(btn_frame, text="すべて許可", command=self._select_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="すべて拒否", command=self._deselect_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="実行", command=self._confirm).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="キャンセル", command=self._cancel).pack(side="right", padx=4)

        self.wait_window()

    def _toggle_check(self, event):
        region = self.tree.identify("column", event.x, event.y)
        if region != "#1":
            return
        iid = self.tree.identify_row(event.y)
        if iid and iid in self.checks:
            var = self.checks[iid]
            var.set(not var.get())
            vals = list(self.tree.item(iid, "values"))
            vals[0] = "✓" if var.get() else ""
            self.tree.item(iid, values=vals)

    def _select_all(self):
        for iid, var in self.checks.items():
            var.set(True)
            vals = list(self.tree.item(iid, "values"))
            vals[0] = "✓"
            self.tree.item(iid, values=vals)

    def _deselect_all(self):
        for iid, var in self.checks.items():
            var.set(False)
            vals = list(self.tree.item(iid, "values"))
            vals[0] = ""
            self.tree.item(iid, values=vals)

    def _confirm(self):
        self.approved = [
            item for item in self.candidates
            if self.checks.get(item.get("id", ""), tk.BooleanVar(value=False)).get()
        ]
        self.destroy()

    def _cancel(self):
        self.approved = []
        self.destroy()


class SummaryDialog(tk.Toplevel):
    """Ch28.3: Post-sync summary dialog."""

    def __init__(self, master, result, log_path: str = ""):
        super().__init__(master)
        self.title("同期結果サマリー")
        self.geometry("600x400")
        self.transient(master)

        info = ttk.Frame(self)
        info.pack(fill="x", padx=12, pady=8)

        ttk.Label(info, text=f"作成: {result.created}").pack(anchor="w")
        ttk.Label(info, text=f"更新: {result.updated}").pack(anchor="w")
        ttk.Label(info, text=f"マージ: {result.merged}").pack(anchor="w")
        ttk.Label(info, text=f"スキップ: {result.skipped}").pack(anchor="w")
        ttk.Label(info, text=f"削除: {result.deleted}").pack(anchor="w")
        ttk.Label(info, text=f"失敗: {result.failed}").pack(anchor="w")

        if result.errors:
            ttk.Label(self, text="失敗一覧:").pack(anchor="w", padx=12)
            err_text = tk.Text(self, height=10)
            err_text.pack(fill="both", expand=True, padx=12, pady=4)
            display_errors = result.errors[:20]
            for err in display_errors:
                err_text.insert("end", err + "\n")
            if len(result.errors) > 20:
                err_text.insert("end", f"\n... 他 {len(result.errors) - 20} 件")
            err_text.configure(state="disabled")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=12, pady=8)
        if log_path:
            ttk.Button(btn_frame, text="詳細ログを開く", command=lambda: _open_log(log_path)).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="監査履歴", command=lambda: _show_audit(self)).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="閉じる", command=self.destroy).pack(side="right")


class RetryDialog(tk.Toplevel):
    """Ch28.1: Retry selection dialog for API errors."""

    def __init__(self, master, error_msg: str):
        super().__init__(master)
        self.title("API エラー")
        self.geometry("500x200")
        self.transient(master)
        self.grab_set()
        self.result = "stop"

        ttk.Label(self, text=error_msg, wraplength=460).pack(padx=12, pady=12)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=12, pady=8)
        ttk.Button(btn_frame, text="続行", command=lambda: self._set("continue")).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="停止", command=lambda: self._set("stop")).pack(side="right", padx=4)

        self.wait_window()

    def _set(self, val: str):
        self.result = val
        self.destroy()


def _open_log(path: str):
    import os
    os.startfile(path)


def _show_audit(parent):
    from ..services.audit_store import format_audit_summary
    win = tk.Toplevel(parent)
    win.title("監査履歴（直近10件）")
    win.geometry("620x350")
    win.transient(parent)
    t = tk.Text(win, wrap="word")
    t.pack(fill="both", expand=True, padx=8, pady=8)
    t.insert("1.0", format_audit_summary())
    t.configure(state="disabled")
    ttk.Button(win, text="閉じる", command=win.destroy).pack(pady=4)
