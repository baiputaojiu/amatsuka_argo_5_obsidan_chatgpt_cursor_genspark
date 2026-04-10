"""Ch6: Settings window with tabs — all configuration centralized here."""

from __future__ import annotations
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


_DEFAULT_STATE = {
    "input_method": "com",
    "detail_level": "full",
    "include_private": True,
    "ics_path": "",
    "calendar_id": "primary",
    "conflict_policy": "overwrite",
    "merge_tolerance_minutes": 2,
    "merge_description_priority": "source",
    "merge_location_priority": "source",
    "notification_enabled": True,
    "log_verbosity": "standard",
    "filter_exclude_all_day": False,
    "filter_exclude_free": False,
    "filter_exclude_tentative": False,
    "filter_exclude_private": False,
    "filter_category_mode": "none",
    "filter_category_list": "",
    "filter_subject_keywords": "",
    "filter_location_keywords": "",
    "macro_output_path": "",
}


class SettingsWindow(tk.Toplevel):
    """SET-WIN-01~06: Single settings window with tabs."""

    def __init__(self, master, state: dict, on_google_auth=None, on_list_calendars=None):
        super().__init__(master)
        self.title("設定")
        self._restore_geometry(state)
        self.resizable(True, True)

        self.state = state
        self.on_google_auth = on_google_auth
        self.on_list_calendars = on_list_calendars
        self._snapshot = dict(state)
        self._vars: dict[str, tk.Variable] = {}

        self._build_tabs()
        self._build_buttons()

    def _restore_geometry(self, state: dict):
        geo = state.get("_settings_geometry", "780x580")
        self.geometry(geo)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _save_geometry(self):
        self.state["_settings_geometry"] = self.geometry()

    # ── Tabs ──

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._build_tab_basic()
        self._build_tab_sync()
        self._build_tab_filter()
        self._build_tab_google()
        self._build_tab_profile()
        self._build_tab_log_notify()
        self._build_tab_advanced()

    def _build_tab_basic(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="基本")
        ttk.Label(f, text="アクティブプロファイル:").grid(row=0, column=0, sticky="w")
        ttk.Label(f, text="default (初版は単一)").grid(row=0, column=1, sticky="w")

    def _build_tab_sync(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="同期")
        row = 0

        ttk.Label(f, text="入力方法").grid(row=row, column=0, sticky="w")
        v = self._sv("input_method")
        ttk.Combobox(f, textvariable=v, values=["com", "ics_manual", "ics_macro"], state="readonly", width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="マクロ出力パス").grid(row=row, column=0, sticky="w")
        mp = self._sv("macro_output_path")
        ttk.Entry(f, textvariable=mp, width=50).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(f, text="詳細度").grid(row=row, column=0, sticky="w")
        v2 = self._sv("detail_level")
        ttk.Combobox(f, textvariable=v2, values=["full", "title_only"], state="readonly", width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Checkbutton(f, text="非公開予定を含める", variable=self._bv("include_private")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(f, text="衝突ポリシー").grid(row=row, column=0, sticky="w")
        cp = self._sv("conflict_policy")
        ttk.Combobox(f, textvariable=cp, values=["overwrite", "detach_new", "merge"], state="readonly", width=20).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="マージ許容差(分)").grid(row=row, column=0, sticky="w")
        mt = self._iv("merge_tolerance_minutes")
        ttk.Spinbox(f, textvariable=mt, from_=0, to=60, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="マージ description 優先").grid(row=row, column=0, sticky="w")
        mdp = self._sv("merge_description_priority")
        ttk.Combobox(f, textvariable=mdp, values=["source", "google"], state="readonly", width=12).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="マージ location 優先").grid(row=row, column=0, sticky="w")
        mlp = self._sv("merge_location_priority")
        ttk.Combobox(f, textvariable=mlp, values=["source", "google"], state="readonly", width=12).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="通常同期 = 差分同期（変更のみ書込）\nフル同期 = 全件再評価", foreground="gray").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1

        ttk.Label(
            f,
            text="⚠ 注意: 「詳細」→「タイトルのみ」に切り替えても、\n"
                 "Google 上の既存 description/location は削除されません。\n"
                 "非公開予定を含めた場合、情報が Google に残ります。",
            foreground="red",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        f.columnconfigure(1, weight=1)

    def _build_tab_filter(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="フィルタ")
        row = 0

        ttk.Checkbutton(f, text="終日予定を除外", variable=self._bv("filter_exclude_all_day")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(f, text="Free（空き）を除外", variable=self._bv("filter_exclude_free")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(f, text="Tentative（仮）を除外", variable=self._bv("filter_exclude_tentative")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(f, text="Private を除外", variable=self._bv("filter_exclude_private")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(f, text="カテゴリフィルタ").grid(row=row, column=0, sticky="w")
        cm = self._sv("filter_category_mode")
        ttk.Combobox(f, textvariable=cm, values=["none", "exclude", "include_only"], state="readonly", width=16).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="カテゴリ名(カンマ区切り)").grid(row=row, column=0, sticky="w")
        cl = self._sv("filter_category_list")
        ttk.Entry(f, textvariable=cl, width=50).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Separator(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(f, text="件名キーワード除外(改行区切り)").grid(row=row, column=0, sticky="nw")
        self.subject_kw = tk.Text(f, height=3, width=40)
        self.subject_kw.grid(row=row, column=1, sticky="ew")
        self.subject_kw.insert("1.0", self.state.get("filter_subject_keywords", ""))
        row += 1

        ttk.Label(f, text="場所キーワード除外(改行区切り)").grid(row=row, column=0, sticky="nw")
        self.location_kw = tk.Text(f, height=3, width=40)
        self.location_kw.grid(row=row, column=1, sticky="ew")
        self.location_kw.insert("1.0", self.state.get("filter_location_keywords", ""))
        row += 1

        ttk.Label(f, text="※ 単純部分一致（大文字小文字無視）", foreground="gray").grid(row=row, column=0, columnspan=2, sticky="w")

        f.columnconfigure(1, weight=1)

    def _build_tab_google(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="Google")

        ttk.Label(f, text="対象カレンダーID:").grid(row=0, column=0, sticky="w")
        cal = self._sv("calendar_id")
        ttk.Entry(f, textvariable=cal, width=50).grid(row=0, column=1, sticky="ew")

        self._cal_fetch_btn = ttk.Button(f, text="カレンダー一覧を取得", command=self._fetch_calendars)
        self._cal_fetch_btn.grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        self.cal_listbox = tk.Listbox(f, height=8)
        self.cal_listbox.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=4)
        self.cal_listbox.bind("<<ListboxSelect>>", self._on_cal_select)

        self._cal_items: list[dict] = []

        self._auth_btn = ttk.Button(f, text="Google 認証", command=self._google_auth)
        self._auth_btn.grid(row=3, column=0, sticky="w")

        f.columnconfigure(1, weight=1)
        f.rowconfigure(2, weight=1)

    def _build_tab_profile(self):
        """Ch6.2 / Ch32: Profile management with export/import."""
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="プロファイル")

        ttk.Label(f, text="プロファイル一覧 (初版は単一運用)").grid(row=0, column=0, columnspan=2, sticky="w")
        self.profile_list = tk.Listbox(f, height=4)
        self.profile_list.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=4)
        self.profile_list.insert("end", "default")

        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._profile_export_btn = ttk.Button(btn_frame, text="エクスポート", command=self._export_settings)
        self._profile_export_btn.pack(side="left", padx=4)
        self._profile_import_btn = ttk.Button(btn_frame, text="インポート", command=self._import_settings)
        self._profile_import_btn.pack(side="left", padx=4)
        self._profile_tab_btns = [self._profile_export_btn, self._profile_import_btn]

        ttk.Label(f, text="エクスポート: 認証情報を除外した設定JSONを出力\nインポート: 外部の設定JSONを読み込み", foreground="gray").grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

    def _export_settings(self):
        from ..config.export_import import export_settings
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            title="設定をエクスポート",
        )
        if path:
            try:
                export_settings(path)
                messagebox.showinfo("エクスポート", f"エクスポート完了: {path}")
            except Exception as exc:
                messagebox.showerror("エラー", str(exc))

    def _import_settings(self):
        from ..config.export_import import import_settings
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            title="設定をインポート",
        )
        if path:
            try:
                ok, msg = import_settings(path)
                if ok:
                    messagebox.showinfo("インポート", msg)
                else:
                    messagebox.showwarning("インポート", msg)
            except Exception as exc:
                messagebox.showerror("エラー", str(exc))

    def _build_tab_log_notify(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="ログ/通知")

        ttk.Label(f, text="ログ詳細度").grid(row=0, column=0, sticky="w")
        lv = self._sv("log_verbosity")
        ttk.Combobox(f, textvariable=lv, values=["standard", "detailed"], state="readonly", width=16).grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(f, text="通知を有効化", variable=self._bv("notification_enabled")).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(f, text="ログ保存先: ~/.outlook_google_sync/logs/", foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _build_tab_advanced(self):
        f = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(f, text="高度な設定")
        ttk.Label(f, text="※ 初版では追加設定なし。\n将来の差分保持期間・デバッグフラグ等を配置する予定。", foreground="gray").pack(anchor="w")

    # ── Buttons ──

    def _build_buttons(self):
        bf = ttk.Frame(self)
        bf.pack(fill="x", padx=8, pady=8)
        self._btn_reset = ttk.Button(bf, text="既定値に戻す", command=self._reset_defaults)
        self._btn_reset.pack(side="left", padx=4)
        self._btn_cancel = ttk.Button(bf, text="キャンセル", command=self._on_cancel)
        self._btn_cancel.pack(side="right", padx=4)
        self._btn_save = ttk.Button(bf, text="保存", command=self._on_save)
        self._btn_save.pack(side="right", padx=4)
        self._footer_btns = [self._btn_reset, self._btn_cancel, self._btn_save]

    def _set_network_busy(self, busy: bool) -> None:
        """Google 認証・API 取得中は関連ボタンとフッターを無効化（誤保存・二重実行防止）。"""
        st = "disabled" if busy else "normal"
        for btn in (getattr(self, "_auth_btn", None), getattr(self, "_cal_fetch_btn", None)):
            if btn is not None:
                try:
                    btn.configure(state=st)
                except tk.TclError:
                    pass
        for btn in getattr(self, "_footer_btns", []):
            try:
                btn.configure(state=st)
            except tk.TclError:
                pass
        for btn in getattr(self, "_profile_tab_btns", []):
            try:
                btn.configure(state=st)
            except tk.TclError:
                pass

    # ── Actions ──

    def _on_save(self):
        for key, var in self._vars.items():
            self.state[key] = var.get()
        self.state["filter_subject_keywords"] = self.subject_kw.get("1.0", "end-1c").strip()
        self.state["filter_location_keywords"] = self.location_kw.get("1.0", "end-1c").strip()
        self._save_geometry()
        self.destroy()

    def _on_cancel(self):
        self.state.update(self._snapshot)
        self.destroy()

    def _reset_defaults(self):
        if not messagebox.askyesno("確認", "すべての設定を既定値に戻しますか？"):
            return
        for key, val in _DEFAULT_STATE.items():
            if key in self._vars:
                self._vars[key].set(val)
        self.subject_kw.delete("1.0", "end")
        self.location_kw.delete("1.0", "end")

    def _fetch_calendars(self):
        if not self.on_list_calendars:
            return

        def run() -> None:
            err: Exception | None = None
            items: list[dict] | None = None
            try:
                items = self.on_list_calendars()
            except Exception as exc:
                err = exc

            def done() -> None:
                try:
                    if not self.winfo_exists():
                        return
                except tk.TclError:
                    return
                self._set_network_busy(False)
                if err is not None:
                    messagebox.showerror("エラー", str(err))
                    return
                assert items is not None
                self._cal_items = items
                self.cal_listbox.delete(0, "end")
                for item in items:
                    label = item.get("summary", item.get("id", ""))
                    if item.get("primary"):
                        label += " (主カレンダー)"
                    self.cal_listbox.insert("end", label)

            self.after(0, done)

        self._set_network_busy(True)
        threading.Thread(target=run, daemon=True).start()

    def _on_cal_select(self, event):
        sel = self.cal_listbox.curselection()
        if sel and self._cal_items:
            item = self._cal_items[sel[0]]
            self._vars["calendar_id"].set(item.get("id", "primary"))

    def _google_auth(self) -> None:
        if not self.on_google_auth:
            return

        def run() -> None:
            err: Exception | None = None
            try:
                self.on_google_auth()
            except Exception as exc:
                err = exc

            def done() -> None:
                try:
                    if not self.winfo_exists():
                        return
                except tk.TclError:
                    return
                self._set_network_busy(False)
                if err is not None:
                    messagebox.showerror("認証エラー", str(err))
                else:
                    messagebox.showinfo("認証", "Google 認証成功")

            self.after(0, done)

        self._set_network_busy(True)
        threading.Thread(target=run, daemon=True).start()

    # ── Variable helpers ──

    def _sv(self, key: str) -> tk.StringVar:
        v = tk.StringVar(value=str(self.state.get(key, _DEFAULT_STATE.get(key, ""))))
        self._vars[key] = v
        return v

    def _bv(self, key: str) -> tk.BooleanVar:
        v = tk.BooleanVar(value=bool(self.state.get(key, _DEFAULT_STATE.get(key, False))))
        self._vars[key] = v
        return v

    def _iv(self, key: str) -> tk.IntVar:
        v = tk.IntVar(value=int(self.state.get(key, _DEFAULT_STATE.get(key, 0))))
        self._vars[key] = v
        return v
