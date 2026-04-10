"""Ch9: Main window — operation hub with worker threads and cancel support."""

from __future__ import annotations
import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox

from tkcalendar import DateEntry

from ..config.settings_store import load_settings, save_settings
from ..connectors.outlook_com import read_events_from_outlook
from ..connectors.outlook_ics import read_events_from_ics
from ..connectors.google_calendar import (
    get_calendar_time_zone,
    list_all_events_in_range,
    list_calendars,
    list_managed_event_items,
    list_managed_events,
)
from ..logging.logger_factory import build_logger
from ..models.profile import FilterConfig
from ..services.connection_test import test_ics_file, test_google, test_outlook_com
from ..services.audit_store import save_audit
from ..services.notifications import notify
from ..sync.engine import SyncEngine
from ..sync.filters import apply_filters
from ..sync.preview import build_preview
from .settings_window import SettingsWindow
from .preview_window import PreviewWindow
from .duplicate_repair_window import DuplicateRepairWindow
from .dialogs import DeleteConfirmDialog, DuplicateRepairOptionsDialog, SummaryDialog, RetryDialog
from .ttk_style import apply_button_contrast_style


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        apply_button_contrast_style(self)
        self.title("Outlook → Google カレンダー同期")
        self.geometry("1020x720")
        self.logger = build_logger()
        self.settings = load_settings()
        self.state_data = {
            "input_method": "com",
            "detail_level": "full",
            "include_private": True,
            "ics_path": "",
            "calendar_id": "primary",
            "conflict_policy": "overwrite",
            "notification_enabled": True,
            "log_verbosity": "standard",
            "duplicate_repair_mode": "sync_key",
            "duplicate_repair_description_mode": "longer",
        }
        self.state_data.update(self.settings.get("runtime_state", {}))
        self.state_data.setdefault("duplicate_repair_mode", "sync_key")
        self.state_data.setdefault("duplicate_repair_description_mode", "longer")
        self._cancel_flag = threading.Event()
        self._last_preview = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._force_quit)

    def _force_quit(self) -> None:
        """タイトルバー × 等でプロセスを即終了（メインスレッド停止中は効かない場合あり）。"""
        os._exit(0)

    def _build(self):
        # Top: date range + ICS path
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="開始日").grid(row=0, column=0, sticky="w")
        self.start = DateEntry(top)
        self.start.grid(row=0, column=1)
        ttk.Label(top, text="終了日").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.end = DateEntry(top)
        self.end.grid(row=0, column=3)

        ttk.Label(top, text="ICS").grid(row=1, column=0, sticky="w")
        self.ics_var = tk.StringVar(value=self.state_data.get("ics_path", ""))
        self.ics_entry = ttk.Entry(top, textvariable=self.ics_var, width=60)
        self.ics_entry.grid(row=1, column=1, columnspan=3, sticky="ew")
        self.ics_pick_btn = ttk.Button(top, text="参照", command=self.pick_ics)
        self.ics_pick_btn.grid(row=1, column=4)

        # Profile display
        ttk.Label(top, text=f"プロファイル: default").grid(row=0, column=5, padx=(16, 0))

        # Buttons
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8)
        self.btns = []
        for text, cmd in [
            ("設定", self.open_settings),
            ("接続テスト", lambda: self.worker(self.connection_test)),
            ("通常プレビュー", lambda: self.worker(self.preview_normal)),
            ("通常同期", lambda: self.worker(lambda: self.sync("normal"))),
            ("フルプレビュー", lambda: self.worker(self.preview_full)),
            ("フル同期", lambda: self.worker(lambda: self.sync("full"))),
            ("重複修復", lambda: self.worker(self.duplicate_repair)),
        ]:
            b = ttk.Button(bar, text=text, command=cmd)
            b.pack(side="left", padx=3)
            self.btns.append(b)

        # Cancel button (GUI-04)
        self.cancel_btn = ttk.Button(bar, text="キャンセル", command=self._on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=3)

        # Status + progress
        self.status = tk.StringVar(value="待機中")
        ttk.Label(self, textvariable=self.status).pack(fill="x", padx=8)

        # Log area
        self.log = tk.Text(self, height=22)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Helpers ──

    def log_msg(self, m):
        self.log.insert("end", m + "\n")
        self.log.see("end")

    def pick_ics(self):
        p = filedialog.askopenfilename(filetypes=[("ICS", "*.ics"), ("All", "*.*")])
        if p:
            self.ics_var.set(p)

    def open_settings(self):
        SettingsWindow(
            self, self.state_data,
            on_google_auth=self._google_auth,
            on_list_calendars=list_calendars,
        )

    def _google_auth(self):
        from ..connectors.google_calendar import get_service
        get_service()

    def _on_cancel(self):
        self._cancel_flag.set()
        self.status.set("キャンセル要求中...")

    def _set_main_busy(self, busy: bool) -> None:
        """処理中は操作ボタン・日付・ICS参照を無効化。キャンセルのみ有効にできる。"""
        st = "disabled" if busy else "normal"
        for b in self.btns:
            b.configure(state=st)
        self.cancel_btn.configure(state="normal" if busy else "disabled")
        if hasattr(self, "ics_pick_btn"):
            self.ics_pick_btn.configure(state=st)
        if hasattr(self, "ics_entry"):
            try:
                self.ics_entry.configure(state=st)
            except tk.TclError:
                pass
        for w in (getattr(self, "start", None), getattr(self, "end", None)):
            if w is not None:
                try:
                    w.configure(state=st)
                except tk.TclError:
                    pass

    # ── Worker thread (GUI-01~06) ──

    def worker(self, fn):
        self._cancel_flag.clear()
        self._set_main_busy(True)
        self.status.set("処理中...")

        def run():
            # Outlook COM はバックグラウンドスレッドで STA を明示初期化しないと、
            # 他処理後に CO_E_NOTINITIALIZED (-2147221008) になることがある。
            com_inited = False
            try:
                import pythoncom  # type: ignore[import-untyped]

                pythoncom.CoInitialize()
                com_inited = True
            except Exception:
                pass
            try:
                fn()
            except Exception as e:
                self.logger.error("Worker error: %s", e, exc_info=True)
                err_text = str(e)
                self.after(0, lambda msg=err_text: self.log_msg(f"ERROR: {msg}"))
            finally:
                if com_inited:
                    try:
                        import pythoncom  # type: ignore[import-untyped]

                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
                self.after(0, self._worker_done)

        threading.Thread(target=run, daemon=True).start()

    def _worker_done(self):
        self._set_main_busy(False)
        self.status.set("待機中")

    # ── Read source with filters ──

    def read_source(self):
        s = self.start.get_date()
        e = self.end.get_date()
        if s > e:
            raise ValueError("開始日が終了日より後です")

        sd = datetime.combine(s, datetime.min.time())
        ed = datetime.combine(e, datetime.max.time())
        m = self.state_data.get("input_method", "com")

        # Ch9.4: COM ↔ ICS switch warning
        last_method = self.settings.get("sync_metadata", {}).get("last_input_method")
        if last_method and last_method != m:
            is_com_switch = (last_method == "com") != (m == "com")
            if is_com_switch:
                answer = [None]
                evt = threading.Event()

                def ask():
                    answer[0] = messagebox.askyesno(
                        "入力方法変更",
                        f"入力方法が {last_method} → {m} に変更されています。\n"
                        "重複が発生する可能性があります。続行しますか？",
                    )
                    evt.set()

                self.after(0, ask)
                evt.wait()
                if not answer[0]:
                    raise ValueError("ユーザーがキャンセルしました")

        self.after(0, lambda: self.status.set("ソース読取中..."))

        if m == "com":
            events = read_events_from_outlook(sd, ed, self.state_data.get("include_private", True))
        else:
            events = read_events_from_ics(
                self.ics_var.get().strip(), m,
                date_start=sd, date_end=ed,
            )

        raw_n = len(events)
        # Apply filters (Ch8)
        fc = self._build_filter_config()
        events = apply_filters(events, fc, self.state_data.get("include_private", True))

        self.after(
            0,
            lambda: self.log_msg(
                f"読取: {raw_n} 件（ソース直後）→ {len(events)} 件（フィルタ適用後）"
            ),
        )
        return events, sd, ed

    def _build_filter_config(self) -> FilterConfig:
        st = self.state_data
        return FilterConfig(
            exclude_all_day=st.get("filter_exclude_all_day", False),
            exclude_free=st.get("filter_exclude_free", False),
            exclude_tentative=st.get("filter_exclude_tentative", False),
            exclude_private=st.get("filter_exclude_private", False),
            category_mode=st.get("filter_category_mode", "none"),
            category_list=[c.strip() for c in st.get("filter_category_list", "").split(",") if c.strip()],
            subject_keywords=[kw.strip() for kw in st.get("filter_subject_keywords", "").split("\n") if kw.strip()],
            location_keywords=[kw.strip() for kw in st.get("filter_location_keywords", "").split("\n") if kw.strip()],
        )

    # ── Preview (Ch15) ──

    def preview_normal(self):
        events, sd, ed = self.read_source()
        cal_id = self.state_data.get("calendar_id", "primary")
        existing = list_managed_events(cal_id, sd, ed) if events else {}
        fp = self.settings.get("sync_metadata", {}).get("per_source_fingerprint", {})
        tz = get_calendar_time_zone(cal_id)
        snap = build_preview(
            events, existing, fp, mode="normal", range_start=sd, range_end=ed, time_zone=tz,
        )
        self._last_preview = snap
        self.after(0, lambda: PreviewWindow(
            self, "通常プレビュー", snap,
            on_execute=lambda s, idx: self._execute_from_preview(s, idx, "normal"),
        ))

    def preview_full(self):
        events, sd, ed = self.read_source()
        cal_id = self.state_data.get("calendar_id", "primary")
        existing = list_managed_events(cal_id, sd, ed) if events else {}
        fp = self.settings.get("sync_metadata", {}).get("per_source_fingerprint", {})
        tz = get_calendar_time_zone(cal_id)
        snap = build_preview(
            events, existing, fp, mode="full", range_start=sd, range_end=ed, time_zone=tz,
        )
        self._last_preview = snap
        self.after(0, lambda: PreviewWindow(
            self, "フルプレビュー", snap,
            on_execute=lambda s, idx: self._execute_from_preview(s, idx, "full"),
        ))

    def _execute_from_preview(self, snapshot, approved_indices, mode):
        """PRE-SNAP-01: Execute sync using the approved snapshot."""
        approved_events = [
            snapshot.actions[i].event for i in approved_indices
            if snapshot.actions[i].event and snapshot.actions[i].action in ("create", "update")
        ]
        if approved_events:
            sd = datetime.combine(self.start.get_date(), datetime.min.time())
            ed = datetime.combine(self.end.get_date(), datetime.max.time())
            self.worker(lambda: self._sync_events(approved_events, mode, range_start=sd, range_end=ed))

    # ── Sync (Ch14) ──

    def sync(self, mode):
        events, sd, ed = self.read_source()
        self._sync_events(events, mode, range_start=sd, range_end=ed)

    def _sync_events(self, events, mode, range_start=None, range_end=None):
        eng = SyncEngine(self.logger)
        cal_id = self.state_data.get("calendar_id", "primary")
        fp = self.settings.get("sync_metadata", {}).get("per_source_fingerprint", {})

        def on_error(msg):
            result = [None]
            evt = threading.Event()

            def ask():
                dlg = RetryDialog(self, msg)
                result[0] = dlg.result
                evt.set()

            self.after(0, ask)
            evt.wait()
            return result[0] or "stop"

        res = eng.run(
            cal_id, events, mode, fp,
            detail_level=self.state_data.get("detail_level", "full"),
            conflict_policy=self.state_data.get("conflict_policy", "overwrite"),
            range_start=range_start,
            range_end=range_end,
            progress=lambda m: self.after(0, lambda: self.status.set(m)),
            cancel_check=lambda: self._cancel_flag.is_set(),
            on_error=on_error,
        )

        # Ch26.3: Delete confirmation UI
        if res.delete_candidates_list:
            approved = [None]
            evt = threading.Event()

            def show_del():
                dlg = DeleteConfirmDialog(self, res.delete_candidates_list)
                approved[0] = dlg.approved
                evt.set()

            self.after(0, show_del)
            evt.wait()

            if approved[0]:
                deleted, del_errors = eng.execute_deletions(
                    cal_id, approved[0],
                    progress=lambda m: self.after(0, lambda: self.status.set(m)),
                )
                res.deleted = deleted
                res.errors.extend(del_errors)

        # Ch13.2: Save sync_metadata
        from datetime import UTC
        from ..utils.hash_utils import sha256_text
        sm = self.settings.setdefault("sync_metadata", {})
        sm["per_source_fingerprint"] = {e.sync_key: e.fingerprint for e in events}
        sm["last_input_method"] = self.state_data.get("input_method", "com")
        sm["last_success_utc"] = datetime.now(UTC).isoformat() if not res.failed else sm.get("last_success_utc")
        sm["last_run_mode"] = mode
        sm["last_D_start"] = self.start.get_date().isoformat() if hasattr(self, "start") else None
        sm["last_D_end"] = self.end.get_date().isoformat() if hasattr(self, "end") else None
        conditions = f"{self.state_data.get('detail_level')}|{self.state_data.get('calendar_id')}|{self.state_data.get('conflict_policy')}"
        sm["last_conditions_hash"] = sha256_text(conditions)[:16]
        self.settings["runtime_state"] = {**self.state_data, "ics_path": self.ics_var.get().strip()}
        save_settings(self.settings)
        save_audit({
            "mode": mode,
            "created": res.created,
            "updated": res.updated,
            "merged": res.merged,
            "deleted": res.deleted,
            "delete_candidates": res.deleted_candidates,
            "skipped": res.skipped,
            "failed": res.failed,
        })

        self.after(0, lambda: self.log_msg(
            f"同期完了 c={res.created} u={res.updated} m={res.merged} "
            f"d={res.deleted} s={res.skipped} f={res.failed}"
        ))

        # Ch28.3: Summary dialog
        self.after(0, lambda: SummaryDialog(self, res))

        # Ch29: Notification
        if self.state_data.get("notification_enabled", True):
            msg = "同期完了" if not res.failed else f"部分失敗 {res.failed}件"
            self.after(0, lambda: notify("Sync", msg))

    # ── Connection test (Ch24) ──

    def connection_test(self):
        m = self.state_data.get("input_method", "com")
        results: list[str] = []

        if m == "com":
            sd = datetime.combine(self.start.get_date(), datetime.min.time())
            ed = datetime.combine(self.end.get_date(), datetime.max.time())
            ok, msg = test_outlook_com(sd, ed)
            results.append(msg)
            self.after(0, lambda: self.log_msg(msg))
            if not ok:
                return
        else:
            ok, msg = test_ics_file(
                self.ics_var.get().strip(),
                range_start=datetime.combine(self.start.get_date(), datetime.min.time()),
                range_end=datetime.combine(self.end.get_date(), datetime.max.time()),
            )
            results.append(msg)
            self.after(0, lambda: self.log_msg(msg))
            if not ok:
                return

        ok, msg = test_google(self.state_data.get("calendar_id") or None)
        results.append(msg)
        self.after(0, lambda: self.log_msg(msg))

    # ── Duplicate repair (Ch27) ──

    def duplicate_repair(self):
        dlg = DuplicateRepairOptionsDialog(
            self,
            initial_mode=self.state_data.get("duplicate_repair_mode", "sync_key"),
        )
        if dlg.result is None:
            return
        mode = dlg.result
        self.state_data["duplicate_repair_mode"] = mode
        self.settings["runtime_state"] = {**self.state_data, "ics_path": self.ics_var.get().strip()}
        save_settings(self.settings)

        cal_id = self.state_data.get("calendar_id", "primary")
        sd = datetime.combine(self.start.get_date(), datetime.min.time())
        ed = datetime.combine(self.end.get_date(), datetime.max.time())
        init_desc = self.state_data.get("duplicate_repair_description_mode", "longer")

        def job():
            from ..sync.duplicate_repair import (
                build_groups_for_mode,
                filter_events_within_start_end_dates,
            )

            try:
                if mode == "sync_key":
                    events = list_managed_event_items(cal_id, sd, ed)
                else:
                    events = list_all_events_in_range(cal_id, sd, ed)
                events = filter_events_within_start_end_dates(events, sd, ed)
                groups = build_groups_for_mode(mode, events)
                self.after(
                    0,
                    lambda g=groups, m=mode, d=init_desc: DuplicateRepairWindow(
                        self,
                        g,
                        cal_id,
                        mode=m,
                        initial_description_mode=d,
                    ),
                )
            except Exception as e:
                self.logger.error("duplicate_repair: %s", e, exc_info=True)
                err_text = str(e)
                self.after(0, lambda msg=err_text: self.log_msg(f"ERROR: {msg}"))
                self.after(
                    0,
                    lambda msg=err_text: messagebox.showerror("重複修復", msg),
                )

        self.worker(job)
