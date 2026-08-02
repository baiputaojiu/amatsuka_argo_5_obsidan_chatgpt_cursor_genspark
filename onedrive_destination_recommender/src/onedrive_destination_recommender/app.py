import sys
import tkinter as tk
from collections.abc import Iterable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from onedrive_destination_recommender.audit import (
    AUDIT_FILE_NAME,
    AuditError,
    DecisionType,
)
from onedrive_destination_recommender.catalog import (
    CATALOG_FILE_NAME,
    CatalogError,
    load_catalog,
    update_catalog,
)
from onedrive_destination_recommender.codex_prompt import (
    CodexConsultation,
    copy_codex_consultation,
)
from onedrive_destination_recommender.ranking import RankingError
from onedrive_destination_recommender.session import (
    InputKind,
    InputSelectionError,
    RecommenderSession,
    format_scanned_at,
    open_folder,
)
from onedrive_destination_recommender.settings import (
    SETTINGS_FILE_NAME,
    Settings,
    SettingsError,
    default_settings_path,
    load_settings,
)

APP_TITLE = "OneDrive保存先レコメンダー"
README_PATH = Path(__file__).resolve().parents[2] / "README.md"
SELECTED_CANDIDATE_PATH_GUIDANCE = "候補を選択すると絶対パスを全文表示します。"


class RecommenderApp:
    """One-screen Tkinter integration for the MVP 0 functions."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        settings_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.root = root
        self._settings_path_arg = Path(settings_path) if settings_path is not None else None
        self._catalog_path_arg = Path(catalog_path) if catalog_path is not None else None
        self._audit_path_arg = Path(audit_path) if audit_path is not None else None
        self.settings_path: Path | None = None
        self.catalog_path: Path | None = None
        self.audit_path: Path | None = None
        self.settings: Settings | None = None
        self.session: RecommenderSession | None = None
        self.consultation: CodexConsultation | None = None
        self._changing_search = False

        self._build_widgets()
        self._initialize_dnd()
        self._load_runtime()

    def _build_widgets(self) -> None:
        self.root.title(APP_TITLE)
        self.root.minsize(1050, 700)
        self.root.geometry("1180x820")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        content = ttk.Frame(self.root, padding=12)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)

        ttk.Label(content, text=APP_TITLE, font=("Yu Gothic UI", 16, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 8),
        )

        self._build_target_area(content)

        body = ttk.Frame(content)
        body.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        self._build_input_area(left)
        self._build_search_area(left)
        self._build_candidate_area(left)
        self._build_action_area(left)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self._build_codex_area(right)
        self._build_result_area(right)

        self.operation_status_var = tk.StringVar(value="起動準備中です。")
        ttk.Label(
            content,
            textvariable=self.operation_status_var,
            foreground="#1f4e79",
            wraplength=1120,
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_target_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="対象フォルダとカタログ", padding=8)
        area.grid(row=1, column=0, sticky="ew")
        area.columnconfigure(1, weight=1)

        self.current_root_var = tk.StringVar(value="未読込")
        self.previous_root_var = tk.StringVar(value="未読込")
        self.pending_root_var = tk.StringVar(value="未読込")
        self.catalog_status_var = tk.StringVar(value="未読込")

        labels = (
            ("今年度", self.current_root_var),
            ("昨年度", self.previous_root_var),
            ("保存先未定", self.pending_root_var),
            ("カタログ", self.catalog_status_var),
        )
        for row, (label, variable) in enumerate(labels):
            ttk.Label(area, text=f"{label}：").grid(row=row, column=0, sticky="nw")
            ttk.Label(area, textvariable=variable, wraplength=850).grid(
                row=row,
                column=1,
                sticky="w",
            )

        self.update_button = ttk.Button(
            area,
            text="フォルダ構成を更新",
            command=self._update_catalog,
        )
        self.update_button.grid(row=0, column=2, rowspan=4, sticky="e", padx=(12, 0))

    def _build_input_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="入力", padding=8)
        area.grid(row=0, column=0, sticky="ew")
        area.columnconfigure(1, weight=1)

        buttons = ttk.Frame(area)
        buttons.grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.select_button = ttk.Button(
            buttons,
            text="ファイルを選択",
            command=self._select_files,
        )
        self.select_button.grid(row=0, column=0, sticky="ew")
        self.manual_button = ttk.Button(
            buttons,
            text="手動検索に戻す",
            command=self._reset_manual,
        )
        self.manual_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        list_frame = ttk.Frame(area)
        list_frame.grid(row=0, column=1, sticky="ew")
        list_frame.columnconfigure(0, weight=1)
        self.input_list = tk.Listbox(list_frame, height=3, activestyle="none")
        self.input_list.grid(row=0, column=0, sticky="ew")
        input_scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.input_list.yview,
        )
        input_scrollbar.grid(row=0, column=1, sticky="ns")
        self.input_list.configure(yscrollcommand=input_scrollbar.set)
        ttk.Label(
            list_frame,
            text="デスクトップまたはExplorerからファイルをここへドロップできます",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.msg_status_var = tk.StringVar(value="手動検索のみ")
        ttk.Label(area, textvariable=self.msg_status_var).grid(
            row=1,
            column=1,
            sticky="w",
            pady=(4, 0),
        )

    def _build_search_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="検索", padding=8)
        area.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        area.columnconfigure(0, weight=1)

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(area, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=0, sticky="ew")
        self.search_var.trace_add("write", self._search_changed)

        self.auxiliary_status_var = tk.StringVar(value="補助照合：使用なし")
        ttk.Label(area, textvariable=self.auxiliary_status_var, wraplength=650).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

    def _build_candidate_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="候補", padding=8)
        area.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)

        columns = ("primary", "auxiliary", "path")
        self.candidate_tree = ttk.Treeview(
            area,
            columns=columns,
            show="headings",
            height=10,
            selectmode="browse",
        )
        self.candidate_tree.heading("primary", text="一致した主検索語")
        self.candidate_tree.heading("auxiliary", text="一致した補助検索語")
        self.candidate_tree.heading("path", text="絶対パス")
        self.candidate_tree.column("primary", width=145, stretch=False)
        self.candidate_tree.column("auxiliary", width=145, stretch=False)
        self.candidate_tree.column("path", width=480, stretch=True)
        self.candidate_tree.grid(row=0, column=0, sticky="nsew")

        vertical = ttk.Scrollbar(area, orient="vertical", command=self.candidate_tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(
            area,
            orient="horizontal",
            command=self.candidate_tree.xview,
        )
        horizontal.grid(row=1, column=0, sticky="ew")
        self.candidate_tree.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        self.candidate_tree.bind("<Double-1>", self._open_candidate_by_click)
        self.candidate_tree.bind("<Return>", self._open_selected_candidate)
        self.candidate_tree.bind(
            "<<TreeviewSelect>>",
            self._candidate_selection_changed,
        )

        self.candidate_status_var = tk.StringVar(value="検索語を入力してください。")
        ttk.Label(area, textvariable=self.candidate_status_var).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

        selected_path_area = ttk.Frame(area)
        selected_path_area.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        selected_path_area.columnconfigure(1, weight=1)
        ttk.Label(selected_path_area, text="選択中の絶対パス：").grid(
            row=0,
            column=0,
            sticky="nw",
        )
        self.selected_candidate_path_var = tk.StringVar(value=SELECTED_CANDIDATE_PATH_GUIDANCE)
        self.selected_candidate_path_label = ttk.Label(
            selected_path_area,
            textvariable=self.selected_candidate_path_var,
            wraplength=620,
            justify="left",
            anchor="w",
        )
        self.selected_candidate_path_label.grid(row=0, column=1, sticky="ew")

        ttk.Label(
            area,
            text=(
                "候補をダブルクリックするとExplorerで確認できます。"
                "開くだけでは保存先は確定されません。"
            ),
            wraplength=760,
        ).grid(row=4, column=0, sticky="w", pady=(2, 0))

    def _build_action_area(self, parent: ttk.Frame) -> None:
        area = ttk.Frame(parent)
        area.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            area.columnconfigure(column, weight=1)

        self.confirm_button = ttk.Button(
            area,
            text="選択パスを確定・コピー",
            command=self._confirm_candidate,
        )
        self.confirm_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.pending_button = ttk.Button(
            area,
            text="保存先未定",
            command=self._confirm_pending,
        )
        self.pending_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.reject_button = ttk.Button(area, text="却下", command=self._reject)
        self.reject_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_codex_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="Codexへ相談", padding=8)
        area.grid(row=0, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        area.rowconfigure(2, weight=1)

        ttk.Label(
            area,
            text=(
                "候補に納得できない場合は、以下のファイルをPCにインストールした"
                "Codexへ添付し、表示されたプロンプトを送信してください。"
            ),
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, sticky="ew")

        self.attachment_guidance_var = tk.StringVar(value="添付案内を準備中です。")
        ttk.Label(
            area,
            textvariable=self.attachment_guidance_var,
            wraplength=420,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 6))

        prompt_frame = ttk.Frame(area)
        prompt_frame.grid(row=2, column=0, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)
        self.prompt_text = tk.Text(prompt_frame, height=18, wrap="word", state="disabled")
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        prompt_scrollbar = ttk.Scrollbar(
            prompt_frame,
            orient="vertical",
            command=self.prompt_text.yview,
        )
        prompt_scrollbar.grid(row=0, column=1, sticky="ns")
        self.prompt_text.configure(yscrollcommand=prompt_scrollbar.set)

        self.codex_button = ttk.Button(
            area,
            text="Codex相談用プロンプトをコピー",
            command=self._copy_codex_prompt,
        )
        self.codex_button.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_result_area(self, parent: ttk.Frame) -> None:
        area = ttk.LabelFrame(parent, text="確定結果と命名例", padding=8)
        area.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        area.columnconfigure(0, weight=1)

        self.confirmed_path_var = tk.StringVar()
        confirmed = ttk.Entry(
            area,
            textvariable=self.confirmed_path_var,
            state="readonly",
        )
        confirmed.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            area,
            text="継続テーマ：テーマ名\n一時案件：YYYYMMDD_案件名",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _resolve_runtime_paths(self) -> None:
        settings_path = self._settings_path_arg or default_settings_path()
        runtime_directory = settings_path.parent
        self.settings_path = settings_path
        self.catalog_path = self._catalog_path_arg or runtime_directory / CATALOG_FILE_NAME
        self.audit_path = self._audit_path_arg or runtime_directory / AUDIT_FILE_NAME

    def _manual_action_text(self) -> str:
        settings_path = self.settings_path or Path(SETTINGS_FILE_NAME)
        return f"\n\n設定保存先: {settings_path}\nREADME: {README_PATH}"

    def _show_issues(self, issues: list[str]) -> None:
        if not issues:
            return
        message = "\n".join(f"・{issue}" for issue in issues) + self._manual_action_text()
        self.root.after_idle(lambda: messagebox.showwarning(APP_TITLE, message, parent=self.root))

    def _load_runtime(self) -> None:
        self._set_ready(False)
        self.update_button.state(["disabled"])
        issues: list[str] = []
        try:
            self._resolve_runtime_paths()
            assert self.settings_path is not None
            self.settings = load_settings(self.settings_path)
        except (SettingsError, OSError) as exc:
            issues.append(f"settings.jsonを利用できません。手動で確認してください。{exc}")
            self.operation_status_var.set("設定を読み込めません。READMEを確認してください。")
            self._show_issues(issues)
            return

        self.current_root_var.set(str(self.settings.current_year_root))
        self.previous_root_var.set(str(self.settings.previous_year_root))
        self.pending_root_var.set(str(self.settings.pending_root))

        roots_available = True
        if not self.settings.current_year_root.is_dir():
            issues.append("今年度フォルダが存在しません。settings.jsonを修正してください。")
            roots_available = False
        if not self.settings.previous_year_root.is_dir():
            issues.append("昨年度フォルダが存在しません。settings.jsonを修正してください。")
            roots_available = False
        if not self.settings.pending_root.is_dir():
            issues.append("保存先未定フォルダが存在しません。手動で作成してください。")

        if roots_available:
            self.update_button.state(["!disabled"])
        else:
            self.catalog_status_var.set("年度フォルダの設定を確認してください。")
            self.operation_status_var.set("年度フォルダがないため処理を開始できません。")
            self._show_issues(issues)
            return

        try:
            assert self.catalog_path is not None
            catalog = load_catalog(self.catalog_path)
            self.session = RecommenderSession(
                self.settings,
                catalog,
                audit_path=self.audit_path,
            )
        except CatalogError as exc:
            issues.append(f"カタログを利用できません。フォルダ構成を更新してください。{exc}")
            self.catalog_status_var.set("未読込：フォルダ構成を更新してください。")
            self.operation_status_var.set("カタログ更新後に検索を開始できます。")
        except RankingError:
            issues.append(
                "カタログと年度フォルダの設定が一致しません。フォルダ構成を更新してください。"
            )
            self.catalog_status_var.set("設定との不一致：フォルダ構成を更新してください。")
            self.operation_status_var.set("カタログ更新後に検索を開始できます。")
        else:
            self._set_ready(True)
            self._render_all()
            self.operation_status_var.set("検索語を入力するか、ファイルを選択してください。")
            self.search_entry.focus_set()

        self._show_issues(issues)

    def _set_ready(self, ready: bool) -> None:
        widgets = (
            self.select_button,
            self.manual_button,
            self.search_entry,
            self.confirm_button,
            self.pending_button,
            self.reject_button,
            self.codex_button,
        )
        state = "!disabled" if ready else "disabled"
        for widget in widgets:
            widget.state([state])

    def _catalog_summary(self, skipped_count: int | None = None) -> str:
        if self.session is None:
            return "未読込"
        catalog = self.session.catalog
        summary = f"最終走査 {format_scanned_at(catalog.scanned_at)} / 収録{catalog.folder_count}件"
        if skipped_count is not None:
            summary += f"、スキップ{skipped_count}件"
        return summary

    def _update_catalog(self) -> None:
        if self.settings is None or self.catalog_path is None:
            return
        self.update_button.state(["disabled"])
        self.operation_status_var.set("フォルダ構成を更新しています。")
        self.root.update_idletasks()
        try:
            result = update_catalog(self.settings, catalog_path=self.catalog_path)
            if self.session is None:
                self.session = RecommenderSession(
                    self.settings,
                    result.catalog,
                    audit_path=self.audit_path,
                )
            else:
                self.session.replace_catalog(result.catalog)
        except (CatalogError, RankingError) as exc:
            self.operation_status_var.set("カタログ更新に失敗しました。既存カタログは維持します。")
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)
        else:
            self._set_ready(True)
            self._render_all(skipped_count=result.skipped_count)
            self.operation_status_var.set(
                f"フォルダ構成を更新しました。収録{result.catalog.folder_count}件、"
                f"スキップ{result.skipped_count}件"
            )
            self.search_entry.focus_set()
        finally:
            self.update_button.state(["!disabled"])

    def _initialize_dnd(self) -> None:
        if sys.platform != "win32":
            return
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            TkinterDnD.require(self.root)
            self.input_list.drop_target_register(DND_FILES)
            self.input_list.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            return

    def _select_files(self) -> None:
        if self.session is None:
            return
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="保存先を判定するファイルを選択",
            filetypes=(("すべてのファイル", "*.*"), ("Outlook MSG", "*.msg")),
        )
        if not selected:
            return
        self._accept_files(selected)

    def _on_drop(self, event: tk.Event) -> None:
        selected = self.root.tk.splitlist(event.data)
        if not selected:
            return
        self._accept_files(selected)

    def _accept_files(self, selected: Iterable[str | Path]) -> None:
        if self.session is None:
            return
        try:
            self.session.select_files(selected)
        except (InputSelectionError, FileNotFoundError, ValueError) as exc:
            self.operation_status_var.set("入力を受け付けられませんでした。再選択してください。")
            messagebox.showwarning(APP_TITLE, str(exc), parent=self.root)
            return
        except Exception:
            self.operation_status_var.set("入力処理に失敗しました。再選択してください。")
            messagebox.showwarning(
                APP_TITLE,
                "ファイルを処理できませんでした。別のファイルを選択してください。",
                parent=self.root,
            )
            return

        self._set_search_text(self.session.search_text)
        self.confirmed_path_var.set("")
        self._render_all()
        self.operation_status_var.set("ファイルから検索語を生成しました。")
        self.search_entry.focus_set()
        self.search_entry.icursor(tk.END)

    def _reset_manual(self) -> None:
        if self.session is None:
            return
        self.session.reset_manual()
        self._set_search_text("")
        self.confirmed_path_var.set("")
        self._render_all()
        self.operation_status_var.set("手動検索へ切り替えました。")
        self.search_entry.focus_set()

    def _set_search_text(self, value: str) -> None:
        self._changing_search = True
        try:
            self.search_var.set(value)
        finally:
            self._changing_search = False

    def _search_changed(self, *_args: object) -> None:
        if self._changing_search or self.session is None:
            return
        try:
            self.session.apply_search_text(self.search_var.get())
        except RankingError:
            self.operation_status_var.set(
                "カタログと設定が一致しません。フォルダ構成を更新してください。"
            )
            self._clear_candidates()
            return
        self.confirmed_path_var.set("")
        self._render_candidates()
        self._render_auxiliary_status()
        self._render_consultation()

    def _render_all(self, skipped_count: int | None = None) -> None:
        if self.session is None:
            return
        self.catalog_status_var.set(self._catalog_summary(skipped_count))
        self._render_input()
        self._render_candidates()
        self._render_auxiliary_status()
        self._render_consultation()

    def _render_input(self) -> None:
        assert self.session is not None
        state = self.session.input_state
        self.input_list.delete(0, tk.END)
        if state.file_paths:
            for path in state.file_paths:
                self.input_list.insert(tk.END, str(path))
        else:
            self.input_list.insert(tk.END, "（ファイルなし：手動検索のみ）")
        self.msg_status_var.set(state.msg_status)

    def _clear_candidates(self) -> None:
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)

    def _render_candidates(self) -> None:
        assert self.session is not None
        self.selected_candidate_path_var.set(SELECTED_CANDIDATE_PATH_GUIDANCE)
        self._clear_candidates()
        for index, candidate in enumerate(self.session.candidates):
            self.candidate_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    " / ".join(candidate.matched_primary_terms) or "—",
                    " / ".join(candidate.matched_auxiliary_terms) or "—",
                    candidate.absolute_path,
                ),
            )

        if not self.session.input_state.current_primary_terms:
            self.candidate_status_var.set("検索語を入力してください。")
        elif not self.session.candidates:
            self.candidate_status_var.set(
                "一致候補がありません。業務テーマ、設備名、案件名等を追加してください。"
            )
        else:
            self.candidate_status_var.set(f"候補{len(self.session.candidates)}件")

    def _render_auxiliary_status(self) -> None:
        assert self.session is not None
        state = self.session.input_state
        if state.kind is not InputKind.MSG:
            self.auxiliary_status_var.set("補助照合：使用なし")
            return
        if not state.auxiliary_terms:
            self.auxiliary_status_var.set("MSG本文の補助検索語：なし")
            return
        matched = any(candidate.auxiliary_match_count for candidate in self.session.candidates)
        if matched:
            self.auxiliary_status_var.set(
                f"MSG本文の補助照合：使用（{len(state.auxiliary_terms)}語、候補に一致あり）"
            )
        else:
            self.auxiliary_status_var.set(
                f"MSG本文の補助照合：{len(state.auxiliary_terms)}語生成、候補への一致なし"
            )

    def _set_prompt_text(self, text: str) -> None:
        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text)
        self.prompt_text.configure(state="disabled")

    def _render_consultation(self) -> None:
        assert self.session is not None
        self.consultation = self.session.build_consultation()
        self.attachment_guidance_var.set(self.consultation.attachment_guidance)
        self._set_prompt_text(self.consultation.prompt)

    def _selected_candidate_path(self) -> str | None:
        if self.session is None:
            return None
        selection = self.candidate_tree.selection()
        if not selection:
            return None
        try:
            index = int(selection[0])
            return self.session.candidates[index].absolute_path
        except (ValueError, IndexError):
            return None

    def _candidate_selection_changed(self, _event: tk.Event | None = None) -> None:
        path = self._selected_candidate_path()
        self.selected_candidate_path_var.set(
            path if path is not None else SELECTED_CANDIDATE_PATH_GUIDANCE
        )

    def _open_candidate_by_click(self, event: tk.Event) -> None:
        if self.candidate_tree.identify_region(event.x, event.y) != "cell":
            return
        row = self.candidate_tree.identify_row(event.y)
        if not row:
            return
        self.candidate_tree.selection_set(row)
        self._candidate_selection_changed()
        self._open_selected_candidate()

    def _open_selected_candidate(self, _event: tk.Event | None = None) -> None:
        path = self._selected_candidate_path()
        if path is None:
            return
        try:
            open_folder(path)
        except FileNotFoundError:
            self.operation_status_var.set(
                "候補フォルダが見つかりません。フォルダ構成を更新してください。"
            )
        except OSError:
            self.operation_status_var.set(
                "候補フォルダをExplorerで開けませんでした。パスを確認して再度お試しください。"
            )

    def _confirm_candidate(self) -> None:
        path = self._selected_candidate_path()
        if path is None:
            messagebox.showwarning(
                APP_TITLE,
                "候補一覧から保存先を1件選択してください。",
                parent=self.root,
            )
            return
        self._finalize(DecisionType.CANDIDATE, path)

    def _confirm_pending(self) -> None:
        if self.settings is None:
            return
        self._finalize(DecisionType.PENDING, str(self.settings.pending_root))

    def _reject(self) -> None:
        self._finalize(DecisionType.REJECTED, None)

    def _copy_text(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _finalize(self, decision_type: DecisionType, path: str | None) -> None:
        if self.session is None:
            return
        if path is not None:
            try:
                self._copy_text(path)
            except tk.TclError:
                messagebox.showerror(
                    APP_TITLE,
                    "クリップボードへコピーできませんでした。",
                    parent=self.root,
                )
                return

        try:
            self.session.record_decision(decision_type, path)
        except AuditError as exc:
            suffix = "パスはコピー済みです。" if path is not None else ""
            self.operation_status_var.set(f"Auditへ記録できませんでした。{suffix}")
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)
            return

        if decision_type is DecisionType.REJECTED:
            self.confirmed_path_var.set("却下")
            self.operation_status_var.set("却下をAuditへ記録しました。")
            return

        assert path is not None
        self.confirmed_path_var.set(path)
        if decision_type is DecisionType.PENDING and not Path(path).is_dir():
            self.operation_status_var.set(
                "保存先未定パスをコピーしました。フォルダが存在しないため手動で作成してください。"
            )
        else:
            self.operation_status_var.set("確定パスをコピーし、Auditへ記録しました。")

    def _copy_codex_prompt(self) -> None:
        if self.consultation is None:
            return
        try:
            copy_codex_consultation(self.consultation, self.root)
        except tk.TclError:
            messagebox.showerror(
                APP_TITLE,
                "クリップボードへコピーできませんでした。",
                parent=self.root,
            )
            return
        self.operation_status_var.set(
            "Codex相談用プロンプトをコピーしました。画面の添付案内を確認してください。"
        )


def create_main_window(
    *,
    settings_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> tk.Tk:
    """Create the Step 5 window without starting the event loop."""
    root = tk.Tk()
    RecommenderApp(
        root,
        settings_path=settings_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
    )
    return root
