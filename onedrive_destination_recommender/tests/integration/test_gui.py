import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def _temporary_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from onedrive_destination_recommender.catalog import Catalog, write_catalog_atomic

    current = tmp_path / "020_FY_CURRENT"
    previous = tmp_path / "010_FY_PREVIOUS"
    pending = current / "（Pending）未分類"
    destination = current / "設備_絶対パス全文表示を確認するための長い候補フォルダ名"
    pending.mkdir(parents=True)
    previous.mkdir()
    destination.mkdir()

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "current_year_root": str(current),
                "previous_year_root": str(previous),
                "pending_root": str(pending),
                "candidate_count": 10,
                "excluded_folder_names": ["除外サンプル"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    write_catalog_atomic(
        Catalog(
            scanned_at="2026-08-01T00:00:00+00:00",
            folders=(str(destination),),
        ),
        catalog_path,
    )
    return settings_path, catalog_path, tmp_path / "audit.jsonl", destination


def _change_setting(settings_path: Path, key: str, value: str) -> None:
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    data[key] = value
    settings_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def shared_tk_root():
    if sys.platform != "win32":
        pytest.skip("Tkinter Windows GUI test")
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_step5_window_connects_search_confirmation_audit_and_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_tk_root,
) -> None:
    from onedrive_destination_recommender import app as app_module
    from onedrive_destination_recommender.app import (
        APP_TITLE,
        SELECTED_CANDIDATE_PATH_GUIDANCE,
        RecommenderApp,
    )
    from onedrive_destination_recommender.ranking import RankingError

    settings_path, catalog_path, audit_path, destination = _temporary_runtime(tmp_path)
    root = shared_tk_root
    clipboard: list[str] = []
    root.clipboard_clear = clipboard.clear  # type: ignore[method-assign]
    root.clipboard_append = clipboard.append  # type: ignore[method-assign]
    app = RecommenderApp(
        root,
        settings_path=settings_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
    )

    try:
        root.update_idletasks()
        assert root.title() == APP_TITLE
        assert root.winfo_exists()

        app._update_catalog()
        assert app.session is not None
        assert app.session.catalog.folder_count == 2

        app.search_var.set("設備")
        assert len(app.session.candidates) == 1
        assert app.selected_candidate_path_var.get() == SELECTED_CANDIDATE_PATH_GUIDANCE
        before_path_display = (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
            app.search_var.get(),
            app.session.input_state,
            app.session.candidates,
            tuple(app.candidate_tree.get_children()),
        )
        app.candidate_tree.selection_set("0")
        app._candidate_selection_changed()
        assert app.selected_candidate_path_var.get() == str(destination)
        assert int(app.selected_candidate_path_label.cget("wraplength")) == 620
        assert (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
            app.search_var.get(),
            app.session.input_state,
            app.session.candidates,
            tuple(app.candidate_tree.get_children()),
        ) == before_path_display

        def reject_search(_text: str) -> None:
            raise RankingError("synthetic catalog mismatch")

        with monkeypatch.context() as catalog_mismatch:
            catalog_mismatch.setattr(app.session, "apply_search_text", reject_search)
            app.search_var.set("設備 不一致")
            assert app.candidate_tree.get_children() == ()
            assert app.selected_candidate_path_var.get() == SELECTED_CANDIDATE_PATH_GUIDANCE

        app.search_var.set("設備")
        assert len(app.session.candidates) == 1
        app.candidate_tree.selection_set("0")
        app._candidate_selection_changed()

        opened: list[str] = []
        monkeypatch.setattr(app_module, "open_folder", opened.append)
        hit = {"region": "cell", "row": "0"}
        monkeypatch.setattr(
            app.candidate_tree,
            "identify_region",
            lambda _x, _y: hit["region"],
        )
        monkeypatch.setattr(
            app.candidate_tree,
            "identify_row",
            lambda _y: hit["row"],
        )
        before_preview = (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
            app.search_var.get(),
            app.session.input_state,
            app.session.candidates,
            tuple(app.candidate_tree.get_children()),
            app.selected_candidate_path_var.get(),
        )

        assert app.candidate_tree.bind("<Double-1>")
        assert app.candidate_tree.bind("<Return>")
        assert app.candidate_tree.bind("<<TreeviewSelect>>")
        app._open_candidate_by_click(SimpleNamespace(x=1, y=1))
        assert opened == [str(destination)]
        assert app.selected_candidate_path_var.get() == str(destination)
        assert (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
            app.search_var.get(),
            app.session.input_state,
            app.session.candidates,
            tuple(app.candidate_tree.get_children()),
            app.selected_candidate_path_var.get(),
        ) == before_preview

        app._open_selected_candidate()
        assert opened == [str(destination), str(destination)]

        hit["region"] = "heading"
        app._open_candidate_by_click(SimpleNamespace(x=1, y=1))
        hit.update(region="cell", row="")
        app._open_candidate_by_click(SimpleNamespace(x=1, y=1))
        app.candidate_tree.selection_remove("0")
        app._candidate_selection_changed()
        assert app.selected_candidate_path_var.get() == SELECTED_CANDIDATE_PATH_GUIDANCE
        app._open_selected_candidate()
        assert opened == [str(destination), str(destination)]

        app.candidate_tree.selection_set("0")
        app._candidate_selection_changed()
        app._confirm_candidate()

        assert clipboard == [str(destination)]
        assert audit_path.read_text(encoding="utf-8").count("\n") == 1
        assert app.confirmed_path_var.get() == str(destination)

        app.search_var.set("設備 追加")
        assert app.confirmed_path_var.get() == ""
        assert app.selected_candidate_path_var.get() == SELECTED_CANDIDATE_PATH_GUIDANCE

        clipboard.clear()
        app._copy_codex_prompt()
        assert app.consultation is not None
        assert clipboard == [app.consultation.prompt]
        assert app.consultation.attachment_guidance not in clipboard[0]

        messages: list[str] = []
        monkeypatch.setattr(
            app_module.messagebox,
            "showwarning",
            lambda _title, message, **_kwargs: messages.append(message),
        )
        original_widgets = set(root.winfo_children())
        cases = (
            ("missing_settings", "settings.jsonを利用できません"),
            ("invalid_settings", "settings.jsonを利用できません"),
            ("missing_current", "今年度フォルダが存在しません"),
            ("missing_previous", "昨年度フォルダが存在しません"),
            ("missing_pending", "保存先未定フォルダが存在しません"),
            ("missing_catalog", "カタログを利用できません"),
        )
        for case, expected_message in cases:
            case_root = tmp_path / case
            settings_path, catalog_path, audit_path, _destination = _temporary_runtime(case_root)
            if case == "missing_settings":
                settings_path.unlink()
            elif case == "invalid_settings":
                settings_path.write_text("{", encoding="utf-8")
            elif case == "missing_current":
                _change_setting(
                    settings_path,
                    "current_year_root",
                    str(case_root / "missing-current"),
                )
            elif case == "missing_previous":
                _change_setting(
                    settings_path,
                    "previous_year_root",
                    str(case_root / "missing-previous"),
                )
            elif case == "missing_pending":
                _change_setting(
                    settings_path,
                    "pending_root",
                    str(case_root / "missing-pending"),
                )
            elif case == "missing_catalog":
                catalog_path.unlink()

            messages.clear()
            app_module.RecommenderApp(
                root,
                settings_path=settings_path,
                catalog_path=catalog_path,
                audit_path=audit_path,
            )
            root.update()

            assert len(messages) == 1, case
            assert expected_message in messages[0], case
            assert str(settings_path) in messages[0], case
            assert "README.md" in messages[0], case
            for widget in root.winfo_children():
                if widget not in original_widgets:
                    widget.destroy()
    finally:
        for widget in root.winfo_children():
            widget.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_drop_splits_single_and_multiple_paths_without_changing_them(
    tmp_path: Path,
    shared_tk_root,
) -> None:
    import tkinter as tk

    from onedrive_destination_recommender.app import RecommenderApp

    settings_path, catalog_path, audit_path, _destination = _temporary_runtime(tmp_path)
    root = tk.Toplevel(shared_tk_root)
    root.withdraw()
    app = RecommenderApp(
        root,
        settings_path=settings_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
    )
    single = (tmp_path / "カメラ 仕様書（秋田）.docx",)
    multiple = (tmp_path / "図面 [Rev1].pdf", tmp_path / "設備 写真（正面）.png")
    for path in (*single, *multiple):
        path.touch()
    accepted: list[tuple[str, ...]] = []
    app._accept_files = lambda paths: accepted.append(tuple(paths))

    try:
        assert app.input_list.dnd_bind("<<Drop>>")
        for expected in (single, multiple):
            event_data = root.tk.call("list", *(str(path) for path in expected))
            app._on_drop(SimpleNamespace(data=event_data))

        assert accepted == [
            tuple(str(path) for path in single),
            tuple(str(path) for path in multiple),
        ]
    finally:
        root.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_drop_and_file_selection_produce_the_same_result_without_finalizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_tk_root,
) -> None:
    import tkinter as tk

    from onedrive_destination_recommender import app as app_module
    from onedrive_destination_recommender.app import RecommenderApp

    settings_path, catalog_path, audit_path, _destination = _temporary_runtime(tmp_path)
    input_paths = (tmp_path / "設備 仕様書（秋田）.pdf", tmp_path / "設備 [Rev1].docx")
    for path in input_paths:
        path.touch()
    root = tk.Toplevel(shared_tk_root)
    root.withdraw()
    clipboard: list[str] = []
    root.clipboard_clear = clipboard.clear  # type: ignore[method-assign]
    root.clipboard_append = clipboard.append  # type: ignore[method-assign]
    app = RecommenderApp(
        root,
        settings_path=settings_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
    )
    monkeypatch.setattr(
        app_module.filedialog,
        "askopenfilenames",
        lambda **_kwargs: tuple(str(path) for path in input_paths),
    )

    try:
        app._select_files()
        assert app.session is not None
        selected_result = (
            app.session.input_state,
            app.session.search_text,
            app.session.candidates,
        )

        app._reset_manual()
        before_drop_effects = (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
        )
        event_data = root.tk.call("list", *(str(path) for path in input_paths))
        app._on_drop(SimpleNamespace(data=event_data))

        assert (
            app.session.input_state,
            app.session.search_text,
            app.session.candidates,
        ) == selected_result
        assert (
            audit_path.read_bytes() if audit_path.exists() else None,
            tuple(clipboard),
            app.confirmed_path_var.get(),
        ) == before_drop_effects
    finally:
        root.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_dnd_initialization_failure_keeps_file_selection_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_tk_root,
) -> None:
    import tkinter as tk

    from onedrive_destination_recommender import app as app_module
    from onedrive_destination_recommender.app import RecommenderApp

    settings_path, catalog_path, audit_path, _destination = _temporary_runtime(tmp_path)
    input_path = tmp_path / "設備.pdf"
    input_path.touch()
    root = tk.Toplevel(shared_tk_root)
    root.withdraw()
    attempted_imports: list[str] = []
    real_import = builtins.__import__

    def reject_dnd_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "tkinterdnd2":
            attempted_imports.append(name)
            raise ImportError("synthetic tkinterdnd2 failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_dnd_import)
    try:
        app = RecommenderApp(
            root,
            settings_path=settings_path,
            catalog_path=catalog_path,
            audit_path=audit_path,
        )
        assert attempted_imports == ["tkinterdnd2"]
        monkeypatch.setattr(builtins, "__import__", real_import)
        monkeypatch.setattr(
            app_module.filedialog,
            "askopenfilenames",
            lambda **_kwargs: (str(input_path),),
        )

        app._select_files()

        assert app.session is not None
        assert app.session.input_state.file_paths == (input_path.resolve(),)
    finally:
        root.destroy()
