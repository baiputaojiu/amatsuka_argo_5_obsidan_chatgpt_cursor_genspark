import json
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DOCX_DOCUMENT_XML = (
    '<?xml version="1.0"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body><w:p><w:r><w:t>設備 秋田</w:t></w:r></w:p></w:body></w:document>"
)


def _write_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", _DOCX_DOCUMENT_XML)
    return path


@pytest.fixture(scope="module")
def tk_root():
    """Share one Tk interpreter, because repeated create-and-destroy cycles are unreliable."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


def _temporary_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from onedrive_destination_recommender.catalog import Catalog, write_catalog_atomic

    current = tmp_path / "020_FY_CURRENT"
    previous = tmp_path / "010_FY_PREVIOUS"
    pending = current / "（Pending）未分類"
    destination = current / "設備"
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


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_document_input_reports_status_without_adding_screen_elements(
    tmp_path: Path,
    tk_root,
) -> None:
    from onedrive_destination_recommender.app import RecommenderApp

    settings_path, catalog_path, audit_path, _destination = _temporary_runtime(tmp_path)
    root = tk_root
    existing_widgets = set(root.winfo_children())
    app = RecommenderApp(
        root,
        settings_path=settings_path,
        catalog_path=catalog_path,
        audit_path=audit_path,
    )

    try:
        root.update_idletasks()
        assert app.session is not None
        widgets_before = len(root.winfo_children())
        columns_before = len(app.candidate_tree["columns"])

        app.session.select_files([_write_docx(tmp_path / "設備仕様.docx")])
        app._render_all()
        root.update_idletasks()

        assert app.msg_status_var.get() == "本文を利用：1/1件"
        assert app.auxiliary_status_var.get().startswith("ファイル本文の補助照合")
        assert "秋田" in str(app.session.input_state.auxiliary_terms)
        assert "秋田" not in app.search_var.get()

        app.session.select_files([tmp_path / "メモ.txt"])
        app._render_all()
        root.update_idletasks()

        assert app.msg_status_var.get() == "ファイル名のみ使用（本文解析なし）"
        assert app.auxiliary_status_var.get() == "ファイル本文の補助検索語：なし"

        assert len(root.winfo_children()) == widgets_before
        assert len(app.candidate_tree["columns"]) == columns_before
    finally:
        for widget in root.winfo_children():
            if widget not in existing_widgets:
                widget.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_step5_window_connects_search_confirmation_audit_and_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tk_root,
) -> None:
    from onedrive_destination_recommender.app import APP_TITLE, RecommenderApp

    settings_path, catalog_path, audit_path, destination = _temporary_runtime(tmp_path)
    root = tk_root
    clipboard: list[str] = []
    monkeypatch.setattr(root, "clipboard_clear", clipboard.clear)
    monkeypatch.setattr(root, "clipboard_append", clipboard.append)
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
        app.candidate_tree.selection_set("0")
        app._confirm_candidate()

        assert clipboard == [str(destination)]
        assert audit_path.read_text(encoding="utf-8").count("\n") == 1
        assert app.confirmed_path_var.get() == str(destination)

        app.search_var.set("設備 追加")
        assert app.confirmed_path_var.get() == ""

        clipboard.clear()
        app._copy_codex_prompt()
        assert app.consultation is not None
        assert clipboard == [app.consultation.prompt]
        assert app.consultation.attachment_guidance not in clipboard[0]

        from onedrive_destination_recommender import app as app_module

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
