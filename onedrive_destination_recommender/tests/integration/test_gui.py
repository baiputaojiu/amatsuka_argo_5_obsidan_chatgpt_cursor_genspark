import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


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


@pytest.mark.skipif(sys.platform != "win32", reason="Tkinter Windows GUI test")
def test_step5_window_connects_search_confirmation_audit_and_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tkinter as tk

    from onedrive_destination_recommender.app import APP_TITLE, RecommenderApp

    settings_path, catalog_path, audit_path, destination = _temporary_runtime(tmp_path)
    root = tk.Tk()
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
        missing_settings = tmp_path / "missing" / "settings.json"
        app_module.RecommenderApp(root, settings_path=missing_settings)
        root.update()
        assert len(messages) == 1
        assert str(missing_settings) in messages[0]
        assert "README.md" in messages[0]
    finally:
        root.destroy()
