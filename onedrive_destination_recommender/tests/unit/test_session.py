import json
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from onedrive_destination_recommender import session as session_module
from onedrive_destination_recommender.audit import DecisionType
from onedrive_destination_recommender.catalog import Catalog
from onedrive_destination_recommender.msg_reader import MsgSearchTerms
from onedrive_destination_recommender.session import (
    InputKind,
    InputSelectionError,
    RecommenderSession,
    format_scanned_at,
)
from onedrive_destination_recommender.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        current_year_root=tmp_path / "020_FY_CURRENT",
        previous_year_root=tmp_path / "010_FY_PREVIOUS",
        pending_root=tmp_path / "020_FY_CURRENT" / "（Pending）未分類",
        candidate_count=10,
        excluded_folder_names=("除外サンプル",),
    )


def _catalog(settings: Settings, *relative_paths: str) -> Catalog:
    folders = tuple(str(settings.current_year_root / relative) for relative in relative_paths)
    return Catalog(scanned_at="2026-08-01T00:00:00+00:00", folders=folders)


def test_manual_search_updates_candidates_without_file_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備", "週報"))

    candidates = session.apply_search_text("設備 カメラ")

    assert session.input_state.kind is InputKind.MANUAL
    assert session.input_state.automatic_terms_zero_candidates is None
    assert [candidate.relative_path for candidate in candidates] == ["設備"]


def test_file_input_uses_names_and_preserves_initial_zero_measurement(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備", "週報"))
    input_path = tmp_path / "未知案件.pdf"
    input_path.touch()

    state = session.select_files([input_path])
    assert state.kind is InputKind.FILES
    assert state.file_names == ("未知案件.pdf",)
    assert state.automatic_terms_zero_candidates is True
    assert session.candidates == ()

    session.apply_search_text("設備")

    assert len(session.candidates) == 1
    assert session.input_state.automatic_terms_zero_candidates is True


def test_multiple_regular_files_form_one_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備_カメラ"))
    input_paths = (tmp_path / "設備.pdf", tmp_path / "カメラ仕様書.docx")
    for path in input_paths:
        path.touch()

    state = session.select_files(input_paths)

    assert state.kind is InputKind.FILES
    assert state.file_names == ("設備.pdf", "カメラ仕様書.docx")
    assert len(session.candidates) == 1


def test_msg_mixed_selection_is_rejected_without_replacing_current_input(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備"))
    session.apply_search_text("設備")
    input_paths = (tmp_path / "mail.msg", tmp_path / "設備.pdf")
    for path in input_paths:
        path.touch()

    with pytest.raises(InputSelectionError, match="MSGは1件ずつ"):
        session.select_files(input_paths)

    assert session.input_state.kind is InputKind.MANUAL
    assert session.search_text == "設備"


def test_msg_input_keeps_body_terms_hidden_from_search_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備_秋田"))
    result = MsgSearchTerms(
        primary_terms=("設備",),
        auxiliary_terms=("秋田",),
        fully_parsed=True,
        body_available=True,
        warning=None,
    )
    monkeypatch.setattr(session_module, "build_msg_search_terms", lambda _path: result)
    input_path = tmp_path / "mail.msg"
    input_path.touch()

    state = session.select_files([input_path])

    assert state.kind is InputKind.MSG
    assert session.search_text == "設備"
    assert "秋田" not in session.search_text
    assert state.auxiliary_terms == ("秋田",)
    assert state.msg_status == "MSG解析完了"


def test_missing_msg_error_does_not_replace_current_input(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備"))
    session.apply_search_text("設備")

    with pytest.raises(InputSelectionError, match="存在しないファイル"):
        session.select_files([tmp_path / "missing.msg"])

    assert session.input_state.kind is InputKind.MANUAL
    assert session.search_text == "設備"


def test_folder_input_is_rejected_without_replacing_current_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備"))
    session.apply_search_text("設備")
    previous = (session.input_state, session.search_text, session.candidates)
    input_path = tmp_path / "カメラ.pdf"
    input_path.touch()
    folder = tmp_path / "図面フォルダ"
    folder.mkdir()

    with pytest.raises(InputSelectionError, match="フォルダや存在しないファイル"):
        session.select_files([input_path, folder])

    assert (session.input_state, session.search_text, session.candidates) == previous


def test_catalog_replacement_reranks_current_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "週報"))
    session.apply_search_text("設備")
    assert session.candidates == ()

    session.replace_catalog(_catalog(settings, "設備"))

    assert len(session.candidates) == 1
    assert session.candidates[0].relative_path == "設備"


def test_catalog_replacement_preserves_initial_zero_measurement(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "週報"))
    input_path = tmp_path / "設備.pdf"
    input_path.touch()
    session.select_files([input_path])
    assert session.input_state.automatic_terms_zero_candidates is True

    session.replace_catalog(_catalog(settings, "設備"))

    assert len(session.candidates) == 1
    assert session.input_state.automatic_terms_zero_candidates is True


def test_decision_writes_audit_from_single_session_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    audit_path = tmp_path / "audit.jsonl"
    session = RecommenderSession(
        settings,
        _catalog(settings, "設備"),
        audit_path=audit_path,
    )
    session.apply_search_text("設備")
    selected_path = session.candidates[0].absolute_path

    record = session.record_decision(DecisionType.CANDIDATE, selected_path)

    assert record.confirmed_path == selected_path
    assert record.manual_terms_used
    assert record.automatic_terms_zero_candidates is None
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["confirmed_path"] == selected_path


def test_consultation_remains_available_after_regular_file_disappears(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備"))
    missing = tmp_path / "設備.pdf"
    missing.touch()
    session.select_files([missing])
    missing.unlink()

    consultation = session.build_consultation()

    assert str(missing.resolve()) in consultation.attachment_guidance
    assert str(missing.resolve()) in consultation.prompt


def test_catalog_timestamp_is_converted_from_utc() -> None:
    japan = timezone(timedelta(hours=9))

    assert format_scanned_at("2026-08-01T00:00:00+00:00", japan) == "2026-08-01 09:00:00"
    assert format_scanned_at("not-a-timestamp", japan) == "not-a-timestamp"


def test_open_folder_passes_exact_existing_path_to_launcher_once(tmp_path: Path) -> None:
    folder = tmp_path / "（Output）定例成果物"
    folder.mkdir()
    launched: list[str] = []

    session_module.open_folder(folder, launcher=launched.append)

    assert launched == [str(folder)]


def test_open_folder_rejects_missing_path_without_calling_launcher(
    tmp_path: Path,
) -> None:
    launched: list[str] = []

    with pytest.raises(FileNotFoundError):
        session_module.open_folder(
            tmp_path / "存在しない（候補）",
            launcher=launched.append,
        )

    assert launched == []


def test_open_folder_propagates_launcher_os_error(tmp_path: Path) -> None:
    folder = tmp_path / "候補"
    folder.mkdir()

    def failing_launcher(_path: str) -> None:
        raise OSError("synthetic Explorer failure")

    with pytest.raises(OSError, match="synthetic Explorer failure"):
        session_module.open_folder(folder, launcher=failing_launcher)
