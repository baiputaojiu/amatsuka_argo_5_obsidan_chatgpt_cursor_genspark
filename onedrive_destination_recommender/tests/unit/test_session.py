import json
import zipfile
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from onedrive_destination_recommender import session as session_module
from onedrive_destination_recommender.audit import DecisionType
from onedrive_destination_recommender.catalog import Catalog
from onedrive_destination_recommender.document_reader import DocumentSearchTerms
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


def _document_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auxiliary_terms: tuple[str, ...],
    parsed_count: int,
    target_count: int,
    warning: str | None = None,
) -> None:
    result = DocumentSearchTerms(
        auxiliary_terms=auxiliary_terms,
        parsed_count=parsed_count,
        target_count=target_count,
        warning=warning,
    )
    monkeypatch.setattr(session_module, "build_document_terms", lambda _paths: result)


def test_document_body_terms_stay_out_of_the_editable_search_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備_秋田"))
    _document_result(monkeypatch, auxiliary_terms=("秋田",), parsed_count=1, target_count=1)
    input_path = tmp_path / "設備.xlsx"
    input_path.touch()

    state = session.select_files([input_path])

    assert state.kind is InputKind.FILES
    assert session.search_text == "設備"
    assert "秋田" not in session.search_text
    assert state.auxiliary_terms == ("秋田",)
    assert state.msg_status == "本文を利用：1/1件"


def test_document_status_reports_partial_and_failed_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備"))
    _document_result(
        monkeypatch,
        auxiliary_terms=(),
        parsed_count=1,
        target_count=2,
        warning="一部のファイルの本文を利用できませんでした。",
    )
    input_paths = (tmp_path / "設備.xlsx", tmp_path / "図面.pdf")
    for path in input_paths:
        path.touch()

    state = session.select_files(input_paths)

    assert state.msg_status == "本文を利用：1/2件（一部のファイルの本文を利用できませんでした。）"


def test_unsupported_files_keep_the_mvp0_name_only_behaviour(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(settings, _catalog(settings, "設備", "週報"))
    input_paths = (tmp_path / "設備.txt", tmp_path / "メモ.md")
    for path in input_paths:
        path.touch()

    state = session.select_files(input_paths)

    assert state.auxiliary_terms == ()
    assert state.msg_status == "ファイル名のみ使用（本文解析なし）"
    assert [candidate.relative_path for candidate in session.candidates] == ["設備"]


def test_extraction_outcome_changes_neither_primary_terms_nor_candidate_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    selected = [tmp_path / "設備.xlsx"]
    selected[0].touch()

    failed_session = RecommenderSession(settings, _catalog(settings, "設備", "設備_秋田"))
    _document_result(monkeypatch, auxiliary_terms=(), parsed_count=0, target_count=1)
    failed_state = failed_session.select_files(selected)

    parsed_session = RecommenderSession(settings, _catalog(settings, "設備", "設備_秋田"))
    _document_result(monkeypatch, auxiliary_terms=("秋田",), parsed_count=1, target_count=1)
    parsed_state = parsed_session.select_files(selected)

    assert parsed_state.current_primary_terms == failed_state.current_primary_terms
    assert {candidate.absolute_path for candidate in parsed_session.candidates} == {
        candidate.absolute_path for candidate in failed_session.candidates
    }


def test_audit_records_auxiliary_effect_for_document_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    audit_path = tmp_path / "audit.jsonl"
    session = RecommenderSession(
        settings,
        _catalog(settings, "設備", "設備_秋田"),
        audit_path=audit_path,
    )
    _document_result(monkeypatch, auxiliary_terms=("秋田",), parsed_count=1, target_count=1)
    input_path = tmp_path / "設備.xlsx"
    input_path.touch()
    session.select_files([input_path])

    record = session.record_decision(
        DecisionType.CANDIDATE,
        session.candidates[0].absolute_path,
    )

    assert record.auxiliary_changed_top_ten is not None
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert set(persisted) == {
        "recorded_at",
        "input_file_names",
        "top_ranked_path",
        "decision_type",
        "confirmed_path",
        "catalog_scanned_at",
        "manual_terms_used",
        "automatic_terms_zero_candidates",
        "auxiliary_changed_top_ten",
    }


def test_manual_input_still_records_no_auxiliary_comparison(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = RecommenderSession(
        settings,
        _catalog(settings, "設備"),
        audit_path=tmp_path / "audit.jsonl",
    )
    session.apply_search_text("設備")

    record = session.record_decision(
        DecisionType.CANDIDATE,
        session.candidates[0].absolute_path,
    )

    assert record.auxiliary_changed_top_ten is None


def test_extracted_document_text_never_reaches_any_runtime_file(tmp_path: Path) -> None:
    from onedrive_destination_recommender.catalog import write_catalog_atomic

    secret = "zzqqsecretmarker"
    document_xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>設備 {secret}</w:t></w:r></w:p></w:body></w:document>"
    )
    document_path = tmp_path / "設備仕様.docx"
    with zipfile.ZipFile(document_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    settings = _settings(tmp_path)
    catalog = _catalog(settings, "設備")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    write_catalog_atomic(catalog, runtime_dir / "catalog.json")
    audit_path = runtime_dir / "audit.jsonl"

    session = RecommenderSession(settings, catalog, audit_path=audit_path)
    session.select_files([document_path])
    assert secret in str(session.input_state.auxiliary_terms)

    session.apply_search_text("設備")
    session.record_decision(DecisionType.CANDIDATE, session.candidates[0].absolute_path)
    session.build_consultation()

    written = list(runtime_dir.iterdir())
    assert {path.name for path in written} == {"catalog.json", "audit.jsonl"}
    for path in written:
        assert secret not in path.read_text(encoding="utf-8"), path.name


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
