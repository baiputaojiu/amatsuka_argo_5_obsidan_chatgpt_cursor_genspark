import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from onedrive_destination_recommender import ranking
from onedrive_destination_recommender.audit import (
    AUDIT_KEYS,
    AuditError,
    DecisionType,
    append_audit_record,
    build_audit_record,
    did_auxiliary_change_top_ten,
    manual_terms_were_used,
)
from onedrive_destination_recommender.ranking import prepare_folders, rank_candidates
from onedrive_destination_recommender.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        current_year_root=tmp_path / "020_FY_CURRENT",
        previous_year_root=tmp_path / "010_FY_PREVIOUS",
        pending_root=tmp_path / "020_FY_CURRENT" / "（Pending）未分類",
        candidate_count=10,
        excluded_folder_names=("除外サンプル",),
    )


def _candidates(tmp_path: Path):
    settings = _settings(tmp_path)
    prepared = prepare_folders(
        [settings.current_year_root / "設備_カメラ"],
        settings,
    )
    return rank_candidates(prepared, settings, ["設備"])


def test_append_audit_record_writes_exact_minimal_schema(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    candidates = _candidates(tmp_path)
    record = build_audit_record(
        input_files=[r"C:\秘密\設備資料.pdf"],
        ranked_candidates=candidates,
        decision_type=DecisionType.CANDIDATE,
        confirmed_path=candidates[0].absolute_path,
        catalog_scanned_at="2026-08-01T00:00:00+00:00",
        manual_terms_used=True,
        automatic_terms_zero_candidates=False,
        auxiliary_changed_top_ten=None,
        recorded_at=datetime(2026, 8, 1, 1, 2, 3, tzinfo=UTC),
    )

    append_audit_record(record, audit_path)

    line = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(line) == 1
    data = json.loads(line[0])
    assert frozenset(data) == AUDIT_KEYS
    assert data["input_file_names"] == ["設備資料.pdf"]
    assert data["decision_type"] == "候補選択"
    assert data["top_ranked_path"] == candidates[0].absolute_path
    assert data["confirmed_path"] == candidates[0].absolute_path
    serialized = line[0].casefold()
    assert "msg本文" not in serialized
    assert "添付内容" not in serialized
    assert "codex相談プロンプト" not in serialized
    assert "検索語全文" not in serialized


def test_append_audit_record_adds_one_line_per_confirmation(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    record = build_audit_record(
        input_files=[],
        ranked_candidates=(),
        decision_type=DecisionType.REJECTED,
        confirmed_path=None,
        catalog_scanned_at="2026-08-01T00:00:00+00:00",
        manual_terms_used=True,
        automatic_terms_zero_candidates=None,
        auxiliary_changed_top_ten=None,
        recorded_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    append_audit_record(record, audit_path)
    append_audit_record(record, audit_path)

    assert len(audit_path.read_text(encoding="utf-8").splitlines()) == 2


def test_append_audit_record_does_not_create_runtime_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "audit.jsonl"
    record = build_audit_record(
        input_files=[],
        ranked_candidates=(),
        decision_type=DecisionType.PENDING,
        confirmed_path=tmp_path / "pending",
        catalog_scanned_at="2026-08-01T00:00:00+00:00",
        manual_terms_used=True,
        automatic_terms_zero_candidates=None,
        auxiliary_changed_top_ten=None,
    )

    with pytest.raises(AuditError, match="存在しません"):
        append_audit_record(record, missing)

    assert not missing.parent.exists()


@pytest.mark.parametrize(
    ("initial", "final", "manual_only", "expected"),
    [
        (("設備",), ("設備",), False, False),
        (("設備",), ("設備", "カメラ"), False, True),
        (("設備", "カメラ"), ("カメラ", "設備"), False, False),
        ((), ("設備",), True, True),
        ((), (), True, False),
    ],
)
def test_manual_terms_were_used_follows_normalized_set_rule(
    initial: tuple[str, ...],
    final: tuple[str, ...],
    manual_only: bool,
    expected: bool,
) -> None:
    assert manual_terms_were_used(initial, final, manual_only=manual_only) is expected


def test_auxiliary_change_compares_top_ten_path_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    prepared = prepare_folders(
        [
            settings.current_year_root / "案件_a_通常",
            settings.current_year_root / "案件_b_特別",
        ],
        settings,
    )

    assert did_auxiliary_change_top_ten(prepared, settings, ["案件"], ["特別"])
    assert not did_auxiliary_change_top_ten(prepared, settings, ["案件"], ["未一致"])


def test_ranking_core_has_no_audit_dependency() -> None:
    assert "audit" not in ranking.__dict__
    assert "onedrive_destination_recommender.audit" not in Path(ranking.__file__).read_text(
        encoding="utf-8"
    )
