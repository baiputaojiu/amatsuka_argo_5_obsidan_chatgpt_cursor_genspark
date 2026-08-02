import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Any

from onedrive_destination_recommender.ranking import (
    Candidate,
    PreparedFolder,
    rank_candidates,
)
from onedrive_destination_recommender.settings import Settings, default_runtime_dir
from onedrive_destination_recommender.terms import normalize_term_sequence

AUDIT_FILE_NAME = "audit.jsonl"
AUDIT_KEYS = frozenset(
    {
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
)

__all__ = [
    "AUDIT_FILE_NAME",
    "AUDIT_KEYS",
    "AuditError",
    "AuditRecord",
    "DecisionType",
    "append_audit_record",
    "build_audit_record",
    "default_audit_path",
    "did_auxiliary_change_top_ten",
    "manual_terms_were_used",
]


class AuditError(RuntimeError):
    """Raised when an Audit record cannot be appended."""


class DecisionType(StrEnum):
    """User decisions persisted in the Audit log."""

    CANDIDATE = "候補選択"
    PENDING = "保存先未定"
    REJECTED = "却下"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One confirmation record without message contents or search terms."""

    recorded_at: str
    input_file_names: tuple[str, ...]
    top_ranked_path: str | None
    decision_type: DecisionType
    confirmed_path: str | None
    catalog_scanned_at: str
    manual_terms_used: bool
    automatic_terms_zero_candidates: bool | None
    auxiliary_changed_top_ten: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "input_file_names": list(self.input_file_names),
            "top_ranked_path": self.top_ranked_path,
            "decision_type": self.decision_type.value,
            "confirmed_path": self.confirmed_path,
            "catalog_scanned_at": self.catalog_scanned_at,
            "manual_terms_used": self.manual_terms_used,
            "automatic_terms_zero_candidates": self.automatic_terms_zero_candidates,
            "auxiliary_changed_top_ten": self.auxiliary_changed_top_ten,
        }


def default_audit_path() -> Path:
    """Return the default Audit path without creating it."""
    return default_runtime_dir() / AUDIT_FILE_NAME


def _base_name(value: str | os.PathLike[str]) -> str:
    return PureWindowsPath(os.fspath(value)).name


def build_audit_record(
    *,
    input_files: Iterable[str | os.PathLike[str]],
    ranked_candidates: Sequence[Candidate],
    decision_type: DecisionType,
    confirmed_path: str | os.PathLike[str] | None,
    catalog_scanned_at: str,
    manual_terms_used: bool,
    automatic_terms_zero_candidates: bool | None,
    auxiliary_changed_top_ten: bool | None,
    recorded_at: datetime | None = None,
) -> AuditRecord:
    """Build a minimal Audit record from precomputed confirmation state."""
    timestamp = recorded_at or datetime.now(UTC)
    top_ranked_path = ranked_candidates[0].absolute_path if ranked_candidates else None
    return AuditRecord(
        recorded_at=timestamp.astimezone(UTC).isoformat(timespec="seconds"),
        input_file_names=tuple(_base_name(value) for value in input_files),
        top_ranked_path=top_ranked_path,
        decision_type=decision_type,
        confirmed_path=os.fspath(confirmed_path) if confirmed_path is not None else None,
        catalog_scanned_at=catalog_scanned_at,
        manual_terms_used=manual_terms_used,
        automatic_terms_zero_candidates=automatic_terms_zero_candidates,
        auxiliary_changed_top_ten=auxiliary_changed_top_ten,
    )


def append_audit_record(
    record: AuditRecord,
    audit_path: str | Path | None = None,
) -> Path:
    """Append one UTF-8 JSON line without reading any previous Audit records."""
    target = Path(audit_path) if audit_path is not None else default_audit_path()
    if not target.parent.is_dir():
        raise AuditError(f"Auditログ保存先フォルダが存在しません: {target.parent}")

    line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AuditError("audit.jsonlへ記録できませんでした。") from exc
    return target


def manual_terms_were_used(
    initial_primary_terms: Iterable[str],
    final_primary_terms: Iterable[str],
    *,
    manual_only: bool,
) -> bool:
    """Apply the requirement's normalized-set rule for manual term usage."""
    initial = frozenset(normalize_term_sequence(initial_primary_terms))
    final = frozenset(normalize_term_sequence(final_primary_terms))
    return bool(final) if manual_only else initial != final


def did_auxiliary_change_top_ten(
    prepared_folders: Iterable[PreparedFolder],
    settings: Settings,
    primary_terms: Iterable[str],
    auxiliary_terms: Iterable[str],
) -> bool:
    """Compare top-ten path order with and without MSG auxiliary terms."""
    prepared = tuple(prepared_folders)
    primary = tuple(primary_terms)
    auxiliary = tuple(auxiliary_terms)
    without_auxiliary = rank_candidates(prepared, settings, primary, limit=10)
    with_auxiliary = rank_candidates(prepared, settings, primary, auxiliary, limit=10)
    return tuple(item.absolute_path for item in without_auxiliary) != tuple(
        item.absolute_path for item in with_auxiliary
    )
