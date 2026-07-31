from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, tzinfo
from enum import StrEnum
from pathlib import Path

from onedrive_destination_recommender.audit import (
    AuditRecord,
    DecisionType,
    append_audit_record,
    build_audit_record,
    did_auxiliary_change_top_ten,
    manual_terms_were_used,
)
from onedrive_destination_recommender.catalog import Catalog
from onedrive_destination_recommender.codex_prompt import (
    CodexConsultation,
    build_codex_consultation,
)
from onedrive_destination_recommender.msg_reader import (
    MsgSearchTerms,
    build_msg_search_terms,
    is_msg_file,
)
from onedrive_destination_recommender.ranking import (
    Candidate,
    PreparedFolder,
    prepare_folders,
    rank_candidates,
)
from onedrive_destination_recommender.settings import Settings
from onedrive_destination_recommender.terms import (
    initial_terms_from_file_names,
    normalize_terms,
)

__all__ = [
    "InputKind",
    "InputSelectionError",
    "InputState",
    "RecommenderSession",
    "format_scanned_at",
]


class InputKind(StrEnum):
    MANUAL = "manual"
    MSG = "msg"
    FILES = "files"


class InputSelectionError(ValueError):
    """Raised when the selected files do not form one supported input."""


@dataclass(frozen=True, slots=True)
class InputState:
    """The single input currently shown by the application."""

    kind: InputKind
    file_paths: tuple[Path, ...]
    initial_primary_terms: tuple[str, ...]
    current_primary_terms: tuple[str, ...]
    auxiliary_terms: tuple[str, ...]
    msg_status: str
    automatic_terms_zero_candidates: bool | None

    @classmethod
    def manual(cls) -> "InputState":
        return cls(
            kind=InputKind.MANUAL,
            file_paths=(),
            initial_primary_terms=(),
            current_primary_terms=(),
            auxiliary_terms=(),
            msg_status="手動検索のみ",
            automatic_terms_zero_candidates=None,
        )

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.file_paths)


def format_scanned_at(value: str, target_timezone: tzinfo | None = None) -> str:
    """Format a catalog UTC timestamp for the user's local timezone."""
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    localized = timestamp.astimezone(target_timezone)
    return localized.strftime("%Y-%m-%d %H:%M:%S")


def _msg_status(result: MsgSearchTerms) -> str:
    if result.warning:
        return result.warning
    return "MSG解析完了"


class RecommenderSession:
    """Coordinate one input with the existing catalog, ranking, and Audit functions."""

    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        *,
        audit_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.prepared_folders = prepare_folders(catalog.folders, settings)
        self.input_state = InputState.manual()
        self.search_text = ""
        self.candidates: tuple[Candidate, ...] = ()

    def _rank(
        self,
        primary_terms: Iterable[str],
        auxiliary_terms: Iterable[str],
        *,
        prepared_folders: Iterable[PreparedFolder] | None = None,
    ) -> tuple[Candidate, ...]:
        prepared = self.prepared_folders if prepared_folders is None else prepared_folders
        return rank_candidates(prepared, self.settings, primary_terms, auxiliary_terms)

    def reset_manual(self) -> InputState:
        self.input_state = InputState.manual()
        self.search_text = ""
        self.candidates = ()
        return self.input_state

    def select_files(self, selected_paths: Iterable[str | Path]) -> InputState:
        paths = tuple(Path(path).resolve() for path in selected_paths)
        if not paths:
            raise InputSelectionError("ファイルを1件以上選択してください。")

        msg_count = sum(is_msg_file(path) for path in paths)
        if msg_count and len(paths) != 1:
            raise InputSelectionError("MSGは1件ずつ選択してください。")

        if msg_count == 1:
            msg_result = build_msg_search_terms(paths[0])
            state = InputState(
                kind=InputKind.MSG,
                file_paths=paths,
                initial_primary_terms=msg_result.primary_terms,
                current_primary_terms=msg_result.primary_terms,
                auxiliary_terms=msg_result.auxiliary_terms,
                msg_status=_msg_status(msg_result),
                automatic_terms_zero_candidates=None,
            )
        else:
            primary_terms = initial_terms_from_file_names(path.name for path in paths)
            state = InputState(
                kind=InputKind.FILES,
                file_paths=paths,
                initial_primary_terms=primary_terms,
                current_primary_terms=primary_terms,
                auxiliary_terms=(),
                msg_status="ファイル名のみ使用（本文解析なし）",
                automatic_terms_zero_candidates=None,
            )

        candidates = self._rank(state.initial_primary_terms, state.auxiliary_terms)
        self.input_state = replace(
            state,
            automatic_terms_zero_candidates=not bool(candidates),
        )
        self.search_text = " ".join(state.initial_primary_terms)
        self.candidates = candidates
        return self.input_state

    def apply_search_text(self, search_text: str) -> tuple[Candidate, ...]:
        primary_terms = normalize_terms(search_text)
        self.input_state = replace(
            self.input_state,
            current_primary_terms=primary_terms,
        )
        self.search_text = search_text
        self.candidates = self._rank(primary_terms, self.input_state.auxiliary_terms)
        return self.candidates

    def replace_catalog(self, catalog: Catalog) -> None:
        """Replace prepared data only after the new catalog is fully usable."""
        prepared = prepare_folders(catalog.folders, self.settings)
        candidates = self._rank(
            self.input_state.current_primary_terms,
            self.input_state.auxiliary_terms,
            prepared_folders=prepared,
        )
        new_input_state = self.input_state
        if self.input_state.kind is not InputKind.MANUAL:
            initial_candidates = self._rank(
                self.input_state.initial_primary_terms,
                self.input_state.auxiliary_terms,
                prepared_folders=prepared,
            )
            new_input_state = replace(
                self.input_state,
                automatic_terms_zero_candidates=not bool(initial_candidates),
            )

        self.catalog = catalog
        self.prepared_folders = prepared
        self.input_state = new_input_state
        self.candidates = candidates

    def build_consultation(self) -> CodexConsultation:
        return build_codex_consultation(
            input_files=self.input_state.file_paths,
            settings=self.settings,
            search_text=self.search_text,
            candidate_paths=(candidate.absolute_path for candidate in self.candidates),
        )

    def record_decision(
        self,
        decision_type: DecisionType,
        confirmed_path: str | Path | None,
    ) -> AuditRecord:
        if decision_type is DecisionType.REJECTED:
            if confirmed_path is not None:
                raise ValueError("却下時に確定パスは指定できません。")
        elif confirmed_path is None:
            raise ValueError("候補選択または保存先未定には確定パスが必要です。")

        auxiliary_changed = None
        if self.input_state.kind is InputKind.MSG:
            auxiliary_changed = did_auxiliary_change_top_ten(
                self.prepared_folders,
                self.settings,
                self.input_state.current_primary_terms,
                self.input_state.auxiliary_terms,
            )
        record = build_audit_record(
            input_files=self.input_state.file_paths,
            ranked_candidates=self.candidates,
            decision_type=decision_type,
            confirmed_path=confirmed_path,
            catalog_scanned_at=self.catalog.scanned_at,
            manual_terms_used=manual_terms_were_used(
                self.input_state.initial_primary_terms,
                self.input_state.current_primary_terms,
                manual_only=self.input_state.kind is InputKind.MANUAL,
            ),
            automatic_terms_zero_candidates=(
                None
                if self.input_state.kind is InputKind.MANUAL
                else self.input_state.automatic_terms_zero_candidates
            ),
            auxiliary_changed_top_ten=auxiliary_changed,
        )
        append_audit_record(record, self.audit_path)
        return record
