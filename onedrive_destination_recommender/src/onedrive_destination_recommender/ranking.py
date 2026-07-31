import os
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from onedrive_destination_recommender.settings import Settings
from onedrive_destination_recommender.terms import (
    normalize_path_for_matching,
    normalize_term_sequence,
)


class RankingError(ValueError):
    """Raised when catalog paths cannot be related to the configured year roots."""


class YearScope(StrEnum):
    CURRENT = "current"
    PREVIOUS = "previous"


@dataclass(frozen=True, slots=True)
class PreparedFolder:
    """Catalog folder data that remains unchanged while the user edits search terms."""

    year: YearScope
    relative_path: str
    absolute_path: str
    normalized_path: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """A displayable folder candidate and its ranking-only match counts."""

    year: YearScope
    relative_path: str
    absolute_path: str
    primary_match_count: int
    auxiliary_match_count: int
    matched_primary_terms: tuple[str, ...]
    matched_auxiliary_terms: tuple[str, ...]


def _normalized_absolute(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(path: str | Path, root: Path) -> bool:
    normalized_path = _normalized_absolute(path)
    normalized_root = _normalized_absolute(root)
    try:
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:
        return False


def _year_and_relative_path(folder: str | Path, settings: Settings) -> tuple[YearScope, str]:
    absolute_path = os.path.abspath(folder)
    for year, root in (
        (YearScope.CURRENT, settings.current_year_root),
        (YearScope.PREVIOUS, settings.previous_year_root),
    ):
        if _is_within(absolute_path, root):
            relative_path = os.path.relpath(absolute_path, root)
            if relative_path == os.curdir:
                raise RankingError("年度ルート自体は保存先候補に含めません。")
            return year, relative_path
    raise RankingError("年度ルート外のフォルダがカタログに含まれています。")


def prepare_folders(
    folder_paths: Iterable[str | Path],
    settings: Settings,
) -> tuple[PreparedFolder, ...]:
    """Resolve and normalize catalog folders once per catalog or settings update."""
    prepared: list[PreparedFolder] = []
    for folder in folder_paths:
        absolute_path = os.path.abspath(folder)
        year, relative_path = _year_and_relative_path(absolute_path, settings)
        prepared.append(
            PreparedFolder(
                year=year,
                relative_path=relative_path,
                absolute_path=absolute_path,
                normalized_path=normalize_path_for_matching(relative_path),
            )
        )
    return tuple(prepared)


def _matched_terms(normalized_path: str, terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(term for term in terms if term in normalized_path)


def _make_candidate(
    *,
    folder: PreparedFolder,
    primary_terms: Sequence[str],
    auxiliary_terms: Sequence[str],
    inherited_primary_count: int | None = None,
) -> Candidate:
    matched_primary = _matched_terms(folder.normalized_path, primary_terms)
    matched_auxiliary = _matched_terms(folder.normalized_path, auxiliary_terms)
    return Candidate(
        year=folder.year,
        relative_path=folder.relative_path,
        absolute_path=folder.absolute_path,
        primary_match_count=(
            len(matched_primary) if inherited_primary_count is None else inherited_primary_count
        ),
        auxiliary_match_count=len(matched_auxiliary),
        matched_primary_terms=matched_primary,
        matched_auxiliary_terms=matched_auxiliary[:3],
    )


def _relative_key(relative_path: str) -> str:
    return os.path.normcase(os.path.normpath(relative_path))


def _candidate_key(candidate: Candidate) -> tuple[YearScope, str]:
    return candidate.year, _relative_key(candidate.relative_path)


def _ancestor_keys(candidate: Candidate) -> Iterable[tuple[YearScope, str]]:
    parent = os.path.dirname(candidate.relative_path)
    while parent and parent != os.curdir:
        yield candidate.year, _relative_key(parent)
        parent = os.path.dirname(parent)


def _fold_descendants(candidates: Sequence[Candidate]) -> list[Candidate]:
    by_key = {_candidate_key(candidate): candidate for candidate in candidates}
    kept: list[Candidate] = []
    for candidate in candidates:
        if any(
            by_key[ancestor_key].primary_match_count >= candidate.primary_match_count
            for ancestor_key in _ancestor_keys(candidate)
            if ancestor_key in by_key
        ):
            continue
        kept.append(candidate)
    return kept


def _fold_siblings(
    candidates: Sequence[Candidate],
    primary_terms: Sequence[str],
    auxiliary_terms: Sequence[str],
) -> list[Candidate]:
    groups: dict[tuple[YearScope, str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        parent = os.path.dirname(candidate.relative_path)
        if not parent or parent == os.curdir:
            continue
        groups[(candidate.year, _relative_key(parent), candidate.primary_match_count)].append(
            candidate
        )

    folded_members: set[Candidate] = set()
    replacements: list[Candidate] = []
    for siblings in groups.values():
        if len(siblings) < 3:
            continue
        first = siblings[0]
        parent_relative = os.path.dirname(first.relative_path)
        parent_absolute = os.path.dirname(first.absolute_path)
        folded_members.update(siblings)
        replacements.append(
            _make_candidate(
                folder=PreparedFolder(
                    year=first.year,
                    relative_path=parent_relative,
                    absolute_path=parent_absolute,
                    normalized_path=normalize_path_for_matching(parent_relative),
                ),
                primary_terms=primary_terms,
                auxiliary_terms=auxiliary_terms,
                inherited_primary_count=first.primary_match_count,
            )
        )
    return [candidate for candidate in candidates if candidate not in folded_members] + replacements


def _deduplicate(candidates: Sequence[Candidate]) -> list[Candidate]:
    selected: dict[tuple[YearScope, str], Candidate] = {}
    for candidate in candidates:
        key = _candidate_key(candidate)
        existing = selected.get(key)
        if existing is None or (
            candidate.primary_match_count,
            candidate.auxiliary_match_count,
        ) > (
            existing.primary_match_count,
            existing.auxiliary_match_count,
        ):
            selected[key] = candidate
    return list(selected.values())


def _sort_key(candidate: Candidate) -> tuple[int, int, int, str]:
    year_order = 0 if candidate.year is YearScope.CURRENT else 1
    relative_order = unicodedata.normalize("NFKC", candidate.relative_path).casefold()
    return (
        year_order,
        -candidate.primary_match_count,
        -candidate.auxiliary_match_count,
        relative_order,
    )


def rank_candidates(
    prepared_folders: Iterable[PreparedFolder],
    settings: Settings,
    primary_terms: Iterable[str],
    auxiliary_terms: Iterable[str] = (),
    *,
    limit: int | None = None,
) -> tuple[Candidate, ...]:
    """Match, fold, deduplicate, and rank catalog folders in the specified order."""
    normalized_primary = normalize_term_sequence(primary_terms)
    normalized_auxiliary = normalize_term_sequence(auxiliary_terms, auxiliary=True)
    if not normalized_primary:
        return ()

    candidates: list[Candidate] = []
    for folder in prepared_folders:
        candidate = _make_candidate(
            folder=folder,
            primary_terms=normalized_primary,
            auxiliary_terms=normalized_auxiliary,
        )
        if candidate.primary_match_count > 0:
            candidates.append(candidate)

    descendants_folded = _fold_descendants(candidates)
    siblings_folded = _fold_siblings(
        descendants_folded,
        normalized_primary,
        normalized_auxiliary,
    )
    deduplicated = _deduplicate(siblings_folded)
    ranked = sorted(deduplicated, key=_sort_key)

    candidate_limit = settings.candidate_count if limit is None else limit
    if candidate_limit <= 0:
        raise ValueError("limitには1以上の整数を指定してください。")
    return tuple(ranked[:candidate_limit])
