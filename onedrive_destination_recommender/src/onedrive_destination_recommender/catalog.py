import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from onedrive_destination_recommender.settings import Settings, default_runtime_dir

CATALOG_FILE_NAME = "catalog.json"
CATALOG_KEYS = frozenset({"scanned_at", "folder_count", "folders"})


class CatalogError(RuntimeError):
    """Raised when folder scanning or catalog persistence fails."""


@dataclass(frozen=True, slots=True)
class Catalog:
    """Persisted catalog content."""

    scanned_at: str
    folders: tuple[str, ...]

    @property
    def folder_count(self) -> int:
        return len(self.folders)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_at": self.scanned_at,
            "folder_count": self.folder_count,
            "folders": list(self.folders),
        }


@dataclass(frozen=True, slots=True)
class CatalogUpdateResult:
    """Catalog plus non-persisted information for the update result display."""

    catalog: Catalog
    skipped_count: int


def default_catalog_path() -> Path:
    """Return the default catalog path without creating it."""
    return default_runtime_dir() / CATALOG_FILE_NAME


def _child_directories(parent: Path, excluded_names: frozenset[str]) -> tuple[list[Path], int]:
    children: list[Path] = []
    skipped_count = 0
    with os.scandir(parent) as entries:
        for entry in entries:
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                skipped_count += 1
                continue
            if not is_directory or entry.name in excluded_names:
                continue
            children.append(Path(entry.path))
    return children, skipped_count


def _scan_root(root: Path, excluded_names: frozenset[str]) -> tuple[list[str], int]:
    if not root.is_dir():
        raise CatalogError(f"年度フォルダが存在しません: {root}")

    try:
        stack, skipped_count = _child_directories(root, excluded_names)
    except OSError as exc:
        raise CatalogError(f"年度フォルダ直下を列挙できません: {root}") from exc

    folders: list[str] = []
    while stack:
        folder = stack.pop()
        folders.append(str(folder))
        try:
            children, skipped = _child_directories(folder, excluded_names)
        except OSError:
            skipped_count += 1
            continue
        skipped_count += skipped
        stack.extend(children)

    return folders, skipped_count


def scan_catalog(settings: Settings, scanned_at: datetime | None = None) -> CatalogUpdateResult:
    """Read directory names under the configured roots without enumerating files."""
    excluded_names = frozenset(settings.excluded_folder_names)
    current_folders, current_skipped = _scan_root(settings.current_year_root, excluded_names)
    previous_folders, previous_skipped = _scan_root(settings.previous_year_root, excluded_names)
    folders = current_folders + previous_folders
    if not folders:
        raise CatalogError("走査結果が0件のため、カタログを更新しませんでした。")

    folders.sort(key=str.casefold)
    timestamp = scanned_at or datetime.now(UTC)
    catalog = Catalog(
        scanned_at=timestamp.astimezone(UTC).isoformat(timespec="seconds"),
        folders=tuple(folders),
    )
    return CatalogUpdateResult(
        catalog=catalog,
        skipped_count=current_skipped + previous_skipped,
    )


def write_catalog_atomic(catalog: Catalog, catalog_path: str | Path | None = None) -> Path:
    """Replace catalog.json atomically while preserving the old file on failure."""
    target = Path(catalog_path) if catalog_path is not None else default_catalog_path()
    if not target.parent.is_dir():
        raise CatalogError(f"カタログ保存先フォルダが存在しません: {target.parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(catalog.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except OSError as exc:
        message = "catalog.jsonを更新できませんでした。既存カタログは維持します。"
        raise CatalogError(message) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            with suppress(OSError):
                temporary_path.unlink()
    return target


def _catalog_from_data(data: Any) -> Catalog:
    if not isinstance(data, dict) or frozenset(data) != CATALOG_KEYS:
        raise CatalogError("catalog.jsonの項目が不正です。")

    scanned_at = data["scanned_at"]
    folder_count = data["folder_count"]
    folders = data["folders"]
    if not isinstance(scanned_at, str) or not scanned_at:
        raise CatalogError("catalog.jsonの最終走査日時が不正です。")
    if type(folder_count) is not int or folder_count < 0:
        raise CatalogError("catalog.jsonの収録件数が不正です。")
    if not isinstance(folders, list) or any(
        not isinstance(folder, str) or not folder for folder in folders
    ):
        raise CatalogError("catalog.jsonのフォルダ一覧が不正です。")
    if folder_count != len(folders):
        raise CatalogError("catalog.jsonの収録件数とフォルダ一覧が一致しません。")
    return Catalog(scanned_at=scanned_at, folders=tuple(folders))


def load_catalog(catalog_path: str | Path | None = None) -> Catalog:
    """Load a previously completed catalog file."""
    path = Path(catalog_path) if catalog_path is not None else default_catalog_path()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise CatalogError(f"catalog.jsonが見つかりません: {path}") from exc
    except OSError as exc:
        raise CatalogError(f"catalog.jsonを読み取れません: {path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError("catalog.jsonのJSON形式が不正です。") from exc
    return _catalog_from_data(data)


def update_catalog(
    settings: Settings,
    catalog_path: str | Path | None = None,
    scanned_at: datetime | None = None,
) -> CatalogUpdateResult:
    """Scan both roots and atomically replace catalog.json after a complete scan."""
    result = scan_catalog(settings, scanned_at=scanned_at)
    write_catalog_atomic(result.catalog, catalog_path=catalog_path)
    return result
