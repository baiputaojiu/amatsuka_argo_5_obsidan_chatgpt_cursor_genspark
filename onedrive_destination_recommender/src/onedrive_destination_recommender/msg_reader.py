from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from onedrive_destination_recommender.terms import (
    clean_msg_body,
    normalize_terms,
    searchable_attachment_names,
)

__all__ = [
    "MsgAccessProbe",
    "MsgSearchTerms",
    "OutlookUnavailableError",
    "build_msg_search_terms",
    "is_msg_file",
    "probe_msg_access",
]


class OutlookUnavailableError(RuntimeError):
    """Raised when Outlook COM cannot read a local MSG file."""


@dataclass(frozen=True, slots=True)
class MsgAccessProbe:
    """Non-sensitive result of an Outlook COM connectivity probe."""

    subject_accessible: bool
    body_accessible: bool
    attachment_count: int


@dataclass(frozen=True, slots=True)
class _MsgContent:
    """MSG fields kept in memory only for search-term generation."""

    subject: str
    body: str
    attachment_file_names: tuple[str, ...]
    attachment_count: int
    subject_available: bool
    body_available: bool
    attachments_available: bool


@dataclass(frozen=True, slots=True)
class MsgSearchTerms:
    """Normalized MSG search terms without retaining the message body."""

    primary_terms: tuple[str, ...]
    auxiliary_terms: tuple[str, ...]
    fully_parsed: bool
    body_available: bool
    warning: str | None


def _load_win32com_client() -> ModuleType:
    """Load pywin32 only when MSG access is explicitly requested."""
    try:
        return import_module("win32com.client")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OutlookUnavailableError(
            "pywin32を利用できないため、MSG本文を読み取れませんでした。"
        ) from exc


def is_msg_file(path: str | Path) -> bool:
    """Classify an input as MSG using only its extension."""
    return Path(path).suffix.casefold() == ".msg"


def _validated_msg_path(msg_path: str | Path) -> Path:
    path = Path(msg_path)
    if not is_msg_file(path):
        raise ValueError("MSGファイルを指定してください。")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@contextmanager
def _opened_msg_item(msg_path: str | Path) -> Iterator[Any]:
    path = _validated_msg_path(msg_path)
    win32com_client = _load_win32com_client()
    outlook: Any | None = None
    namespace: Any | None = None
    item: Any | None = None

    try:
        outlook = win32com_client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        item = namespace.OpenSharedItem(str(path.resolve()))
    except Exception as exc:
        raise OutlookUnavailableError(
            "Outlook COMでMSGを読み取れませんでした。MSGファイル名だけで処理を続けます。"
        ) from exc

    try:
        yield item
    finally:
        if item is not None:
            with suppress(Exception):
                item.Close(1)
        item = None
        namespace = None
        outlook = None


def _read_text_attribute(item: Any, name: str) -> tuple[str, bool]:
    try:
        value = getattr(item, name)
    except Exception:
        return "", False
    if value is None:
        return "", False
    return str(value), True


def _read_attachment_names(item: Any) -> tuple[tuple[str, ...], int, bool]:
    try:
        attachments = item.Attachments
        count = int(attachments.Count)
    except Exception:
        return (), 0, False

    names: list[str] = []
    complete = True
    for index in range(1, count + 1):
        try:
            attachment = attachments.Item(index)
            file_name = attachment.FileName
        except Exception:
            complete = False
            continue
        if file_name:
            names.append(str(file_name))
    return tuple(names), count, complete


def _read_msg_content(msg_path: str | Path) -> _MsgContent:
    """Read MSG fields through Outlook COM without saving attachments or message text."""
    with _opened_msg_item(msg_path) as item:
        subject, subject_available = _read_text_attribute(item, "Subject")
        body, body_available = _read_text_attribute(item, "Body")
        attachment_names, attachment_count, attachments_available = _read_attachment_names(item)
        return _MsgContent(
            subject=subject,
            body=body,
            attachment_file_names=attachment_names,
            attachment_count=attachment_count,
            subject_available=subject_available,
            body_available=body_available,
            attachments_available=attachments_available,
        )


def probe_msg_access(msg_path: str | Path) -> MsgAccessProbe:
    """Confirm read-only access to MSG fields without returning their contents."""
    content = _read_msg_content(msg_path)
    return MsgAccessProbe(
        subject_accessible=content.subject_available,
        body_accessible=content.body_available,
        attachment_count=content.attachment_count,
    )


def build_msg_search_terms(msg_path: str | Path) -> MsgSearchTerms:
    """Build primary and auxiliary terms, falling back to the MSG file name."""
    path = _validated_msg_path(msg_path)
    fallback_primary = normalize_terms(path.name)
    try:
        content = _read_msg_content(path)
    except OutlookUnavailableError:
        return MsgSearchTerms(
            primary_terms=fallback_primary,
            auxiliary_terms=(),
            fully_parsed=False,
            body_available=False,
            warning="メール本文を利用できませんでした。",
        )

    attachment_names = searchable_attachment_names(content.attachment_file_names)
    primary_sources = [path.name]
    if content.subject:
        primary_sources.append(content.subject)
    primary_sources.extend(attachment_names)
    auxiliary_terms = (
        normalize_terms(clean_msg_body(content.body), auxiliary=True)
        if content.body_available
        else ()
    )
    fully_parsed = (
        content.subject_available and content.body_available and content.attachments_available
    )
    warning = None
    if not content.body_available:
        warning = "メール本文を利用できませんでした。"
    elif not fully_parsed:
        warning = "メールの一部を利用できませんでした。"
    return MsgSearchTerms(
        primary_terms=normalize_terms(" ".join(primary_sources)),
        auxiliary_terms=auxiliary_terms,
        fully_parsed=fully_parsed,
        body_available=content.body_available,
        warning=warning,
    )
