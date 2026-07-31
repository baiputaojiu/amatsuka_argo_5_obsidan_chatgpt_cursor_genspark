from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any


class OutlookUnavailableError(RuntimeError):
    """Raised when Outlook COM cannot read a local MSG file."""


@dataclass(frozen=True)
class MsgAccessProbe:
    """Non-sensitive result of an Outlook COM connectivity probe."""

    subject_accessible: bool
    body_accessible: bool
    attachment_count: int


def _load_win32com_client() -> ModuleType:
    """Load pywin32 only when MSG access is explicitly requested."""
    try:
        return import_module("win32com.client")
    except (ImportError, ModuleNotFoundError) as exc:
        raise OutlookUnavailableError(
            "pywin32を利用できないため、MSG本文を読み取れませんでした。"
        ) from exc


def probe_msg_access(msg_path: str | Path) -> MsgAccessProbe:
    """Confirm read-only access to MSG fields without returning their contents."""
    path = Path(msg_path)
    if path.suffix.lower() != ".msg":
        raise ValueError("MSGファイルを指定してください。")
    if not path.is_file():
        raise FileNotFoundError(path)

    win32com_client = _load_win32com_client()
    outlook: Any | None = None
    namespace: Any | None = None
    item: Any | None = None

    try:
        outlook = win32com_client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        item = namespace.OpenSharedItem(str(path.resolve()))

        subject = getattr(item, "Subject", None)
        body = getattr(item, "Body", None)
        attachments = getattr(item, "Attachments", None)
        attachment_count = int(getattr(attachments, "Count", 0))

        return MsgAccessProbe(
            subject_accessible=subject is not None,
            body_accessible=body is not None,
            attachment_count=attachment_count,
        )
    except OutlookUnavailableError:
        raise
    except Exception as exc:
        raise OutlookUnavailableError(
            "Outlook COMでMSGを読み取れませんでした。MSGファイル名だけで処理を続けます。"
        ) from exc
    finally:
        item = None
        namespace = None
        outlook = None
