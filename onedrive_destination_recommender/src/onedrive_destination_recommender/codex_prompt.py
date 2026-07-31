import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from onedrive_destination_recommender.msg_reader import is_msg_file
from onedrive_destination_recommender.settings import Settings

__all__ = [
    "CodexConsultation",
    "build_codex_consultation",
    "copy_codex_consultation",
]


class _ClipboardTarget(Protocol):
    def clipboard_clear(self) -> None: ...

    def clipboard_append(self, text: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CodexConsultation:
    """Attachment guidance and a fixed consultation prompt kept in memory."""

    attachment_guidance: str
    prompt: str

    @property
    def clipboard_text(self) -> str:
        return f"{self.attachment_guidance}\n\n{self.prompt}"


def _absolute_existing_files(input_files: Iterable[str | os.PathLike[str]]) -> tuple[str, ...]:
    paths: list[str] = []
    for value in input_files:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path.resolve()))
    return tuple(paths)


def _attachment_guidance(input_paths: tuple[str, ...]) -> str:
    heading = "モードAの候補に納得できない場合は、次の案内に従ってCodexへ相談してください。"
    if not input_paths:
        return f"{heading}\n添付ファイルはありません（手動検索のみ）。"

    if len(input_paths) == 1 and is_msg_file(input_paths[0]):
        label = "元のMSGファイルを添付してください。"
    elif len(input_paths) == 1:
        label = "入力ファイルを添付してください。"
    else:
        label = "入力した全ファイルを添付してください。"
    items = "\n".join(f"- {path}" for path in input_paths)
    return f"{heading}\n{label}\n{items}"


def _path_list(paths: tuple[str, ...], *, empty_text: str) -> str:
    if not paths:
        return empty_text
    return "\n".join(f"{index}. {path}" for index, path in enumerate(paths, start=1))


def build_codex_consultation(
    *,
    input_files: Iterable[str | os.PathLike[str]],
    settings: Settings,
    search_text: str,
    candidate_paths: Iterable[str | os.PathLike[str]],
) -> CodexConsultation:
    """Generate guidance and a fixed prompt without saving or sending either one."""
    input_paths = _absolute_existing_files(input_files)
    candidates = tuple(os.path.abspath(path) for path in candidate_paths)[:10]
    input_section = _path_list(input_paths, empty_text="なし（手動検索のみ）")
    candidate_section = _path_list(candidates, empty_text="該当候補なし")
    prompt = f"""OneDrive業務フォルダの保存先を再検討してください。

【入力ファイルの絶対パス】
{input_section}

【対象年度フォルダ】
今年度: {os.path.abspath(settings.current_year_root)}
昨年度: {os.path.abspath(settings.previous_year_root)}

【現在の検索語】
{search_text}

【モードAの既存候補（上位10件）】
{candidate_section}

【依頼】
- 添付したファイルまたはMSGの内容を確認し、保存先を再検討してください。
- 適切な保存先があれば、提示された既存パスから選んでください。
- 適切な既存パスがなければ、既存の親パスと新規フォルダ名を提案してください。
- 判断材料が不足する場合は「保存先未定」としてください。
- ファイルの移動やフォルダの作成は行わず、提案だけを返してください。"""
    return CodexConsultation(
        attachment_guidance=_attachment_guidance(input_paths),
        prompt=prompt,
    )


def copy_codex_consultation(
    consultation: CodexConsultation,
    clipboard: _ClipboardTarget,
) -> None:
    """Copy guidance and prompt only after an explicit UI action."""
    clipboard.clipboard_clear()
    clipboard.clipboard_append(consultation.clipboard_text)
