import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

_FILE_EXTENSION = re.compile(
    r"\.(?:[a-z][a-z0-9]{0,9}|7z)(?=$|[\s,;、，])",
    re.IGNORECASE,
)
_REPLY_HEADER = re.compile(
    r"^(from|sent|to|subject|差出人|送信日時|宛先|件名)\s*[:：]",
    re.IGNORECASE,
)
_SIGNATURE_SEPARATOR = re.compile(r"^(?:-{3,}|={3,}|_{3,})$")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_ADDRESS = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_NUMBER = re.compile(r"(?<!\d)\d{2,4}-\d{2,4}-\d{3,4}(?!\d)")
_MEETING_LINE_MARKERS = (
    "microsoft teams",
    "会議 id",
    "パスコード",
    "join the meeting",
)
_INLINE_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".emz", ".wmz"})
_INLINE_IMAGE_STEM = re.compile(r"image\d+", re.IGNORECASE)


def _character_kind(character: str) -> str:
    if character.isascii() and character.isalnum():
        return "ascii"
    if unicodedata.category(character)[0] in {"L", "M", "N"}:
        return "wide"
    return "separator"


def _split_normalized_text(text: str, *, remove_extensions: bool) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if remove_extensions:
        while (without_extension := _FILE_EXTENSION.sub("", normalized)) != normalized:
            normalized = without_extension

    pieces: list[str] = []
    current: list[str] = []
    previous_kind = "separator"
    for character in normalized:
        kind = _character_kind(character)
        if kind == "separator":
            if current:
                pieces.append("".join(current))
                current = []
            previous_kind = kind
            continue
        if current and {kind, previous_kind} == {"ascii", "wide"}:
            pieces.append("".join(current))
            current = []
        current.append(character)
        previous_kind = kind
    if current:
        pieces.append("".join(current))
    return pieces


def _distinct(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def normalize_terms(text: str, *, auxiliary: bool = False) -> tuple[str, ...]:
    """Normalize user or file-derived text into distinct search terms."""
    pieces = _split_normalized_text(text, remove_extensions=True)
    return _distinct(
        piece
        for piece in pieces
        if len(piece) > 1 and not (auxiliary and piece.isdecimal() and len(piece) < 4)
    )


def normalize_term_sequence(
    terms: Iterable[str],
    *,
    auxiliary: bool = False,
) -> tuple[str, ...]:
    """Normalize a sequence supplied by a caller of the ranking core."""
    return normalize_terms(" ".join(terms), auxiliary=auxiliary)


def normalize_path_for_matching(path: str) -> str:
    """Normalize a relative folder path while retaining short path fragments."""
    return " ".join(_split_normalized_text(path, remove_extensions=False))


def initial_terms_from_file_names(file_names: Iterable[str]) -> tuple[str, ...]:
    """Build primary terms for one or more file names without reading the files."""
    return normalize_terms(" ".join(file_names))


def _reply_history_start(lines: list[str]) -> int | None:
    for start, line in enumerate(lines):
        first_match = _REPLY_HEADER.match(line.strip())
        if first_match is None:
            continue
        labels = {first_match.group(1).casefold()}
        for nearby_line in lines[start + 1 : start + 6]:
            if match := _REPLY_HEADER.match(nearby_line.strip()):
                labels.add(match.group(1).casefold())
        if len(labels) >= 2:
            return start
    return None


def _is_meeting_line(line: str) -> bool:
    normalized = unicodedata.normalize("NFKC", line).casefold()
    return any(marker in normalized for marker in _MEETING_LINE_MARKERS)


def _without_contact_noise(line: str) -> str:
    without_contacts = _URL.sub(" ", line)
    without_contacts = _EMAIL_ADDRESS.sub(" ", without_contacts)
    return _PHONE_NUMBER.sub(" ", without_contacts)


def _validated_limit(limit: int) -> int:
    if limit <= 0:
        raise ValueError("limitには1以上の整数を指定してください。")
    return limit


def clean_document_text(text: str, *, limit: int = 2000) -> str:
    """Remove contact noise from document text without applying email-specific rules."""
    _validated_limit(limit)
    cleaned_lines = [_without_contact_noise(line).rstrip() for line in text.splitlines()]
    return "\n".join(cleaned_lines).strip()[:limit]


def clean_msg_body(body: str, *, limit: int = 2000) -> str:
    """Remove common reply, signature, quote, meeting, and contact noise in memory."""
    _validated_limit(limit)

    lines = body.splitlines()
    cutoff = len(lines)
    for index, line in enumerate(lines):
        if line.strip().casefold() == "-----original message-----":
            cutoff = index
            break
    if (reply_start := _reply_history_start(lines[:cutoff])) is not None:
        cutoff = reply_start

    cleaned_lines: list[str] = []
    for line in lines[:cutoff]:
        stripped = line.strip()
        if line.lstrip().startswith(">") or _is_meeting_line(line):
            continue
        if _SIGNATURE_SEPARATOR.fullmatch(stripped):
            break
        cleaned_lines.append(_without_contact_noise(line).rstrip())

    return "\n".join(cleaned_lines).strip()[:limit]


def is_inline_image_attachment(file_name: str) -> bool:
    """Return whether an attachment name matches the generated image-number pattern."""
    path = Path(file_name)
    if path.suffix.casefold() not in _INLINE_IMAGE_EXTENSIONS:
        return False
    normalized_stem = unicodedata.normalize("NFKC", path.stem)
    return _INLINE_IMAGE_STEM.fullmatch(normalized_stem) is not None


def searchable_attachment_names(file_names: Iterable[str]) -> tuple[str, ...]:
    """Exclude generated inline image names without reading attachment content."""
    return tuple(
        file_name
        for file_name in file_names
        if file_name and not is_inline_image_attachment(file_name)
    )
