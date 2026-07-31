import re
import unicodedata
from collections.abc import Iterable

_FILE_EXTENSION = re.compile(r"\.[a-z0-9]{1,10}(?![a-z0-9])", re.IGNORECASE)


def _character_kind(character: str) -> str:
    if character.isascii() and character.isalnum():
        return "ascii"
    if unicodedata.category(character)[0] in {"L", "M", "N"}:
        return "wide"
    return "separator"


def _split_normalized_text(text: str, *, remove_extensions: bool) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if remove_extensions:
        normalized = _FILE_EXTENSION.sub(" ", normalized)

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
