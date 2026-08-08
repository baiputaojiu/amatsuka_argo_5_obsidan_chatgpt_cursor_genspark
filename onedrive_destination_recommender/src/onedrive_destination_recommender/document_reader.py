import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

from onedrive_destination_recommender.terms import clean_document_text, normalize_terms

__all__ = [
    "DocumentSearchTerms",
    "DocumentUnavailableError",
    "build_document_terms",
    "is_supported_document",
    "supported_document_extensions",
]


class DocumentUnavailableError(RuntimeError):
    """Raised when a supported document's text cannot be read on this machine."""


@dataclass(frozen=True, slots=True)
class DocumentSearchTerms:
    """Normalized document terms without retaining any extracted text."""

    auxiliary_terms: tuple[str, ...]
    parsed_count: int
    target_count: int
    warning: str | None


_EXCEL_EXTENSIONS = frozenset({".xlsx", ".xlsm"})
_OOXML_TEXT_EXTENSIONS = frozenset({".docx", ".pptx"})
_PDF_EXTENSIONS = frozenset({".pdf"})
_SUPPORTED_EXTENSIONS = _EXCEL_EXTENSIONS | _OOXML_TEXT_EXTENSIONS | _PDF_EXTENSIONS

_MAX_EXTRACTED_CHARACTERS = 20_000
_MAX_EXCEL_CELLS = 50_000
_MAX_PDF_PAGES = 20
_MAX_OOXML_PARTS = 200
_MAX_OOXML_PART_BYTES = 20 * 1024 * 1024

# OOXML parts never declare a DTD, so refusing one avoids entity-expansion parsing entirely.
_DOCTYPE_DECLARATION = b"<!DOCTYPE"


def supported_document_extensions() -> tuple[str, ...]:
    """List the extensions whose contents may be read for auxiliary terms."""
    return tuple(sorted(_SUPPORTED_EXTENSIONS))


def is_supported_document(path: str | Path) -> bool:
    """Classify an input by extension only, without touching the file."""
    return Path(path).suffix.casefold() in _SUPPORTED_EXTENSIONS


def _load_optional_module(module_name: str) -> ModuleType:
    """Load a document dependency only when that format is actually requested."""
    try:
        return import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise DocumentUnavailableError(
            f"{module_name}を利用できないため、この形式の本文を読み取れませんでした。"
        ) from exc


def _load_openpyxl() -> ModuleType:
    return _load_optional_module("openpyxl")


def _load_pypdf() -> ModuleType:
    return _load_optional_module("pypdf")


def _validated_document_path(path: str | Path) -> Path:
    document_path = Path(path)
    if not is_supported_document(document_path):
        raise ValueError("対応形式のファイルを指定してください。")
    if not document_path.is_file():
        raise FileNotFoundError(document_path)
    return document_path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_ooxml_text_part(extension: str, part_name: str) -> bool:
    if extension == ".docx":
        return part_name == "word/document.xml"
    return part_name.startswith("ppt/slides/slide") and part_name.endswith(".xml")


def _extract_excel_text(path: Path) -> str:
    """Read sheet titles and string cells through a read-only workbook view."""
    openpyxl = _load_openpyxl()
    workbook = openpyxl.load_workbook(filename=path, read_only=True, data_only=True)
    collected: list[str] = []
    total = 0
    cells = 0
    try:
        for worksheet in workbook.worksheets:
            if total >= _MAX_EXTRACTED_CHARACTERS:
                break
            if title := str(worksheet.title).strip():
                collected.append(title)
                total += len(title) + 1
            for row in worksheet.iter_rows(values_only=True):
                for value in row:
                    cells += 1
                    if cells > _MAX_EXCEL_CELLS or total >= _MAX_EXTRACTED_CHARACTERS:
                        return " ".join(collected)
                    if isinstance(value, str) and (cleaned := value.strip()):
                        collected.append(cleaned)
                        total += len(cleaned) + 1
    finally:
        workbook.close()
    return " ".join(collected)


def _extract_ooxml_text(path: Path) -> str:
    """Read Word and PowerPoint text nodes straight from the package parts."""
    extension = path.suffix.casefold()
    collected: list[str] = []
    total = 0
    with zipfile.ZipFile(path) as archive:
        parts = sorted(
            (
                entry
                for entry in archive.infolist()
                if _is_ooxml_text_part(extension, entry.filename)
                and entry.file_size <= _MAX_OOXML_PART_BYTES
            ),
            key=lambda entry: entry.filename,
        )
        for entry in parts[:_MAX_OOXML_PARTS]:
            part_bytes = archive.read(entry.filename)
            if _DOCTYPE_DECLARATION in part_bytes:
                continue
            root = ElementTree.fromstring(part_bytes)
            for element in root.iter():
                if _local_name(element.tag) != "t" or not element.text:
                    continue
                if not (cleaned := element.text.strip()):
                    continue
                collected.append(cleaned)
                total += len(cleaned) + 1
                if total >= _MAX_EXTRACTED_CHARACTERS:
                    return " ".join(collected)
    return " ".join(collected)


def _extract_pdf_text(path: Path) -> str:
    """Read the PDF text layer only; scanned pages yield no text and are not OCRed."""
    pypdf = _load_pypdf()
    reader = pypdf.PdfReader(str(path))
    collected: list[str] = []
    total = 0
    for page in reader.pages[:_MAX_PDF_PAGES]:
        if not (cleaned := (page.extract_text() or "").strip()):
            continue
        collected.append(cleaned)
        total += len(cleaned) + 1
        if total >= _MAX_EXTRACTED_CHARACTERS:
            break
    return " ".join(collected)


def _read_document_text(path: str | Path) -> str:
    """Return in-memory text for one supported document, or raise a document error."""
    document_path = _validated_document_path(path)
    extension = document_path.suffix.casefold()
    if extension in _EXCEL_EXTENSIONS:
        extractor = _extract_excel_text
    elif extension in _OOXML_TEXT_EXTENSIONS:
        extractor = _extract_ooxml_text
    else:
        extractor = _extract_pdf_text

    try:
        return extractor(document_path)[:_MAX_EXTRACTED_CHARACTERS]
    except DocumentUnavailableError:
        raise
    except Exception as exc:
        raise DocumentUnavailableError(
            "ファイルの本文を読み取れませんでした。ファイル名だけで処理を続けます。"
        ) from exc


def build_document_terms(paths: Iterable[str | Path]) -> DocumentSearchTerms:
    """Build auxiliary terms from supported inputs, keeping no extracted text."""
    target_paths = [Path(path) for path in paths if is_supported_document(path)]
    if not target_paths:
        return DocumentSearchTerms(
            auxiliary_terms=(),
            parsed_count=0,
            target_count=0,
            warning=None,
        )

    cleaned_texts: list[str] = []
    for document_path in target_paths:
        try:
            text = _read_document_text(document_path)
        except (DocumentUnavailableError, FileNotFoundError, ValueError):
            continue
        if cleaned_text := clean_document_text(text):
            cleaned_texts.append(cleaned_text)

    parsed_count = len(cleaned_texts)
    if parsed_count == 0:
        warning = "選択したファイルの本文を利用できませんでした。"
    elif parsed_count < len(target_paths):
        warning = "一部のファイルの本文を利用できませんでした。"
    else:
        warning = None

    return DocumentSearchTerms(
        auxiliary_terms=normalize_terms("\n".join(cleaned_texts), auxiliary=True),
        parsed_count=parsed_count,
        target_count=len(target_paths),
        warning=warning,
    )
