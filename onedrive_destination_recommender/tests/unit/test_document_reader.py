import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from onedrive_destination_recommender import document_reader

_DOCX_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>定例 サンプル</w:t></w:r></w:p>
    <w:p><w:r><w:t>設備検討会</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

_PPTX_SLIDE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree><p:sp><p:txBody>
    <a:p><a:r><a:t>月報 資料</a:t></a:r></a:p>
    <a:p><a:r><a:t>巻き取りカメラ</a:t></a:r></a:p>
  </p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>
"""


def _write_ooxml(path: Path, part_name: str, part_xml: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(part_name, part_xml)
        archive.writestr("docProps/app.xml", "<Properties><T>無関係</T></Properties>")
    return path


def _write_docx(path: Path) -> Path:
    return _write_ooxml(path, "word/document.xml", _DOCX_DOCUMENT_XML)


def _write_pptx(path: Path) -> Path:
    return _write_ooxml(path, "ppt/slides/slide1.xml", _PPTX_SLIDE_XML)


def _write_xlsx(path: Path, sheet_title: str, values: list[str]) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_title
    for row_index, value in enumerate(values, start=1):
        worksheet.cell(row=row_index, column=1, value=value)
    worksheet.cell(row=1, column=2, value=1234)
    workbook.save(path)
    return path


def _write_pdf(path: Path, ascii_text: str) -> Path:
    """Write a minimal single-page PDF whose text layer holds ascii_text."""
    stream = f"BT /F1 24 Tf 72 720 Td ({ascii_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]

    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, payload)

    xref_offset = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        body += b"%010d 00000 n \n" % offset
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    path.write_bytes(body)
    return path


def _write_empty_pdf(path: Path) -> Path:
    """Write a valid PDF whose single page carries no text layer."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]

    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, payload)

    xref_offset = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        body += b"%010d 00000 n \n" % offset
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    path.write_bytes(body)
    return path


def test_importing_document_reader_does_not_load_optional_dependencies() -> None:
    code = (
        "import sys; "
        "import onedrive_destination_recommender.document_reader; "
        "assert not any(name == 'openpyxl' or name.startswith('openpyxl.') "
        "or name == 'pypdf' or name.startswith('pypdf.') "
        "for name in sys.modules)"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_api_does_not_expose_raw_document_content() -> None:
    assert "is_supported_document" in document_reader.__all__
    assert "supported_document_extensions" in document_reader.__all__
    for hidden in ("DocumentContent", "read_document_text", "_read_document_text"):
        assert hidden not in document_reader.__all__


def test_document_reader_has_no_file_or_log_write_path() -> None:
    source = Path(document_reader.__file__).read_text(encoding="utf-8")

    assert "write_text(" not in source
    assert ".write(" not in source
    assert "open(" not in source
    assert "logging" not in source


@pytest.mark.parametrize("module_name", ["openpyxl", "pypdf"])
def test_optional_dependency_import_failure_becomes_document_error(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_modules: list[str] = []

    def unavailable_import(name: str):
        requested_modules.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(document_reader, "import_module", unavailable_import)
    loader = getattr(document_reader, f"_load_{module_name}")

    with pytest.raises(document_reader.DocumentUnavailableError):
        loader()

    assert requested_modules == [module_name]


@pytest.mark.parametrize(
    "file_name",
    ["book.xlsx", "macro.XLSM", "memo.docx", "deck.PPTX", "report.pdf"],
)
def test_supported_documents_are_classified_case_insensitively(file_name: str) -> None:
    assert document_reader.is_supported_document(file_name)


@pytest.mark.parametrize(
    "file_name",
    ["mail.msg", "legacy.xls", "legacy.doc", "legacy.ppt", "photo.png", "notes.txt", "noext"],
)
def test_unsupported_documents_are_rejected_by_extension(file_name: str) -> None:
    assert not document_reader.is_supported_document(file_name)


def test_supported_extensions_are_reported_without_touching_disk() -> None:
    assert document_reader.supported_document_extensions() == (
        ".docx",
        ".pdf",
        ".pptx",
        ".xlsm",
        ".xlsx",
    )


def test_excel_text_includes_sheet_title_and_string_cells(tmp_path: Path) -> None:
    workbook_path = _write_xlsx(tmp_path / "book.xlsx", "週報", ["秋田", "スリッター変更"])

    text = document_reader._read_document_text(workbook_path)

    assert "週報" in text
    assert "秋田" in text
    assert "スリッター変更" in text
    assert "1234" not in text


def test_macro_enabled_excel_is_read_with_the_same_path(tmp_path: Path) -> None:
    workbook_path = _write_xlsx(tmp_path / "macro.xlsm", "検討会", ["治具"])

    text = document_reader._read_document_text(workbook_path)

    assert "検討会" in text
    assert "治具" in text


def test_word_text_nodes_are_extracted(tmp_path: Path) -> None:
    document_path = _write_docx(tmp_path / "memo.docx")

    text = document_reader._read_document_text(document_path)

    assert "定例 サンプル" in text
    assert "設備検討会" in text
    assert "無関係" not in text


def test_powerpoint_slide_text_nodes_are_extracted(tmp_path: Path) -> None:
    deck_path = _write_pptx(tmp_path / "deck.pptx")

    text = document_reader._read_document_text(deck_path)

    assert "月報 資料" in text
    assert "巻き取りカメラ" in text
    assert "無関係" not in text


def test_pdf_text_layer_is_extracted(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pdf_path = _write_pdf(tmp_path / "report.pdf", "weekly report akita")

    text = document_reader._read_document_text(pdf_path)

    assert "weekly" in text.casefold()
    assert "akita" in text.casefold()


def test_pdf_without_text_layer_returns_empty_text_instead_of_failing(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pdf_path = _write_empty_pdf(tmp_path / "scanned.pdf")

    assert document_reader._read_document_text(pdf_path) == ""


def test_unsupported_extension_is_rejected_before_reading(tmp_path: Path) -> None:
    other_path = tmp_path / "legacy.xls"
    other_path.write_bytes(b"anything")

    with pytest.raises(ValueError, match="対応形式"):
        document_reader._read_document_text(other_path)


def test_missing_document_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        document_reader._read_document_text(tmp_path / "missing.docx")


@pytest.mark.parametrize("file_name", ["broken.xlsx", "broken.docx", "broken.pptx", "broken.pdf"])
def test_corrupt_documents_become_document_errors(file_name: str, tmp_path: Path) -> None:
    corrupt_path = tmp_path / file_name
    corrupt_path.write_bytes(b"not a real document")

    with pytest.raises(document_reader.DocumentUnavailableError):
        document_reader._read_document_text(corrupt_path)


def test_excel_cell_budget_stops_reading_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_path = _write_xlsx(tmp_path / "long.xlsx", "sheet", ["先頭語", "末尾語"])
    monkeypatch.setattr(document_reader, "_MAX_EXCEL_CELLS", 1)

    text = document_reader._read_document_text(workbook_path)

    assert "末尾語" not in text


def test_extracted_text_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document_path = _write_docx(tmp_path / "memo.docx")
    monkeypatch.setattr(document_reader, "_MAX_EXTRACTED_CHARACTERS", 4)

    assert len(document_reader._read_document_text(document_path)) <= 4


def test_part_declaring_a_dtd_is_skipped_without_parsing(tmp_path: Path) -> None:
    bomb_xml = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE w:document [<!ENTITY a "aaaaaaaaaa">'
        '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>"
    )
    document_path = _write_ooxml(tmp_path / "bomb.docx", "word/document.xml", bomb_xml)

    assert document_reader._read_document_text(document_path) == ""


def test_sheet_scan_stops_once_the_character_budget_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "先頭シート"
    workbook.create_sheet("末尾シート")
    workbook_path = tmp_path / "sheets.xlsx"
    workbook.save(workbook_path)
    monkeypatch.setattr(document_reader, "_MAX_EXTRACTED_CHARACTERS", 5)

    text = document_reader._read_document_text(workbook_path)

    assert "末尾シート" not in text


def test_oversized_package_part_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = _write_docx(tmp_path / "memo.docx")
    monkeypatch.setattr(document_reader, "_MAX_OOXML_PART_BYTES", 1)

    assert document_reader._read_document_text(document_path) == ""
