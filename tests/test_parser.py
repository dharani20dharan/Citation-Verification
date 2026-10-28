"""Unit tests for DoclingDocumentParser."""

import json
from pathlib import Path
import pytest

from src.models.schemas import Domain, DocumentParsingResult
from src.parsing.docling_parser import (
    DoclingDocumentParser,
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedFormatError,
)

FIXTURES_DIR = Path(__file__).resolve().parent
SAMPLE_PDF = FIXTURES_DIR / "sample_fixture.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample_fixture.docx"


@pytest.fixture(scope="module")
def parser():
    """Instantiate a shared DoclingDocumentParser."""
    return DoclingDocumentParser()


def test_missing_file_raises_error(parser):
    """Test that non-existent document file raises DocumentNotFoundError."""
    missing_path = FIXTURES_DIR / "non_existent_file.pdf"
    with pytest.raises(DocumentNotFoundError):
        parser.parse(file_path=missing_path, domain=Domain.SCIENTIFIC)


def test_unsupported_format_raises_error(parser, tmp_path):
    """Test that unsupported file extensions raise UnsupportedFormatError."""
    fake_txt = tmp_path / "notes.txt"
    fake_txt.write_text("Plain text file")
    with pytest.raises(UnsupportedFormatError):
        parser.parse(file_path=fake_txt, domain=Domain.SCIENTIFIC)


def test_corrupt_file_raises_error(parser, tmp_path):
    """Test that corrupt/invalid document raises DocumentParsingError."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"")
    with pytest.raises(DocumentParsingError):
        parser.parse(file_path=corrupt_pdf, domain=Domain.LEGAL)


def test_document_id_determinism(parser):
    """Test that document_id is stable and deterministic across multiple calls."""
    doc_id_1 = parser.generate_document_id(Domain.LEGAL, SAMPLE_PDF)
    doc_id_2 = parser.generate_document_id(Domain.LEGAL, SAMPLE_PDF)
    assert doc_id_1 == doc_id_2
    assert doc_id_1.startswith("legal_")
    assert "sample_fixture" in doc_id_1


def test_valid_docx_parsing(parser, tmp_path):
    """Test parsing a valid DOCX document with headings, paragraphs, and tables."""
    assert SAMPLE_DOCX.exists(), "DOCX fixture missing"

    result = parser.parse(file_path=SAMPLE_DOCX, domain=Domain.SCIENTIFIC)

    assert isinstance(result, DocumentParsingResult)
    assert result.domain == Domain.SCIENTIFIC
    assert result.document_id == "scientific_sample_fixture"
    assert len(result.elements) > 0

    # Verify structural hierarchy & elements
    headings = [e for e in result.elements if e.element_type in {"heading", "title"}]
    assert len(headings) >= 2
    paragraphs = [e for e in result.elements if e.element_type == "paragraph"]
    assert len(paragraphs) >= 1

    # Verify table extraction
    assert len(result.tables) >= 1
    table = result.tables[0]
    assert table.num_rows > 0
    assert "|" in table.markdown

    # Verify JSON serialization and reloading
    saved_path = parser.save_parsed_result(result, output_dir=tmp_path)
    assert saved_path.exists()
    reloaded = parser.load_parsed_result(saved_path)
    assert reloaded.document_id == result.document_id
    assert len(reloaded.elements) == len(result.elements)
    assert len(reloaded.tables) == len(result.tables)


def test_valid_pdf_parsing(parser, tmp_path):
    """Test parsing a valid multi-page PDF document."""
    assert SAMPLE_PDF.exists(), "PDF fixture missing"

    result = parser.parse(file_path=SAMPLE_PDF, domain=Domain.LEGAL)

    assert isinstance(result, DocumentParsingResult)
    assert result.domain == Domain.LEGAL
    assert result.document_id == "legal_sample_fixture"
    assert result.pages >= 2

    # Verify page preservation
    pages_found = {e.page for e in result.elements if e.page is not None}
    assert 1 in pages_found
    assert 2 in pages_found

    # Verify structural elements
    headings = [e for e in result.elements if e.element_type in {"heading", "title"}]
    assert len(headings) >= 1

    # Verify child-parent linking: paragraphs following headings have parent_element_id set
    child_elements = [e for e in result.elements if e.parent_element_id is not None]
    assert len(child_elements) >= 1

    # Test saving output
    saved_path = parser.save_parsed_result(result, output_dir=tmp_path)
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pages"] >= 2
    assert "elements" in data
