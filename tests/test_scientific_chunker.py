"""Unit tests for ScientificChunker."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.chunking.scientific_chunker import ScientificChunker
from src.chunking.base_chunker import split_into_sentences, pack_sentences
from src.models.schemas import (
    ContentType,
    Domain,
    DocumentParsingResult,
    EvidenceChunk,
    ParsedElement,
    ParsedTable,
)
from src.parsing.docling_parser import DoclingDocumentParser

FIXTURES_DIR = Path(__file__).resolve().parent
SAMPLE_DOCX = FIXTURES_DIR / "sample_fixture.docx"


@pytest.fixture(scope="module")
def parsed_scientific_doc():
    """Parse the real sample DOCX fixture."""
    parser = DoclingDocumentParser()
    return parser.parse(file_path=SAMPLE_DOCX, domain=Domain.SCIENTIFIC)


@pytest.fixture
def chunker():
    """Default ScientificChunker instance."""
    return ScientificChunker(max_characters=500, min_characters=20, overlap_characters=50)


def test_scientific_document_chunking(chunker, parsed_scientific_doc):
    """Test that a scientific document is converted into valid EvidenceChunks."""
    chunks = chunker.chunk_document(parsed_scientific_doc)

    assert len(chunks) > 0
    # Every chunk must be a valid EvidenceChunk instance
    for c in chunks:
        assert isinstance(c, EvidenceChunk)
        assert c.domain == Domain.SCIENTIFIC
        assert c.document_id == parsed_scientific_doc.document_id
        assert c.page >= 1
        assert len(c.text.strip()) > 0


def test_chunk_ids_unique(chunker, parsed_scientific_doc):
    """Test that all generated chunk_ids are strictly unique."""
    chunks = chunker.chunk_document(parsed_scientific_doc)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"


def test_chunk_ids_deterministic(chunker, parsed_scientific_doc):
    """Test that running chunker twice on same input produces identical IDs and chunks."""
    run1 = chunker.chunk_document(parsed_scientific_doc)
    run2 = chunker.chunk_document(parsed_scientific_doc)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.section == c2.section
        assert c1.section_id == c2.section_id
        assert c1.parent_id == c2.parent_id
        assert c1.content_type == c2.content_type


def test_parent_child_relationships(chunker, parsed_scientific_doc):
    """Test that child chunks retain parent_id matching parent section chunks."""
    chunks = chunker.chunk_document(parsed_scientific_doc)

    parent_chunks = {c.chunk_id: c for c in chunks if c.parent_id is None}
    child_chunks = [c for c in chunks if c.parent_id is not None]

    assert len(parent_chunks) >= 2, "Expected at least 2 parent sections"
    assert len(child_chunks) >= 1, "Expected child chunks"

    for child in child_chunks:
        assert child.parent_id in parent_chunks, (
            f"Child {child.chunk_id} references non-existent parent {child.parent_id}"
        )
        parent = parent_chunks[child.parent_id]
        # Child section name must match parent section name
        assert child.section == parent.section


def test_table_preservation_as_table_chunks(chunker, parsed_scientific_doc):
    """Test that tables in parsed document are preserved as table EvidenceChunks."""
    chunks = chunker.chunk_document(parsed_scientific_doc)

    table_chunks = [c for c in chunks if c.content_type == ContentType.TABLE]
    assert len(table_chunks) >= 1, "Expected at least one table chunk"

    tbl = table_chunks[0]
    assert "|" in tbl.text, "Markdown table structure not found in text"
    assert tbl.table_id is not None
    assert tbl.parent_id is not None


def test_figure_preservation(chunker):
    """Test that figures and captions are properly captured as figure EvidenceChunks."""
    mock_doc = DocumentParsingResult(
        document_id="scientific_paper_fig",
        domain=Domain.SCIENTIFIC,
        source_path="paper.pdf",
        title="Figure Test Paper",
        pages=1,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Results",
                page=1,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="figure",
                text="Figure 1: Accuracy comparison between models across benchmark datasets.",
                page=1,
                metadata={"figure_id": "Fig_1"},
            ),
        ],
    )
    chunks = chunker.chunk_document(mock_doc)
    fig_chunks = [c for c in chunks if c.content_type == ContentType.FIGURE]
    assert len(fig_chunks) == 1
    assert fig_chunks[0].figure_id == "Fig_1"
    assert "Figure 1: Accuracy" in fig_chunks[0].text
    assert fig_chunks[0].parent_id is not None


def test_oversized_paragraph_splitting(chunker):
    """Test that oversized paragraphs are split into coherent sentence chunks with overlap."""
    sentences = [
        f"This is sentence number {i} containing important scientific findings and experimental data."
        for i in range(1, 20)
    ]
    long_text = " ".join(sentences)

    mock_doc = DocumentParsingResult(
        document_id="scientific_paper_long",
        domain=Domain.SCIENTIFIC,
        source_path="paper.pdf",
        title="Long Paragraph Paper",
        pages=1,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Methodology",
                page=1,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="paragraph",
                text=long_text,
                page=1,
            ),
        ],
    )

    # Chunker configured with max_characters=300
    custom_chunker = ScientificChunker(max_characters=300, overlap_characters=50)
    chunks = custom_chunker.chunk_document(mock_doc)

    child_chunks = [c for c in chunks if c.parent_id is not None]
    assert len(child_chunks) > 1, "Long text should have been split into multiple chunks"

    for c in child_chunks:
        assert len(c.text) <= 350, f"Sub-chunk too large: {len(c.text)}"
        assert c.section == "Methodology"
        assert c.parent_id == "scientific_paper_long_section_001"


def test_small_paragraph_not_fragmented(chunker):
    """Test that a reasonably sized paragraph is not artificially fragmented."""
    normal_text = "This is a single coherent paragraph describing the research setup. It consists of three sentences. None of them should be separated."

    mock_doc = DocumentParsingResult(
        document_id="scientific_paper_normal",
        domain=Domain.SCIENTIFIC,
        source_path="paper.pdf",
        title="Normal Paper",
        pages=1,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Setup",
                page=1,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="paragraph",
                text=normal_text,
                page=1,
            ),
        ],
    )

    chunks = chunker.chunk_document(mock_doc)
    para_chunks = [c for c in chunks if c.content_type == ContentType.PARAGRAPH]
    assert len(para_chunks) == 1
    assert para_chunks[0].text == normal_text


def test_empty_chunk_rejected(chunker):
    """Test that whitespace-only or micro-elements are skipped and do not produce empty chunks."""
    mock_doc = DocumentParsingResult(
        document_id="scientific_paper_empty",
        domain=Domain.SCIENTIFIC,
        source_path="paper.pdf",
        title="Empty Test Paper",
        pages=1,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Results",
                page=1,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="paragraph",
                text="   ",
                page=1,
            ),
            ParsedElement(
                element_id="elem_03",
                element_type="paragraph",
                text="tiny",  # < min_characters (20)
                page=1,
            ),
        ],
    )

    chunks = chunker.chunk_document(mock_doc)
    # Only the heading parent section should be created
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.SECTION
