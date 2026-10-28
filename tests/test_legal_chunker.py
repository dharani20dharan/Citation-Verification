"""Unit tests for LegalChunker, verifying exact structural identifier preservation."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.chunking.legal_chunker import LegalChunker
from src.chunking.base_chunker import normalize_source_identifier
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
SAMPLE_PDF = FIXTURES_DIR / "sample_fixture.pdf"


@pytest.fixture(scope="module")
def parsed_legal_doc():
    """Parse the real sample PDF fixture."""
    parser = DoclingDocumentParser()
    return parser.parse(file_path=SAMPLE_PDF, domain=Domain.LEGAL)


@pytest.fixture
def chunker():
    """Default LegalChunker instance."""
    return LegalChunker(max_characters=500, min_characters=20, overlap_characters=50)


def test_legal_document_chunking(chunker, parsed_legal_doc):
    """Test converting a parsed legal document into valid EvidenceChunks."""
    chunks = chunker.chunk_document(parsed_legal_doc)
    assert len(chunks) > 0

    for c in chunks:
        assert isinstance(c, EvidenceChunk)
        assert c.domain == Domain.LEGAL
        assert c.document_id == parsed_legal_doc.document_id
        assert c.page >= 1
        assert len(c.text.strip()) > 0


def test_no_absolute_paths_in_source_metadata(chunker, parsed_legal_doc):
    """Test that EvidenceChunk.source contains no absolute local developer paths."""
    chunks = chunker.chunk_document(parsed_legal_doc)
    assert len(chunks) > 0

    for c in chunks:
        # Must not contain Windows drive letters like C:\ or Users
        assert not c.source.startswith("C:"), f"Absolute path found in source: {c.source}"
        assert "\\" not in c.source, f"Unnormalized backslash in source: {c.source}"
        assert not c.source.startswith("/Users/"), f"Absolute Unix path found: {c.source}"
        # Source must be relative identifier like 'legal/sample_fixture.pdf'
        assert c.source.endswith(".pdf")


def test_chunk_ids_unique(chunker, parsed_legal_doc):
    """Test that all legal chunk_ids are unique."""
    chunks = chunker.chunk_document(parsed_legal_doc)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in legal corpus!"


def test_chunk_ids_deterministic(chunker, parsed_legal_doc):
    """Test that chunking the same legal document twice produces identical chunk IDs and content."""
    run1 = chunker.chunk_document(parsed_legal_doc)
    run2 = chunker.chunk_document(parsed_legal_doc)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.section == c2.section
        assert c1.section_id == c2.section_id
        assert c1.parent_id == c2.parent_id
        assert c1.content_type == c2.content_type
        assert c1.clause_id == c2.clause_id


def test_parent_child_relationships(chunker, parsed_legal_doc):
    """Test that legal child chunks link correctly to existing parent sections."""
    chunks = chunker.chunk_document(parsed_legal_doc)

    parent_chunks = {c.chunk_id: c for c in chunks if c.parent_id is None}
    child_chunks = [c for c in chunks if c.parent_id is not None]

    assert len(parent_chunks) >= 1
    assert len(child_chunks) >= 1

    for child in child_chunks:
        assert child.parent_id in parent_chunks, (
            f"Child {child.chunk_id} references missing parent {child.parent_id}"
        )


def test_exact_section_3d_vs_3e_distinction(chunker):
    """CRITICAL TEST: Verify Section 3(d) remains distinct from Section 3(e) and 3(d)(i) from 3(d)(ii)."""
    # Create structured legal document containing Section 3 with clause (d) and clause (e)
    statute_doc = DocumentParsingResult(
        document_id="legal_patents_act_1970",
        domain=Domain.LEGAL,
        source_path="data/raw/legal/patents_act_1970.pdf",
        title="The Patents Act, 1970",
        pages=14,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Section 3. What are not inventions",
                page=14,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="paragraph",
                text="(d) the mere discovery of a new form of a known substance which does not result in the enhancement of the known efficacy of that substance...",
                page=14,
            ),
            ParsedElement(
                element_id="elem_03",
                element_type="paragraph",
                text="(i) salts, esters, ethers, polymorphs, metabolites, pure form, particle size, isomers...",
                page=14,
            ),
            ParsedElement(
                element_id="elem_04",
                element_type="paragraph",
                text="(ii) combinations and other derivatives of known substance shall be considered to be the same substance...",
                page=14,
            ),
            ParsedElement(
                element_id="elem_05",
                element_type="paragraph",
                text="(e) a substance obtained by a mere admixture resulting only in the aggregation of the properties of the components thereof...",
                page=14,
            ),
        ],
        metadata={"statute_name": "The Patents Act, 1970"},
    )

    chunks = chunker.chunk_document(statute_doc)

    # 1. Locate the parent Section 3 chunk
    sec3_parents = [c for c in chunks if c.section_id == "sec_3" and c.parent_id is None]
    assert len(sec3_parents) == 1, "Section 3 parent chunk not found"
    sec3_chunk_id = sec3_parents[0].chunk_id

    # 2. Locate clause 3(d) chunks and clause 3(e) chunks
    clause_3d = [c for c in chunks if c.clause_id == "3(d)"]
    clause_3e = [c for c in chunks if c.clause_id == "3(e)"]

    assert len(clause_3d) == 1, "Clause 3(d) chunk not found"
    assert len(clause_3e) == 1, "Clause 3(e) chunk not found"

    chunk_3d = clause_3d[0]
    chunk_3e = clause_3e[0]

    # Verify 3(d) and 3(e) are completely distinct
    assert chunk_3d.chunk_id != chunk_3e.chunk_id
    assert chunk_3d.clause_id != chunk_3e.clause_id
    assert chunk_3d.section_id != chunk_3e.section_id
    assert "known efficacy" in chunk_3d.text
    assert "mere admixture" in chunk_3e.text

    # Both must trace back to the same parent section (Section 3)
    assert chunk_3d.parent_id == sec3_chunk_id
    assert chunk_3e.parent_id == sec3_chunk_id

    # 3. Locate subclause 3(d)(i) and 3(d)(ii)
    subclause_3di = [c for c in chunks if c.clause_id == "3(d)(i)"]
    subclause_3dii = [c for c in chunks if c.clause_id == "3(d)(ii)"]

    assert len(subclause_3di) == 1, "Subclause 3(d)(i) not found"
    assert len(subclause_3dii) == 1, "Subclause 3(d)(ii) not found"
    assert subclause_3di[0].chunk_id != subclause_3dii[0].chunk_id
    assert subclause_3di[0].section_id != subclause_3dii[0].section_id


def test_table_preservation_in_legal_document(chunker):
    """Test that tables in legal documents are preserved as table chunks."""
    mock_legal_doc = DocumentParsingResult(
        document_id="legal_schedule",
        domain=Domain.LEGAL,
        source_path="data/raw/legal/schedule.pdf",
        title="First Schedule: Patent Fees",
        pages=2,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="The First Schedule - Fees",
                page=1,
                level=1,
            ),
            ParsedElement(
                element_id="elem_02",
                element_type="table",
                text="| Number of Entry | On what payable | Amount of fee |\n|---|---|---|\n| 1 | On application for patent | Rs. 1600 |",
                page=1,
                metadata={"table_id": "fees_table_01"},
            ),
        ],
    )
    chunks = chunker.chunk_document(mock_legal_doc)
    table_chunks = [c for c in chunks if c.content_type == ContentType.TABLE]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_id == "fees_table_01"
    assert "Rs. 1600" in table_chunks[0].text
    assert table_chunks[0].parent_id is not None


def test_empty_legal_chunks_rejected(chunker):
    """Test that whitespace-only and micro elements do not produce empty chunks."""
    mock_legal_doc = DocumentParsingResult(
        document_id="legal_empty_test",
        domain=Domain.LEGAL,
        source_path="data/raw/legal/test.pdf",
        title="Empty Test",
        pages=1,
        elements=[
            ParsedElement(
                element_id="elem_01",
                element_type="heading",
                text="Section 1. Short title",
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
                text="abc",  # < min_characters (20)
                page=1,
            ),
        ],
    )
    chunks = chunker.chunk_document(mock_legal_doc)
    # Only the Section heading should be preserved
    assert len(chunks) == 1
    assert chunks[0].section == "Section 1"
