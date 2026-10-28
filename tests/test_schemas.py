"""Tests for canonical EvidenceChunk and related schemas."""

import pytest
from pydantic import ValidationError
from src.models.schemas import (
    ContentType,
    Domain,
    EvidenceChunk,
    ProcessingReport,
)


def test_valid_scientific_evidence_chunk():
    """Test instantiating a valid scientific EvidenceChunk."""
    chunk = EvidenceChunk(
        chunk_id="scientific_paper123_sec_intro_001",
        document_id="scientific_paper123",
        domain=Domain.SCIENTIFIC,
        title="Attention Is All You Need",
        source="arxiv:1706.03762.pdf",
        page=1,
        section="Introduction",
        section_id="sec_intro",
        parent_id=None,
        content_type=ContentType.PARAGRAPH,
        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        doi="10.48550/arXiv.1706.03762",
    )

    assert chunk.chunk_id == "scientific_paper123_sec_intro_001"
    assert chunk.domain == Domain.SCIENTIFIC
    assert chunk.content_type == ContentType.PARAGRAPH
    assert chunk.page == 1
    assert chunk.metadata_version == "1.0"

    # Test Chroma metadata serialization
    meta = chunk.to_chroma_metadata()
    assert meta["chunk_id"] == chunk.chunk_id
    assert meta["domain"] == "scientific"
    assert meta["page"] == 1
    assert meta["authors"] == "Ashish Vaswani, Noam Shazeer"
    assert meta["parent_id"] == ""
    assert meta["year"] == 2017


def test_valid_legal_evidence_chunk():
    """Test instantiating a valid legal EvidenceChunk with exact structural identifiers."""
    chunk = EvidenceChunk(
        chunk_id="legal_patents_act_1970_sec_3_d_001",
        document_id="legal_patents_act_1970",
        domain=Domain.LEGAL,
        title="The Patents Act, 1970",
        source="patents_act_1970.pdf",
        page=14,
        section="What are not inventions",
        section_id="sec_3",
        parent_id="legal_patents_act_1970_sec_3_root",
        content_type=ContentType.CLAUSE,
        clause_id="clause_d",
        subsection_id="subsection_3",
        statute_name="The Patents Act, 1970",
        text="the mere discovery of a new form of a known substance which does not result in the enhancement of the known efficacy of that substance...",
    )

    assert chunk.chunk_id == "legal_patents_act_1970_sec_3_d_001"
    assert chunk.domain == Domain.LEGAL
    assert chunk.clause_id == "clause_d"
    assert chunk.parent_id == "legal_patents_act_1970_sec_3_root"

    meta = chunk.to_chroma_metadata()
    assert meta["clause_id"] == "clause_d"
    assert meta["parent_id"] == "legal_patents_act_1970_sec_3_root"


def test_invalid_evidence_chunk_empty_text():
    """Test that empty or whitespace-only text raises ValidationError."""
    with pytest.raises(ValidationError):
        EvidenceChunk(
            chunk_id="test_001",
            document_id="doc_001",
            domain=Domain.SCIENTIFIC,
            title="Test Document",
            source="test.pdf",
            page=1,
            section="Intro",
            section_id="sec_1",
            content_type=ContentType.PARAGRAPH,
            text="   ",
        )


def test_invalid_evidence_chunk_invalid_page():
    """Test that page < 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        EvidenceChunk(
            chunk_id="test_001",
            document_id="doc_001",
            domain=Domain.SCIENTIFIC,
            title="Test Document",
            source="test.pdf",
            page=0,
            section="Intro",
            section_id="sec_1",
            content_type=ContentType.PARAGRAPH,
            text="Valid text",
        )


def test_invalid_evidence_chunk_missing_required():
    """Test that missing required fields raise ValidationError."""
    with pytest.raises(ValidationError):
        # Missing domain and content_type
        EvidenceChunk(  # type: ignore
            chunk_id="test_001",
            document_id="doc_001",
            title="Test Document",
            source="test.pdf",
            page=1,
            section="Intro",
            section_id="sec_1",
            text="Valid text",
        )


def test_processing_report():
    """Test ProcessingReport initialization and serialization."""
    report = ProcessingReport(
        domain="scientific",
        documents_discovered=5,
        documents_successfully_parsed=4,
        documents_failed=1,
        pages_processed=42,
        chunks_generated=120,
        chunks_indexed=120,
        embedding_model="allenai/specter2_base",
        embedding_dimension=768,
        chroma_collection="scientific_corpus",
        processing_time_seconds=12.5,
    )
    assert report.documents_discovered == 5
    assert report.chunks_indexed == 120
    data = report.model_dump()
    assert data["domain"] == "scientific"
