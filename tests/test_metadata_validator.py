"""Unit and regression tests for corpus and metadata validation layer."""

import json
from pathlib import Path
import pytest

from src.metadata.validator import CorpusValidator, ValidationReport
from src.models.schemas import ContentType, Domain, EvidenceChunk


@pytest.fixture
def validator():
    return CorpusValidator(min_embeddable_characters=10)


@pytest.fixture
def sample_legal_chunk_3d():
    return EvidenceChunk(
        chunk_id="doc_statute_sec_3_d",
        document_id="doc_statute",
        domain=Domain.LEGAL,
        title="The Patents Act",
        source="legal/patents_act.pdf",
        page=14,
        section="Section 3(d)",
        section_id="sec_3_d",
        parent_id="doc_statute_sec_3",
        content_type=ContentType.CLAUSE,
        text="the mere discovery of a new form of a known substance...",
        clause_id="3(d)",
    )


@pytest.fixture
def sample_legal_chunk_3e():
    return EvidenceChunk(
        chunk_id="doc_statute_sec_3_e",
        document_id="doc_statute",
        domain=Domain.LEGAL,
        title="The Patents Act",
        source="legal/patents_act.pdf",
        page=14,
        section="Section 3(e)",
        section_id="sec_3_e",
        parent_id="doc_statute_sec_3",
        content_type=ContentType.CLAUSE,
        text="a substance obtained by a mere admixture resulting only in...",
        clause_id="3(e)",
    )


@pytest.fixture
def sample_legal_parent_sec3():
    return EvidenceChunk(
        chunk_id="doc_statute_sec_3",
        document_id="doc_statute",
        domain=Domain.LEGAL,
        title="The Patents Act",
        source="legal/patents_act.pdf",
        page=14,
        section="Section 3",
        section_id="sec_3",
        parent_id=None,
        content_type=ContentType.SECTION,
        text="What are not inventions within the meaning of this Act.",
    )


@pytest.fixture
def sample_scientific_parent():
    return EvidenceChunk(
        chunk_id="sci_doc_sec_01",
        document_id="sci_doc",
        domain=Domain.SCIENTIFIC,
        title="Citation Verification Analysis",
        source="scientific/analysis.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01_introduction",
        parent_id=None,
        content_type=ContentType.SECTION,
        text="1. Introduction and problem statement.",
    )


@pytest.fixture
def sample_scientific_child():
    return EvidenceChunk(
        chunk_id="sci_doc_sec_01_chunk_001",
        document_id="sci_doc",
        domain=Domain.SCIENTIFIC,
        title="Citation Verification Analysis",
        source="scientific/analysis.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01_introduction",
        parent_id="sci_doc_sec_01",
        content_type=ContentType.PARAGRAPH,
        text="This paper investigates citation integrity and evidence extraction.",
    )


# 1. Valid scientific corpus
def test_valid_scientific_corpus(validator, sample_scientific_parent, sample_scientific_child):
    chunks = [sample_scientific_parent, sample_scientific_child]
    report = validator.validate_chunks(chunks, domain=Domain.SCIENTIFIC)
    assert report.passed is True
    assert report.total_chunks == 2
    assert report.valid_chunks == 2
    assert report.embeddable_chunks == 2
    assert report.parent_references_valid is True
    assert report.scientific_metadata_valid is True
    assert len(report.errors) == 0


# 2. Valid legal corpus
def test_valid_legal_corpus(validator, sample_legal_parent_sec3, sample_legal_chunk_3d, sample_legal_chunk_3e):
    chunks = [sample_legal_parent_sec3, sample_legal_chunk_3d, sample_legal_chunk_3e]
    report = validator.validate_chunks(chunks, domain=Domain.LEGAL)
    assert report.passed is True
    assert report.total_chunks == 3
    assert report.valid_chunks == 3
    assert report.legal_identifiers_valid is True
    assert report.parent_references_valid is True
    assert len(report.errors) == 0


# 3. Duplicate chunk IDs
def test_duplicate_chunk_ids(validator, sample_scientific_parent, sample_scientific_child):
    # Duplicate ID
    duplicate_child = sample_scientific_child.model_copy(update={"chunk_id": sample_scientific_parent.chunk_id})
    chunks = [sample_scientific_parent, duplicate_child]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.chunk_ids_unique is False
    assert any("Duplicate chunk_id" in err for err in report.errors)


# 4. Missing required metadata
def test_missing_required_metadata(validator, sample_scientific_parent):
    # Empty title
    invalid_chunk = sample_scientific_parent.model_copy(update={"title": ""})
    chunks = [invalid_chunk]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.invalid_chunks == 1
    assert any("title is empty" in err for err in report.errors)


# 5. Empty text
def test_empty_text(validator, sample_scientific_parent):
    empty_text_chunk = sample_scientific_parent.model_copy(update={"text": "   "})
    chunks = [empty_text_chunk]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.embeddable_chunks == 0
    assert report.non_embeddable_chunks == 1
    assert any("empty or whitespace" in err for err in report.errors)


# 6. Absolute Windows source path
def test_absolute_windows_source_path(validator, sample_legal_parent_sec3):
    non_portable_chunk = sample_legal_parent_sec3.model_copy(
        update={"source": "C:\\Users\\DHARANIDHARAN\\Desktop\\Project\\doc.pdf"}
    )
    chunks = [non_portable_chunk]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.source_paths_portable is False
    assert any("Non-portable source path" in err for err in report.errors)


# 7. Missing parent
def test_missing_parent(validator, sample_scientific_child):
    # Child referencing non-existent parent
    orphan_chunk = sample_scientific_child.model_copy(update={"parent_id": "non_existent_parent_id"})
    chunks = [orphan_chunk]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.parent_references_valid is False
    assert any("Missing parent chunk" in err for err in report.errors)


# 8. Cross-document parent
def test_cross_document_parent(validator, sample_scientific_parent, sample_scientific_child):
    cross_doc_child = sample_scientific_child.model_copy(update={"document_id": "different_doc_id"})
    chunks = [sample_scientific_parent, cross_doc_child]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.parent_references_valid is False
    assert any("Cross-document parent reference" in err for err in report.errors)


# 9. Self-parent
def test_self_parent(validator, sample_scientific_parent):
    self_parent_chunk = sample_scientific_parent.model_copy(
        update={"parent_id": sample_scientific_parent.chunk_id}
    )
    chunks = [self_parent_chunk]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.parent_references_valid is False
    assert any("Self-referencing parent" in err for err in report.errors)


# 10. Circular parent relationship
def test_circular_parent_relationship(validator, sample_scientific_parent, sample_scientific_child):
    # A points to B, B points to A
    cyclic_parent = sample_scientific_parent.model_copy(update={"parent_id": sample_scientific_child.chunk_id})
    chunks = [cyclic_parent, sample_scientific_child]
    report = validator.validate_chunks(chunks)
    assert report.passed is False
    assert report.parent_references_valid is False
    assert any("Circular parent-child relationship" in err for err in report.errors)


# 11. 3(d) vs 3(e) conflation detection
def test_section_3d_vs_3e_conflation_detection(
    validator,
    sample_legal_parent_sec3,
    sample_legal_chunk_3d,
    sample_legal_chunk_3e,
):
    # First: distinct section_ids should pass
    chunks_valid = [sample_legal_parent_sec3, sample_legal_chunk_3d, sample_legal_chunk_3e]
    report_valid = validator.validate_chunks(chunks_valid)
    assert report_valid.passed is True

    # Second: collapsed identical section_ids should fail
    collapsed_3e = sample_legal_chunk_3e.model_copy(update={"section_id": "sec_3_d"})
    chunks_invalid = [sample_legal_parent_sec3, sample_legal_chunk_3d, collapsed_3e]
    report_invalid = validator.validate_chunks(chunks_invalid)
    assert report_invalid.passed is False
    assert report_invalid.legal_identifiers_valid is False
    assert any("Legal conflation" in err for err in report_invalid.errors)


# 12. 3(d)(i) vs 3(d)(ii) distinction
def test_subclause_3di_vs_3dii_conflation_detection(validator, sample_legal_parent_sec3, sample_legal_chunk_3d):
    subclause_3di = EvidenceChunk(
        chunk_id="doc_statute_sec_3_d_i",
        document_id="doc_statute",
        domain=Domain.LEGAL,
        title="The Patents Act",
        source="legal/patents_act.pdf",
        page=14,
        section="Section 3(d)(i)",
        section_id="sec_3_d_i",
        parent_id="doc_statute_sec_3",
        content_type=ContentType.CLAUSE,
        text="salts, esters, ethers, polymorphs, metabolites...",
        clause_id="3(d)(i)",
    )
    subclause_3dii = EvidenceChunk(
        chunk_id="doc_statute_sec_3_d_ii",
        document_id="doc_statute",
        domain=Domain.LEGAL,
        title="The Patents Act",
        source="legal/patents_act.pdf",
        page=14,
        section="Section 3(d)(ii)",
        section_id="sec_3_d_ii",
        parent_id="doc_statute_sec_3",
        content_type=ContentType.CLAUSE,
        text="combinations and other derivatives of known substance...",
        clause_id="3(d)(ii)",
    )

    # Valid: distinct section_ids
    valid_chunks = [sample_legal_parent_sec3, sample_legal_chunk_3d, subclause_3di, subclause_3dii]
    rep_valid = validator.validate_chunks(valid_chunks)
    assert rep_valid.passed is True

    # Invalid: conflated section_ids
    conflated_3dii = subclause_3dii.model_copy(update={"section_id": "sec_3_d_i"})
    invalid_chunks = [sample_legal_parent_sec3, sample_legal_chunk_3d, subclause_3di, conflated_3dii]
    rep_invalid = validator.validate_chunks(invalid_chunks)
    assert rep_invalid.passed is False
    assert any("Legal subclause conflation" in err for err in rep_invalid.errors)


# 13. Malformed JSONL
def test_malformed_jsonl(validator, tmp_path):
    bad_jsonl_file = tmp_path / "corrupted.jsonl"
    bad_content = '{"chunk_id": "good_01", "document_id": "doc1", "domain": "scientific", "title": "T", "source": "s.pdf", "page": 1, "section": "S", "section_id": "s1", "parent_id": null, "content_type": "section", "text": "Valid text here."}\n'
    bad_content += '{"chunk_id": "broken_02", INVALID_JSON_HERE\n'
    bad_jsonl_file.write_text(bad_content, encoding="utf-8")

    report = validator.validate_jsonl_file(bad_jsonl_file)
    assert report.passed is False
    assert report.invalid_chunks >= 1
    assert any("Malformed JSON" in err for err in report.errors)


# 14. Deterministic validation
def test_deterministic_validation(validator, sample_legal_parent_sec3, sample_legal_chunk_3d):
    chunks = [sample_legal_parent_sec3, sample_legal_chunk_3d]
    rep1 = validator.validate_chunks(chunks)
    rep2 = validator.validate_chunks(chunks)
    rep3 = validator.validate_chunks(chunks)

    assert rep1.passed == rep2.passed == rep3.passed
    assert rep1.total_chunks == rep2.total_chunks == rep3.total_chunks
    assert rep1.valid_chunks == rep2.valid_chunks == rep3.valid_chunks
    assert rep1.errors == rep2.errors == rep3.errors


# 15. Table metadata preservation
def test_table_metadata_preservation(validator, sample_scientific_parent):
    # Table chunk missing table_id
    bad_table_chunk = EvidenceChunk(
        chunk_id="sci_doc_table_01",
        document_id="sci_doc",
        domain=Domain.SCIENTIFIC,
        title="Citation Verification Analysis",
        source="scientific/analysis.pdf",
        page=1,
        section="2. Results",
        section_id="sec_02_results",
        parent_id="sci_doc_sec_01",
        content_type=ContentType.TABLE,
        text="| Col1 | Col2 |\n|---|---|\n| A | B |",
        table_id=None,  # Missing!
    )
    chunks = [sample_scientific_parent, bad_table_chunk]
    report = validator.validate_chunks(chunks, domain=Domain.SCIENTIFIC)
    assert report.passed is False
    assert report.scientific_metadata_valid is False
    assert any("missing table_id" in err for err in report.errors)


# 16. Scientific section/parent metadata preservation
def test_scientific_section_parent_metadata_preservation(
    validator,
    sample_scientific_parent,
    sample_scientific_child,
):
    chunks = [sample_scientific_parent, sample_scientific_child]
    report = validator.validate_chunks(chunks, domain=Domain.SCIENTIFIC)
    assert report.passed is True
    assert sample_scientific_child.parent_id == sample_scientific_parent.chunk_id
    assert sample_scientific_child.section == sample_scientific_parent.section
    assert sample_scientific_child.section_id == sample_scientific_parent.section_id
