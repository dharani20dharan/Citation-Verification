"""Unit and integration tests for Citation and Claim Extraction (Stage 8)."""

import pytest
from pydantic import ValidationError

from src.models.schemas import ContentType, Domain, EvidenceChunk
from src.extraction.citation_detector import CitationDetector
from src.extraction.claim_extractor import ClaimExtractor
from src.extraction.extractor import ClaimCitationExtractor
from src.extraction.models import CitationClaim, ReferenceType
from src.extraction.reference_normalizer import ReferenceNormalizer


@pytest.fixture
def normalizer():
    return ReferenceNormalizer()


@pytest.fixture
def detector():
    return CitationDetector()


@pytest.fixture
def claim_extractor():
    return ClaimExtractor()


@pytest.fixture
def extractor():
    return ClaimCitationExtractor()


# ==============================================================================
# 1-6. SCIENTIFIC CITATION TESTS
# ==============================================================================

def test_single_digit_numeric_citation(extractor):
    text = "The baseline model achieves strong empirical performance [1]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "[1]"
    assert claims[0].normalized_reference == "1"
    assert claims[0].normalized_references == ["1"]
    assert claims[0].reference_type == ReferenceType.NUMERIC.value
    assert claims[0].claim_text == "The baseline model achieves strong empirical performance"


def test_multi_digit_numeric_citation(extractor):
    text = "The proposed method reduces error by 40% [12]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "[12]"
    assert claims[0].normalized_reference == "12"
    assert claims[0].normalized_references == ["12"]
    assert claims[0].reference_type == ReferenceType.NUMERIC.value
    assert claims[0].claim_text == "The proposed method reduces error by 40%"


def test_comma_separated_numeric_citations(extractor):
    text = "Several recent architectures demonstrate improved convergence [3, 4]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "[3, 4]"
    assert claims[0].normalized_reference == "3, 4"
    assert claims[0].normalized_references == ["3", "4"]
    assert claims[0].reference_type == ReferenceType.NUMERIC.value
    assert claims[0].claim_text == "Several recent architectures demonstrate improved convergence"


def test_range_numeric_citations_with_unicode_dash(extractor):
    # Test both unicode en-dash and ascii hyphen
    text_unicode = "Previous investigations highlighted scaling bottlenecks [5\u20137]."
    claims_u = extractor.extract_from_text(
        text=text_unicode,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims_u) == 1
    assert claims_u[0].normalized_reference == "5-7"
    assert claims_u[0].normalized_references == ["5-7"]

    text_ascii = "Previous investigations highlighted scaling bottlenecks [5-7]."
    claims_a = extractor.extract_from_text(
        text=text_ascii,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims_a) == 1
    assert claims_a[0].normalized_reference == "5-7"
    assert claims_a[0].normalized_references == ["5-7"]


def test_multiple_citations_in_one_sentence(extractor):
    text = "The method is faster [12] and more accurate [13]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 2
    assert claims[0].normalized_reference == "12"
    assert claims[0].claim_text == "The method is faster"
    assert claims[1].normalized_reference == "13"
    assert claims[1].claim_text == "The method is more accurate"


def test_author_year_citations(extractor):
    text = "Deep learning architectures have transformed document analysis (Smith et al., 2024)."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "(Smith et al., 2024)"
    assert claims[0].normalized_reference == "Smith et al., 2024"
    assert claims[0].reference_type == ReferenceType.AUTHOR_YEAR.value
    assert claims[0].claim_text == "Deep learning architectures have transformed document analysis"

    text2 = "Standard benchmarks provide robust comparative evaluation (Smith, 2023)."
    claims2 = extractor.extract_from_text(
        text=text2,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims2) == 1
    assert claims2[0].normalized_reference == "Smith, 2023"


# ==============================================================================
# 7-12. LEGAL REFERENCE TESTS
# ==============================================================================

def test_legal_section_3d(extractor):
    text = "Under Section 3(d), the mere discovery of a new form of a known substance is not patentable."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Section 3(d)"
    assert claims[0].normalized_reference == "3(d)"
    assert claims[0].reference_type == ReferenceType.LEGAL_SECTION.value
    assert claims[0].claim_text == "the mere discovery of a new form of a known substance is not patentable"


def test_legal_section_3e(extractor):
    text = "Section 3(e) prohibits the patenting of a substance obtained by a mere admixture."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Section 3(e)"
    assert claims[0].normalized_reference == "3(e)"
    assert claims[0].reference_type == ReferenceType.LEGAL_SECTION.value
    assert claims[0].claim_text == "prohibits the patenting of a substance obtained by a mere admixture"


def test_legal_section_3di(extractor):
    text = "Pursuant to Section 3(d)(i), salts and esters of known substances are considered identical."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Section 3(d)(i)"
    assert claims[0].normalized_reference == "3(d)(i)"
    assert claims[0].claim_text == "salts and esters of known substances are considered identical"


def test_legal_section_3dii(extractor):
    text = "Under Section 3(d)(ii), polymorphs and metabolites require enhanced efficacy proof."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Section 3(d)(ii)"
    assert claims[0].normalized_reference == "3(d)(ii)"
    assert claims[0].claim_text == "polymorphs and metabolites require enhanced efficacy proof"


def test_legal_article_14_2(extractor):
    text = "Article 14(2) guarantees equality before the law."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="const_act",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Article 14(2)"
    assert claims[0].normalized_reference == "14(2)"
    assert claims[0].reference_type == ReferenceType.LEGAL_ARTICLE.value
    assert claims[0].claim_text == "guarantees equality before the law"


def test_legal_clause_7(extractor):
    text = "Under Clause 7, arbitration proceedings must commence within thirty days."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="contract_01",
    )
    assert len(claims) == 1
    assert claims[0].reference_text == "Clause 7"
    assert claims[0].normalized_reference == "7"
    assert claims[0].reference_type == ReferenceType.LEGAL_CLAUSE.value
    assert claims[0].claim_text == "arbitration proceedings must commence within thirty days"


# ==============================================================================
# 13-15. CRITICAL IDENTIFIER PRESERVATION TESTS
# ==============================================================================

def test_section_3d_differs_from_3e(normalizer):
    norm_3d, _ = normalizer.normalize("Section 3(d)", ReferenceType.LEGAL_SECTION.value)
    norm_3e, _ = normalizer.normalize("Section 3(e)", ReferenceType.LEGAL_SECTION.value)
    assert norm_3d == "3(d)"
    assert norm_3e == "3(e)"
    assert norm_3d != norm_3e


def test_section_3di_differs_from_3dii(normalizer):
    norm_3di, _ = normalizer.normalize("Section 3(d)(i)", ReferenceType.LEGAL_SECTION.value)
    norm_3dii, _ = normalizer.normalize("Section 3(d)(ii)", ReferenceType.LEGAL_SECTION.value)
    assert norm_3di == "3(d)(i)"
    assert norm_3dii == "3(d)(ii)"
    assert norm_3di != norm_3dii


def test_no_lossy_normalization(normalizer):
    # 3(d) must not become 3
    norm_3d, _ = normalizer.normalize("Section 3(d)", ReferenceType.LEGAL_SECTION.value)
    assert norm_3d != "3"
    assert norm_3d == "3(d)"

    # 3(d)(i) must not become 3(d)
    norm_3di, _ = normalizer.normalize("Section 3(d)(i)", ReferenceType.LEGAL_SECTION.value)
    assert norm_3di != "3(d)"
    assert norm_3di == "3(d)(i)"

    # Section sign symbol § 3(d)
    norm_symbol, _ = normalizer.normalize("§ 3(d)", ReferenceType.LEGAL_SECTION.value)
    assert norm_symbol == "3(d)"


# ==============================================================================
# 16-18. CLAIM EXTRACTION & SUBSTANTIVE WORDING TESTS
# ==============================================================================

def test_citation_removed_from_claim_text(extractor):
    text = "The proposed algorithm achieves lower computational overhead [42]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert "[42]" not in claims[0].claim_text
    assert claims[0].claim_text == "The proposed algorithm achieves lower computational overhead"


def test_substantive_wording_preserved(extractor):
    original_statement = "The synthesis of compound 4b exhibited superior thermodynamic stability"
    text = f"{original_statement} [19]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].claim_text == original_statement


def test_grouped_citations_single_claim(extractor):
    text = "Prior studies validate this theoretical foundation [12, 13]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    # One claim representing both cited sources
    assert len(claims) == 1
    assert claims[0].claim_text == "Prior studies validate this theoretical foundation"
    assert claims[0].normalized_reference == "12, 13"
    assert claims[0].normalized_references == ["12", "13"]


# ==============================================================================
# 19-22. PROVENANCE PRESERVATION TESTS
# ==============================================================================

def test_provenance_preservation_from_chunk(extractor):
    chunk = EvidenceChunk(
        chunk_id="paper_sec_02_chunk_005",
        document_id="scientific_paper_42",
        domain=Domain.SCIENTIFIC,
        title="Neural Citation Verification",
        source="scientific/paper_42.pdf",
        page=7,
        section="3. Empirical Results",
        section_id="sec_03_results",
        parent_id="paper_sec_02",
        content_type=ContentType.PARAGRAPH,
        text="The fine-tuned model outperforms baseline architectures [8].",
    )

    claims = extractor.extract_from_chunk(chunk)
    assert len(claims) == 1
    claim = claims[0]

    assert claim.document_id == "scientific_paper_42"
    assert claim.domain == Domain.SCIENTIFIC
    assert claim.source_chunk_id == "paper_sec_02_chunk_005"
    assert claim.page == 7
    assert claim.section == "3. Empirical Results"
    assert claim.section_id == "sec_03_results"
    assert claim.character_start is not None
    assert claim.character_end is not None


# ==============================================================================
# 23. DETERMINISM TESTS
# ==============================================================================

def test_deterministic_claim_ids(extractor):
    text = "Under Section 3(d), mere discovery is excluded. Section 3(e) excludes mere admixture."
    claims_run_1 = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    claims_run_2 = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="patents_act",
    )

    assert len(claims_run_1) == 2
    assert len(claims_run_2) == 2
    assert [c.claim_id for c in claims_run_1] == [c.claim_id for c in claims_run_2]
    assert [c.claim_text for c in claims_run_1] == [c.claim_text for c in claims_run_2]
    assert [c.normalized_reference for c in claims_run_1] == [c.normalized_reference for c in claims_run_2]


# ==============================================================================
# 24-25. VALIDATION TESTS
# ==============================================================================

def test_empty_claim_rejected():
    with pytest.raises(ValidationError):
        CitationClaim(
            claim_id="doc_claim_001",
            document_id="doc_01",
            domain=Domain.SCIENTIFIC,
            claim_text="   ",  # Invalid empty whitespace
            reference_text="[1]",
            normalized_reference="1",
            reference_type=ReferenceType.NUMERIC.value,
        )


def test_empty_reference_rejected():
    with pytest.raises(ValidationError):
        CitationClaim(
            claim_id="doc_claim_001",
            document_id="doc_01",
            domain=Domain.SCIENTIFIC,
            claim_text="Valid substantive statement",
            reference_text="   ",  # Invalid empty
            normalized_reference="1",
            reference_type=ReferenceType.NUMERIC.value,
        )

    with pytest.raises(ValidationError):
        CitationClaim(
            claim_id="doc_claim_001",
            document_id="doc_01",
            domain=Domain.SCIENTIFIC,
            claim_text="Valid substantive statement",
            reference_text="[1]",
            normalized_reference="",  # Invalid empty
            reference_type=ReferenceType.NUMERIC.value,
        )


# ==============================================================================
# 26-31. STAGE 8.1 QUALITY FIX REGRESSION TESTS
# ==============================================================================

def test_heading_with_section_3d_produces_no_claim(extractor):
    """Verify that headings containing citations are not treated as substantive claims."""
    heading_chunk = EvidenceChunk(
        chunk_id="legal_sec_3_d_heading",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="Patents Act 1970",
        source="legal/patents.pdf",
        page=1,
        section="Section 3(d)",
        section_id="sec_3_d",
        parent_id=None,
        content_type=ContentType.HEADING,
        text="1. Section 3(d) Provisions",
    )
    claims_from_chunk = extractor.extract_from_chunk(heading_chunk)
    assert len(claims_from_chunk) == 0

    # Also test directly via extract_from_text
    claims_from_text = extractor.extract_from_text(
        text="1. Section 3(d) Provisions",
        domain=Domain.LEGAL,
        document_id="patents_act",
    )
    assert len(claims_from_text) == 0


def test_legal_paragraph_with_section_3d_produces_claim(extractor):
    """Verify that substantive paragraphs containing Section 3(d) continue to be extracted."""
    para_chunk = EvidenceChunk(
        chunk_id="legal_sec_3_d_para_001",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="Patents Act 1970",
        source="legal/patents.pdf",
        page=1,
        section="Section 3(d)",
        section_id="sec_3_d",
        parent_id="legal_sec_3_d_heading",
        content_type=ContentType.PARAGRAPH,
        text="Under Section 3(d), the mere discovery of a new form of a known substance is not patentable.",
    )
    claims = extractor.extract_from_chunk(para_chunk)
    assert len(claims) == 1
    assert claims[0].reference_text == "Section 3(d)"
    assert claims[0].normalized_reference == "3(d)"
    assert claims[0].claim_text == "the mere discovery of a new form of a known substance is not patentable"


def test_author_year_preserves_complete_subject_and_predicate(extractor):
    """Verify author-year citation preserves complete substantive subject + predicate."""
    # 1. Single citation sentence
    text_single = "The proposed architecture improves precision (Smith et al., 2024)."
    claims_s = extractor.extract_from_text(
        text=text_single,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims_s) == 1
    assert claims_s[0].claim_text == "The proposed architecture improves precision"
    assert claims_s[0].normalized_reference == "Smith et al., 2024"

    # 2. Compound coordinate sentence
    text_compound = "The proposed architecture reduces verification latency [5–7] and improves precision (Smith et al., 2024)."
    claims_c = extractor.extract_from_text(
        text=text_compound,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims_c) == 2
    assert claims_c[0].claim_text == "The proposed architecture reduces verification latency"
    assert claims_c[0].normalized_reference == "5-7"
    assert claims_c[1].claim_text == "The proposed architecture improves precision"
    assert claims_c[1].normalized_reference == "Smith et al., 2024"


def test_numeric_citation_preserves_complete_claim_context(extractor):
    """Verify numeric citation preserves full WHO/WHAT + ACTION/PROPERTY + VALUE."""
    text = "Deep neural networks achieve state-of-the-art results on benchmark tasks [42]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="paper_01",
    )
    assert len(claims) == 1
    assert claims[0].claim_text == "Deep neural networks achieve state-of-the-art results on benchmark tasks"
    assert claims[0].normalized_reference == "42"


def test_citation_removal_preserves_essential_grammatical_object(extractor):
    """Verify that when a reference is an essential grammatical object, it is preserved in claim_text."""
    text = "In Novartis AG v. Union of India, the Supreme Court upheld Section 3(d)."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="legal_case",
    )
    assert len(claims) == 1
    claim = claims[0]
    assert "Novartis AG" in claim.claim_text
    assert "Supreme Court" in claim.claim_text
    assert "upheld" in claim.claim_text
    assert "Section 3(d)" in claim.claim_text
    assert claim.claim_text == "In Novartis AG v. Union of India, the Supreme Court upheld Section 3(d)"
    assert claim.reference_text == "Section 3(d)"
    assert claim.normalized_reference == "3(d)"


def test_repeated_extraction_remains_deterministic_quality(extractor):
    """Verify determinism across multiple runs with quality fixes."""
    chunk = EvidenceChunk(
        chunk_id="sci_eval_chunk_001",
        document_id="sci_eval_doc",
        domain=Domain.SCIENTIFIC,
        title="Evaluation",
        source="scientific/eval.pdf",
        page=1,
        section="Methods",
        section_id="methods",
        parent_id=None,
        content_type=ContentType.PARAGRAPH,
        text="The proposed architecture reduces verification latency [5–7] and improves precision (Smith et al., 2024).",
    )
    run_1 = extractor.extract_from_chunk(chunk)
    run_2 = extractor.extract_from_chunk(chunk)
    run_3 = extractor.extract_from_chunk(chunk)

    assert len(run_1) == 2
    assert [c.claim_id for c in run_1] == [c.claim_id for c in run_2] == [c.claim_id for c in run_3]
    assert [c.claim_text for c in run_1] == [c.claim_text for c in run_2] == [c.claim_text for c in run_3]
    assert [c.normalized_reference for c in run_1] == [c.normalized_reference for c in run_2] == [c.normalized_reference for c in run_3]

