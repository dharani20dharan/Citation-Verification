"""Comprehensive regression tests for Stage 10B.1: Semantic Verification Grounding Fixes.

Covers:
1. Resolved bibliography entry with unavailable source content.
2. Resolved citation with unrelated semantic retrieval results.
3. Supported claim with substantive evidence from the correct source.
4. Contradiction established by evidence from the correct source.
5. Partially supported compound claim.
6. Insufficient evidence.
7. Structural-heading-only evidence.
8. Unknown evidence IDs.
9. Missing-reference guard.
10. Deterministic mock behavior.
"""

from pathlib import Path
import pytest

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain
from src.references.models import BibliographicEntry
from src.references.resolution_engine import ReferenceResolutionEngine
from src.retrieval.models import MatchType, RetrievalResult
from src.retrieval.retriever import Retriever
from src.verification.evidence_selector import EvidenceSelector
from src.verification.llm_verifier import MockSemanticVerifier
from src.verification.models import (
    EvidenceItem,
    LLMVerificationDecision,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
    VerificationResult,
)
from src.verification.response_parser import ResponseParser, ResponseParserError
from src.verification.source_verifier import SourceIdentityVerifier
from src.verification.verification_engine import VerificationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser() -> ResponseParser:
    return ResponseParser()


@pytest.fixture
def source_verifier() -> SourceIdentityVerifier:
    return SourceIdentityVerifier()


@pytest.fixture
def sample_bib_entry_12() -> BibliographicEntry:
    return BibliographicEntry(
        entry_id="sci_eval_01_ref_12",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        reference_label="[12]",
        normalized_label="12",
        raw_text="[12] Brown, T., et al. Language Models are Few-Shot Learners. NeurIPS, 2020. doi:10.48550/arXiv.2005.14165",
        title="Language Models are Few-Shot Learners",
        authors=["Brown, T.", "Mann, B."],
        year=2020,
        doi="10.48550/arXiv.2005.14165",
        provenance={"document_id": "sci_eval_01", "page": 3, "section": "References"},
    )


@pytest.fixture
def sample_bib_entry_smith() -> BibliographicEntry:
    return BibliographicEntry(
        entry_id="sci_eval_01_ref_smith_2024",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        reference_label="Smith et al., 2024",
        normalized_label="Smith_2024",
        raw_text="Smith, J., Johnson, K., and Williams, R. (2024). Precision and Recall in Automated Citation Verification. JAIR, 79, 101-125. https://doi.org/10.1613/jair.1.12345",
        title="Precision and Recall in Automated Citation Verification",
        authors=["Smith, J.", "Johnson, K.", "Williams, R."],
        year=2024,
        doi="10.1613/jair.1.12345",
        provenance={"document_id": "sci_eval_01", "source_document_id": "sci_doc_smith_2024"},
    )


# ---------------------------------------------------------------------------
# 1. Resolved bibliography entry with unavailable source content
# ---------------------------------------------------------------------------

def test_resolved_bib_entry_with_unavailable_source_content(sample_bib_entry_12):
    """Scientific [12] resolves bibliographic identity, but because the cited publication's

    content is not available locally, it produces an inconclusive insufficient_evidence verdict.
    """
    claim = CitationClaim(
        claim_id="sci_eval_01_claim_001",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="Recent studies demonstrate that hallucinated citations compromise scientific integrity",
        reference_text="[12]",
        normalized_reference="12",
        normalized_references=["12"],
        reference_type=ReferenceType.NUMERIC.value,
        page=1,
    )

    ref_engine = ReferenceResolutionEngine()
    engine = VerificationEngine(
        verifier=MockSemanticVerifier(),
        reference_resolver=ref_engine,
    )

    result = engine.verify(claim)

    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert result.is_conclusive is False
    assert result.provenance.get("source_availability") == "unavailable"
    assert "source_unavailable" in result.reasoning or "not available in the local corpus" in result.reasoning


# ---------------------------------------------------------------------------
# 2. Resolved citation with unrelated semantic retrieval results
# ---------------------------------------------------------------------------

def test_resolved_citation_with_unrelated_semantic_retrieval(sample_bib_entry_12, source_verifier):
    """Unrelated local corpus evidence must NOT be eligible to support the claim,

    even if semantic retrieval returned high similarity scores.
    """
    claim = CitationClaim(
        claim_id="claim_001",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="Recent studies demonstrate that hallucinated citations compromise scientific integrity",
        reference_text="[12]",
        normalized_reference="12",
        reference_type=ReferenceType.NUMERIC.value,
    )

    # Retrieval returns chunks from citing document and an unrelated paper
    evidence = [
        EvidenceItem(
            chunk_id="sci_eval_01_chunk_001",
            document_id="sci_eval_01",  # Citing document
            text="Recent studies demonstrate that hallucinated citations compromise scientific integrity [12].",
            match_type="semantic",
            score=0.92,
        ),
        EvidenceItem(
            chunk_id="unrelated_paper_chunk_005",
            document_id="unrelated_doc_99",  # Unrelated paper
            text="Hallucinations in language models compromise truthfulness.",
            match_type="semantic",
            score=0.89,
        ),
    ]

    filtered_ev, meta = source_verifier.verify_source_eligibility(
        claim=claim,
        evidence=evidence,
        resolved_references=[sample_bib_entry_12],
    )

    assert meta["source_availability"] == "unavailable"
    assert meta["eligible_substantive_count"] == 0
    # Both items must be marked ineligible
    assert all(not e.is_eligible for e in filtered_ev)


# ---------------------------------------------------------------------------
# 3. Supported claim with substantive evidence from the correct source
# ---------------------------------------------------------------------------

def test_supported_claim_with_correct_source_substantive_evidence():
    """When the cited publication's substantive content is genuinely available locally

    and supports the claim, the verdict is conclusive 'supported'.
    """
    claim = CitationClaim(
        claim_id="sci_eval_01_claim_004",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture improves precision in automated citation verification",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    # Mock verifier configured to validate the semantic support
    mock_verifier = MockSemanticVerifier(
        response={
            "verdict": "supported",
            "confidence": 0.95,
            "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
            "reasoning": "The empirical results in Smith et al. (2024) directly demonstrate improved precision.",
        }
    )

    ref_engine = ReferenceResolutionEngine()
    engine = VerificationEngine(
        verifier=mock_verifier,
        reference_resolver=ref_engine,
    )

    result = engine.verify(claim)

    assert result.verdict == Verdict.SUPPORTED
    assert result.is_conclusive is True
    assert result.provenance.get("source_availability") == "available"
    assert any(e.chunk_id == "sci_doc_smith_2024_chunk_001" and e.is_eligible for e in result.evidence)


# ---------------------------------------------------------------------------
# 4. Contradiction established by evidence from the correct source
# ---------------------------------------------------------------------------

def test_contradiction_from_correct_source(sample_bib_entry_smith, source_verifier, parser):
    """When verified source evidence directly contradicts the claim, return 'unsupported'."""
    claim = CitationClaim(
        claim_id="claim_contradiction",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture degrades precision and increases latency.",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith_2024",
        reference_type=ReferenceType.AUTHOR_YEAR.value,
    )

    evidence = [
        EvidenceItem(
            chunk_id="sci_doc_smith_2024_chunk_001",
            document_id="sci_doc_smith_2024",
            text="Our empirical evaluation demonstrates that the proposed architecture reduces verification latency and improves precision.",
            match_type="exact",
            score=0.98,
        )
    ]

    filtered_ev, meta = source_verifier.verify_source_eligibility(
        claim=claim,
        evidence=evidence,
        resolved_references=[sample_bib_entry_smith],
    )
    assert meta["source_availability"] == "available"
    assert filtered_ev[0].is_eligible is True

    # Verifier establishes contradiction
    mock_resp = {
        "verdict": "unsupported",
        "confidence": 0.94,
        "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
        "reasoning": "The evidence directly refutes the claim: it demonstrates reduced latency and improved precision, whereas the claim asserts degradation.",
    }
    decision, _ = parser.parse(mock_resp, supplied_evidence=filtered_ev)
    assert decision.verdict == Verdict.UNSUPPORTED
    assert decision.confidence == 0.94


# ---------------------------------------------------------------------------
# 5. Partially supported compound claim
# ---------------------------------------------------------------------------

def test_partially_supported_compound_claim(sample_bib_entry_smith, parser):
    """A compound proposition where one conjunct is supported and one is unsupported

    must produce 'partially_correct'.
    """
    evidence = [
        EvidenceItem(
            chunk_id="sci_doc_smith_2024_chunk_001",
            document_id="sci_doc_smith_2024",
            text="The proposed architecture reduces verification latency and improves precision.",
            match_type="exact",
            score=0.98,
        )
    ]

    mock_resp = {
        "verdict": "partially_correct",
        "confidence": 0.88,
        "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
        "reasoning": "The evidence supports reduced latency, but does not substantiate the 99% accuracy claim.",
    }
    decision, _ = parser.parse(mock_resp, supplied_evidence=evidence)
    assert decision.verdict == Verdict.PARTIALLY_CORRECT
    assert decision.confidence == 0.88


# ---------------------------------------------------------------------------
# 6. Insufficient evidence outcome
# ---------------------------------------------------------------------------

def test_insufficient_evidence_outcome(parser):
    """When the evidence neither proves nor refutes the claim, return insufficient_evidence

    as an inconclusive verification state.
    """
    evidence = [
        EvidenceItem(
            chunk_id="chunk_tangential",
            document_id="doc_src",
            text="The evaluation framework was initialized with default hyperparameter configurations.",
            match_type="semantic",
            score=0.65,
        )
    ]

    mock_resp = {
        "verdict": "insufficient_evidence",
        "confidence": 0.70,
        "evidence_ids": ["chunk_tangential"],
        "reasoning": "The supplied text discusses initialization parameters and contains no findings regarding retrieval accuracy.",
    }
    decision, _ = parser.parse(mock_resp, supplied_evidence=evidence)
    assert decision.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert decision.confidence == 0.70

    # Test inconclusive state in VerificationDecision and VerificationResult
    result_decision = VerificationDecision(
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=0.70,
        reasoning="Inconclusive",
        verifier_version="1.0.0",
    )
    assert result_decision.is_conclusive is False


# ---------------------------------------------------------------------------
# 7. Structural-heading-only evidence rejected
# ---------------------------------------------------------------------------

def test_structural_heading_only_evidence_rejected_for_support(parser):
    """A verifier claiming 'supported' while citing only structural headings must be rejected."""
    evidence = [
        EvidenceItem(
            chunk_id="head_01",
            evidence_id="ev_head",
            document_id="doc_01",
            text="1. Introduction and Background",
            match_type="exact",
            score=0.90,
            is_structural=True,
            evidence_type="structural",
        )
    ]

    invalid_resp = {
        "verdict": "supported",
        "confidence": 0.95,
        "evidence_ids": ["ev_head"],
        "reasoning": "The heading proves the claim.",
    }

    with pytest.raises(ResponseParserError) as exc_info:
        parser.parse(invalid_resp, supplied_evidence=evidence)

    assert exc_info.value.failure.error_type == "structural_only_support"


# ---------------------------------------------------------------------------
# 8. Unknown evidence IDs rejected
# ---------------------------------------------------------------------------

def test_unknown_evidence_ids_rejected(parser):
    """Citing evidence IDs that were not supplied must be rejected with unknown_evidence_id."""
    evidence = [
        EvidenceItem(
            chunk_id="chunk_valid",
            evidence_id="ev_valid",
            document_id="doc_01",
            text="Substantive evidence text.",
            match_type="exact",
            score=0.95,
        )
    ]

    invalid_resp = {
        "verdict": "supported",
        "confidence": 0.90,
        "evidence_ids": ["ev_hallucinated_id"],
        "reasoning": "Evidence supports claim.",
    }

    with pytest.raises(ResponseParserError) as exc_info:
        parser.parse(invalid_resp, supplied_evidence=evidence)

    assert exc_info.value.failure.error_type == "unknown_evidence_id"


# ---------------------------------------------------------------------------
# 9. Missing-reference guard
# ---------------------------------------------------------------------------

def test_missing_reference_guard_bypasses_llm():
    """When a cited reference does not exist in the corpus, LLM semantic evaluation

    is completely bypassed and not_found is returned immediately.
    """
    claim = CitationClaim(
        claim_id="legal_missing_claim",
        document_id="doc_legal",
        domain=Domain.LEGAL,
        claim_text="Section 999 excludes this category of subject matter.",
        reference_text="Section 999",
        normalized_reference="999",
        reference_type=ReferenceType.LEGAL_SECTION.value,
    )

    verifier = MockSemanticVerifier(
        response={"verdict": "supported", "confidence": 1.0, "reasoning": "Fake"}
    )
    engine = VerificationEngine(verifier=verifier)

    result = engine.verify(claim)

    assert result.verdict == Verdict.NOT_FOUND
    assert result.reference_status == ReferenceStatus.NOT_FOUND
    assert "LLM semantic verification skipped by reference guard" in result.reasoning
    assert result.is_conclusive is True


# ---------------------------------------------------------------------------
# 10. Deterministic mock behavior
# ---------------------------------------------------------------------------

def test_deterministic_mock_behavior(sample_bib_entry_smith):
    """The default mock must not return supported merely because substantive evidence exists,

    and configured mock responses must return deterministic, identical judgments.
    """
    claim = CitationClaim(
        claim_id="claim_mock_det",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The model demonstrates superior reasoning capabilities.",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith_2024",
        reference_type=ReferenceType.AUTHOR_YEAR.value,
    )

    evidence = [
        EvidenceItem(
            chunk_id="sci_doc_smith_2024_chunk_001",
            document_id="sci_doc_smith_2024",
            text="Our empirical evaluation demonstrates that the proposed architecture reduces verification latency.",
            match_type="exact",
            score=0.98,
        )
    ]

    # Default mock without response returns insufficient_evidence, NOT supported
    default_mock = MockSemanticVerifier()
    default_decision = default_mock.verify(
        claim=claim,
        reference_status=ReferenceStatus.EXACT_MATCH,
        evidence=evidence,
    )
    assert default_decision.verdict == Verdict.INSUFFICIENT_EVIDENCE

    # Configured mock is strictly deterministic
    cfg_resp = {
        "verdict": "partially_correct",
        "confidence": 0.85,
        "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
        "reasoning": "Latency is reduced, but superior reasoning is unaddressed.",
    }
    m1 = MockSemanticVerifier(response=cfg_resp)
    m2 = MockSemanticVerifier(response=cfg_resp)

    d1 = m1.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)
    d2 = m2.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)

    assert d1.verdict == d2.verdict
    assert d1.confidence == d2.confidence
    assert d1.reasoning == d2.reasoning
