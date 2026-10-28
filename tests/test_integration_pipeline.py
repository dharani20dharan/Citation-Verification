"""Comprehensive Integration and Regression Test Suite for Stage 10C.

Validates the complete automated citation verification pipeline as one deterministic,
auditable workflow:
Document → Parsing → Chunking → Citation/Claim Extraction → Reference Resolution
→ Retrieval → Source Attribution → Evidence Selection → Semantic Verification
→ Auditable Final Result
"""

import json
from pathlib import Path
import pytest

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain, EvidenceChunk
from src.pipeline.models import (
    PipelineClaimResult,
    PipelineConfiguration,
    PipelineResult,
    ProcessingStatus,
)
from src.pipeline.orchestrator import VerificationPipelineOrchestrator
from src.references.models import BibliographicEntry
from src.references.resolution_engine import ReferenceResolutionEngine
from src.verification.llm_verifier import MockSemanticVerifier
from src.verification.models import EvidenceItem, Verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator() -> VerificationPipelineOrchestrator:
    return VerificationPipelineOrchestrator()


@pytest.fixture
def legal_chunks() -> list[EvidenceChunk]:
    chunks_path = Path("data/chunks/legal/legal_sample_fixture.jsonl")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(EvidenceChunk.model_validate(json.loads(line)))
    return chunks


@pytest.fixture
def scientific_citations_chunks() -> list[EvidenceChunk]:
    chunks_path = Path("data/chunks/scientific/scientific_citations_fixture.jsonl")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(EvidenceChunk.model_validate(json.loads(line)))
    return chunks


# ---------------------------------------------------------------------------
# Test A — Legal Supported Citation (Section 3(d))
# ---------------------------------------------------------------------------

def test_legal_supported_citation_end_to_end(orchestrator, legal_chunks):
    """Test A: Parse/load chunks -> extract legal claim -> resolve Section 3(d)

    -> exact retrieval -> source available -> substantive evidence selected
    -> semantic verification -> supported.
    """
    result: PipelineResult = orchestrator.run(
        input_document=legal_chunks,
        domain=Domain.LEGAL,
        config=PipelineConfiguration(verifier_mode="deterministic"),
        document_id="legal_sample_fixture",
    )

    assert result.processing_status == ProcessingStatus.SUCCESS
    assert len(result.claims) >= 1

    claim_res = next((c for c in result.claims if "3(d)" in c.normalized_reference), None)
    assert claim_res is not None, "Claim referencing Section 3(d) must be extracted"

    # Exact reference match
    assert claim_res.reference_status == "exact_match"
    # Source availability
    assert claim_res.source_attribution.get("source_availability") == "available"
    # Final verdict
    assert claim_res.verdict == Verdict.SUPPORTED
    assert claim_res.is_conclusive is True
    # Substantive evidence selected
    assert any(not e.is_structural for e in claim_res.selected_evidence)
    # Structural headings not treated as factual evidence
    for e in claim_res.selected_evidence:
        if e.is_structural:
            assert getattr(e, "is_eligible", True) is False or e.evidence_type == "structural"
    # Provenance contains page/section
    assert claim_res.provenance.get("page") is not None or any(e.page is not None for e in claim_res.selected_evidence)


# ---------------------------------------------------------------------------
# Test B — Legal Wrong/Nonexistent Reference (Section 9)
# ---------------------------------------------------------------------------

def test_legal_nonexistent_reference_not_found(orchestrator):
    """Test B: Legal claim referencing nonexistent Section 9 must produce not_found.

    Semantic candidates must NOT establish reference existence.
    """
    claim = CitationClaim(
        claim_id="legal_claim_sec_9",
        document_id="legal_sample_fixture",
        domain=Domain.LEGAL,
        claim_text="Section 9 provides for compulsory provisional specifications within twelve months.",
        reference_text="Section 9",
        normalized_reference="9",
        normalized_references=["9"],
        reference_type=ReferenceType.LEGAL_SECTION.value,
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.LEGAL,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.reference_status == "not_found"
    assert cr.verdict == Verdict.NOT_FOUND
    assert cr.is_conclusive is True
    assert "not found in the evidence corpus" in cr.reasoning or "not found" in cr.reasoning.lower()


# ---------------------------------------------------------------------------
# Test C — Scientific Citation with Resolved Bibliography but Unavailable Source ([12])
# ---------------------------------------------------------------------------

def test_scientific_resolved_ref_unavailable_source(orchestrator):
    """Test C: Scientific [12] bibliography resolves, but cited source is unavailable locally.

    Must return insufficient_evidence (inconclusive).
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

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    # Bibliography resolved
    assert cr.reference_status == "exact_match"
    # Source availability is unavailable
    assert cr.source_attribution.get("source_availability") == "unavailable"
    # Inconclusive insufficient_evidence verdict
    assert cr.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert cr.is_conclusive is False
    assert cr.verdict not in (Verdict.SUPPORTED, Verdict.UNSUPPORTED, Verdict.PARTIALLY_CORRECT)


# ---------------------------------------------------------------------------
# Test D — Scientific Citation with Available Source & Supporting Evidence
# ---------------------------------------------------------------------------

def test_scientific_available_source_supported(orchestrator):
    """Test D: Smith et al. 2024 source is available locally and supports claim."""
    claim = CitationClaim(
        claim_id="sci_eval_01_claim_004",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture improves precision",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.source_attribution.get("source_availability") == "available"
    assert cr.verdict == Verdict.SUPPORTED
    assert cr.is_conclusive is True
    # Evidence must originate from verified source sci_doc_smith_2024
    assert any(e.chunk_id == "sci_doc_smith_2024_chunk_001" and getattr(e, "is_eligible", True) for e in cr.selected_evidence)


# ---------------------------------------------------------------------------
# Test E — Scientific Contradiction
# ---------------------------------------------------------------------------

def test_scientific_contradiction_unsupported(orchestrator):
    """Test E: Verified source evidence contradicts the claim proposition -> unsupported."""
    claim = CitationClaim(
        claim_id="sci_eval_contradiction_claim",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture degrades precision and increases verification latency",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    config = PipelineConfiguration(
        verifier_mode="mock",
        custom_mock_response={
            "verdict": "unsupported",
            "confidence": 0.94,
            "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
            "reasoning": "The evidence directly refutes the claim: it demonstrates reduced latency and improved precision.",
        },
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=config,
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.source_attribution.get("source_availability") == "available"
    assert cr.verdict == Verdict.UNSUPPORTED
    assert cr.is_conclusive is True


# ---------------------------------------------------------------------------
# Test F — Partially Supported Claim
# ---------------------------------------------------------------------------

def test_partially_supported_compound_claim(orchestrator):
    """Test F: Compound claim with supported and unsupported components -> partially_correct."""
    claim = CitationClaim(
        claim_id="sci_eval_partial_claim",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture reduces verification latency and achieves zero memory overhead",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    config = PipelineConfiguration(
        verifier_mode="mock",
        custom_mock_response={
            "verdict": "partially_correct",
            "confidence": 0.88,
            "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
            "reasoning": "Reduced latency is corroborated, but zero memory overhead is unsupported.",
        },
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=config,
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.verdict == Verdict.PARTIALLY_CORRECT
    assert cr.is_conclusive is True
    assert "sci_doc_smith_2024_chunk_001" in [e.chunk_id for e in cr.selected_evidence]


# ---------------------------------------------------------------------------
# Test G — Insufficient Evidence
# ---------------------------------------------------------------------------

def test_insufficient_evidence_scenario(orchestrator):
    """Test G: Available source where evidence neither proves nor refutes the claim -> insufficient_evidence."""
    claim = CitationClaim(
        claim_id="sci_eval_insufficient_claim",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The model was trained on multimodal biomedical imagery datasets",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    config = PipelineConfiguration(
        verifier_mode="mock",
        custom_mock_response={
            "verdict": "insufficient_evidence",
            "confidence": 0.70,
            "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
            "reasoning": "The evidence does not address training datasets or multimodal imagery.",
        },
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=config,
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert cr.is_conclusive is False


# ---------------------------------------------------------------------------
# Missing Reference Guard
# ---------------------------------------------------------------------------

def test_missing_reference_guard_bypasses_semantic_verifier(orchestrator):
    """Unresolved citation sets reference_status = not_found and bypasses semantic verifier."""
    claim = CitationClaim(
        claim_id="missing_ref_claim",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Section 999 excludes this subject matter completely.",
        reference_text="Section 999",
        normalized_reference="999",
        reference_type=ReferenceType.LEGAL_SECTION.value,
    )

    # Configured mock claiming supported should NEVER be called because reference guard intervenes
    config = PipelineConfiguration(
        verifier_mode="mock",
        custom_mock_response={
            "verdict": "supported",
            "confidence": 1.0,
            "evidence_ids": [],
            "reasoning": "Fabricated support that should never appear",
        },
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.LEGAL,
        config=config,
    )

    assert len(result.claims) == 1
    cr = result.claims[0]
    assert cr.verdict == Verdict.NOT_FOUND
    assert "skipped by reference guard" in cr.reasoning or "not found" in cr.reasoning


# ---------------------------------------------------------------------------
# Evidence Eligibility Tests
# ---------------------------------------------------------------------------

def test_evidence_eligibility_enforcement(orchestrator):
    """Ineligible evidence (citing doc, unrelated doc, structural heading) cannot substantiate claim."""
    claim = CitationClaim(
        claim_id="sci_claim_eligibility",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="Language models exhibit few-shot learning capabilities",
        reference_text="[12]",
        normalized_reference="12",
        reference_type=ReferenceType.NUMERIC.value,
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    cr = result.claims[0]
    # Source is unavailable because citing document & unrelated chunks are filtered out
    assert cr.source_attribution["source_availability"] == "unavailable"
    assert cr.verdict == Verdict.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Provenance / Auditability Tests
# ---------------------------------------------------------------------------

def test_provenance_and_auditability_trace(orchestrator):
    """Every claim result must answer the 7 core audit questions."""
    claim = CitationClaim(
        claim_id="audit_claim_01",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture improves precision",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
        page=2,
    )

    result: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    cr = result.claims[0]
    # 1. What was claimed?
    assert cr.what_was_claimed() == "The proposed architecture improves precision"
    # 2. What was cited?
    assert cr.what_was_cited()["reference_text"] == "(Smith et al., 2024)"
    # 3. Which source was identified?
    assert cr.which_source_identified() is not None
    # 4. Was source available?
    assert cr.was_source_available() == "available"
    # 5. What evidence was considered?
    assert len(cr.what_evidence_considered()) > 0
    # 6. Why was evidence eligible?
    eligibility_info = cr.why_evidence_eligible()
    assert "sci_doc_smith_2024_chunk_001" in eligibility_info
    assert eligibility_info["sci_doc_smith_2024_chunk_001"]["is_eligible"] is True
    # 7. What verdict was produced?
    verdict_info = cr.what_verdict_produced()
    assert verdict_info["verdict"] == "supported"
    assert verdict_info["is_conclusive"] is True


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------

def test_deterministic_pipeline_execution(orchestrator):
    """Running the pipeline multiple times on identical input produces identical outputs."""
    claim = CitationClaim(
        claim_id="det_claim_01",
        document_id="sci_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture improves precision",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        normalized_references=["Smith et al., 2024"],
        reference_type=ReferenceType.AUTHOR_YEAR.value,
    )

    r1: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )
    r2: PipelineResult = orchestrator.run(
        input_document=[claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    c1 = r1.claims[0]
    c2 = r2.claims[0]

    assert c1.claim_id == c2.claim_id
    assert c1.normalized_reference == c2.normalized_reference
    assert c1.reference_status == c2.reference_status
    assert c1.verdict == c2.verdict
    assert c1.confidence == c2.confidence
    assert [e.chunk_id for e in c1.selected_evidence] == [e.chunk_id for e in c2.selected_evidence]
    assert c1.reasoning == c2.reasoning


# ---------------------------------------------------------------------------
# Domain Isolation Tests
# ---------------------------------------------------------------------------

def test_domain_isolation(orchestrator):
    """Legal claims query the legal collection; scientific claims query scientific collection."""
    legal_claim = CitationClaim(
        claim_id="legal_claim_iso",
        document_id="doc_legal",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type=ReferenceType.LEGAL_SECTION.value,
    )
    sci_claim = CitationClaim(
        claim_id="sci_claim_iso",
        document_id="doc_sci",
        domain=Domain.SCIENTIFIC,
        claim_text="The proposed architecture improves precision",
        reference_text="(Smith et al., 2024)",
        normalized_reference="Smith et al., 2024",
        reference_type=ReferenceType.AUTHOR_YEAR.value,
    )

    r_legal = orchestrator.run(
        input_document=[legal_claim],
        domain=Domain.LEGAL,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )
    r_sci = orchestrator.run(
        input_document=[sci_claim],
        domain=Domain.SCIENTIFIC,
        config=PipelineConfiguration(verifier_mode="deterministic"),
    )

    assert r_legal.domain == Domain.LEGAL
    assert r_legal.claims[0].domain == Domain.LEGAL
    assert all(e.chunk_id.startswith("legal_") for e in r_legal.claims[0].selected_evidence)

    assert r_sci.domain == Domain.SCIENTIFIC
    assert r_sci.claims[0].domain == Domain.SCIENTIFIC
    assert all(e.chunk_id.startswith("sci") for e in r_sci.claims[0].selected_evidence)


# ---------------------------------------------------------------------------
# Failure Propagation Tests
# ---------------------------------------------------------------------------

def test_nonexistent_file_parsing_failure(orchestrator):
    """Providing a nonexistent file produces a controlled FAILED status with error logging."""
    result: PipelineResult = orchestrator.run(
        input_document="nonexistent_document_file.pdf",
        domain=Domain.LEGAL,
    )

    assert result.processing_status == ProcessingStatus.FAILED
    assert result.stage_status["parsing"] == "failed"
    assert len(result.errors) > 0
    assert "Document parsing failed" in result.errors[0]
