"""Data models and schemas for Stage 10C End-to-End Integration Pipeline."""

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.models.schemas import Domain
from src.verification.models import EvidenceItem, Verdict


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    PARSING = "parsing"
    CHUNKING = "chunking"
    EXTRACTION = "extraction"
    REFERENCE_RESOLUTION = "reference_resolution"
    RETRIEVAL = "retrieval"
    SOURCE_ATTRIBUTION = "source_attribution"
    EVIDENCE_SELECTION = "evidence_selection"
    VERIFICATION = "verification"


class StageStatus(str, Enum):
    """Status for an individual pipeline stage."""
    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ProcessingStatus(str, Enum):
    """Overall status of end-to-end pipeline execution."""
    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"


class PipelineClaimResult(BaseModel):
    """Auditable result for a single claim preserving the full verification chain:

    claim
      ↓
    citation/reference
      ↓
    reference resolution
      ↓
    retrieval candidates
      ↓
    source attribution
      ↓
    selected evidence
      ↓
    semantic verdict
    """

    claim_id: str = Field(..., description="Unique deterministic claim identifier")
    document_id: str = Field(..., description="Source document containing the claim")
    domain: Domain = Field(..., description="Corpus domain: legal or scientific")

    # 1. Claim
    claim_text: str = Field(..., description="Exact extracted substantive proposition")

    # 2. Citation / Reference
    reference_text: str = Field(..., description="Raw cited reference marker")
    normalized_reference: str = Field(..., description="Canonical normalized reference identifier")
    reference_type: str = Field(..., description="Classification category of the reference")
    page: Optional[int] = Field(default=None, description="Page number where claim occurred")
    section: Optional[str] = Field(default=None, description="Section heading where claim occurred")

    # 3. Reference Resolution
    reference_status: str = Field(..., description="Status in corpus: exact_match, semantic_candidate, not_found")
    reference_resolution: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bibliographic resolution details from Stage 10A",
    )

    # 4. Retrieval Candidates
    retrieval_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of candidates returned by Stage 7 retrieval before selection",
    )

    # 5. Source Attribution
    source_attribution: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source identity & availability verification details from Stage 10B.1",
    )

    # 6. Selected Evidence
    selected_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Ranked evidence items selected for evaluation",
    )

    # 7. Semantic Verdict
    verdict: Verdict = Field(..., description="Final verification verdict")
    confidence: float = Field(..., description="Deterministic confidence tier", ge=0.0, le=1.0)
    is_conclusive: bool = Field(default=True, description="Whether verdict is conclusive vs inconclusive")
    reasoning: str = Field(..., description="Evidence-grounded rationale for verdict")
    verifier_version: str = Field(..., description="Version of the verifier implementation")

    # Provenance
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete traceable audit trail",
    )

    # Diagnostic Audit Helpers
    def what_was_claimed(self) -> str:
        return self.claim_text

    def what_was_cited(self) -> Dict[str, str]:
        return {
            "reference_text": self.reference_text,
            "normalized_reference": self.normalized_reference,
            "reference_type": self.reference_type,
        }

    def which_source_identified(self) -> Optional[str]:
        bib = self.reference_resolution
        if bib and bib.get("matched_entry_ids"):
            return ", ".join(bib["matched_entry_ids"])
        return self.normalized_reference

    def was_source_available(self) -> str:
        return self.source_attribution.get("source_availability", "unknown")

    def what_evidence_considered(self) -> List[str]:
        return [e.chunk_id for e in self.selected_evidence]

    def why_evidence_eligible(self) -> Dict[str, Any]:
        return {
            e.chunk_id: {
                "is_eligible": getattr(e, "is_eligible", True),
                "is_structural": e.is_structural,
                "document_id": e.document_id,
                "match_type": e.match_type,
            }
            for e in self.selected_evidence
        }

    def what_verdict_produced(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "is_conclusive": self.is_conclusive,
            "reasoning": self.reasoning,
        }


class PipelineSummary(BaseModel):
    """Statistical summary of end-to-end verification pipeline results."""

    total_claims: int = Field(default=0, description="Total claims extracted and evaluated")
    verdicts: Dict[str, int] = Field(default_factory=dict, description="Count of claims by verdict")
    conclusive_count: int = Field(default=0, description="Count of conclusive verdicts")
    inconclusive_count: int = Field(default=0, description="Count of inconclusive verdicts (insufficient_evidence)")
    reference_status_counts: Dict[str, int] = Field(default_factory=dict, description="Counts by reference status")
    source_availability_counts: Dict[str, int] = Field(default_factory=dict, description="Counts by source availability")
    eligible_evidence_total: int = Field(default=0, description="Total eligible evidence chunks used")
    ineligible_evidence_total: int = Field(default=0, description="Total ineligible evidence chunks excluded")


class PipelineConfiguration(BaseModel):
    """Configuration options for end-to-end pipeline execution."""

    top_k: int = Field(default=5, description="Max retrieval candidates from Stage 7")
    max_evidence: int = Field(default=5, description="Max evidence items selected for evaluation")
    verifier_mode: str = Field(default="deterministic", description="'deterministic', 'mock', or 'llm'")
    expand_parent_context: bool = Field(default=True, description="Whether to expand parent context")
    semantic_search_enabled: bool = Field(default=True, description="Whether to use dense vector search")
    custom_mock_response: Optional[Dict[str, Any]] = Field(default=None, description="Preset response for mock verifier")


class PipelineResult(BaseModel):
    """Top-level auditable end-to-end integration result schema."""

    document_id: str = Field(..., description="Document identifier")
    domain: Domain = Field(..., description="Corpus domain: legal or scientific")
    pipeline_version: str = Field(default="1.0.0-stage10c", description="Pipeline orchestrator version")

    claims: List[PipelineClaimResult] = Field(default_factory=list, description="Ordered verification results for claims")
    summary: PipelineSummary = Field(default_factory=PipelineSummary, description="High-level execution summary")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS, description="Overall pipeline status")

    stage_status: Dict[str, str] = Field(default_factory=dict, description="Status per pipeline stage")
    timings: Dict[str, float] = Field(default_factory=dict, description="Execution timing in seconds per stage")

    provenance: Dict[str, Any] = Field(default_factory=dict, description="Document and environment provenance")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during execution")
    warnings: List[str] = Field(default_factory=list, description="Warnings encountered during execution")
