"""Data models and schemas for citation verification and verdict engine."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.models.schemas import Domain


class Verdict(str, Enum):
    """Canonical verdict enumeration for claim verification against evidence."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIALLY_CORRECT = "partially_correct"
    WRONG_REFERENCE = "wrong_reference"
    NOT_FOUND = "not_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReferenceStatus(str, Enum):
    """Classification of cited reference discovery in the evidence corpus."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_CANDIDATE = "semantic_candidate"
    NOT_FOUND = "not_found"
    NOT_APPLICABLE = "not_applicable"


class EvidenceItem(BaseModel):
    """Structured, auditable evidence item selected from retrieval candidates."""

    chunk_id: str = Field(..., description="Unique source chunk identifier", min_length=1)
    evidence_id: str = Field(
        default="",
        description="Unique evidence identifier for LLM grounding (defaults to chunk_id)",
    )
    document_id: str = Field(..., description="Source document identifier", min_length=1)
    text: str = Field(..., description="Canonical textual content of the evidence chunk")
    match_type: str = Field(..., description="Retrieval match type: 'exact', 'semantic', or 'parent_context'")
    score: float = Field(..., description="Normalized retrieval or ranking score")
    similarity: Optional[float] = Field(default=None, description="Cosine similarity score if semantically retrieved")
    is_structural: bool = Field(
        default=False,
        description="Whether this is structural reference evidence (e.g. heading/title) rather than substantive factual evidence",
    )
    is_eligible: bool = Field(
        default=True,
        description="Whether this evidence item is eligible to substantiate claim support based on source identity",
    )
    evidence_type: str = Field(
        default="substantive",
        description="Classification: 'substantive' factual evidence vs 'structural' reference evidence",
    )

    # Complete retrieval and document provenance
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auditable metadata preserving source, page, section, section_id, parent_id, clause_id, etc.",
    )

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Identifier must not be empty or whitespace only")
        return cleaned

    def model_post_init(self, __context: Any) -> None:
        """Ensure evidence_id is set to chunk_id if not explicitly provided."""
        if not self.evidence_id:
            self.evidence_id = self.chunk_id

    @property
    def page(self) -> Optional[int]:
        return self.provenance.get("page")

    @property
    def section(self) -> Optional[str]:
        return self.provenance.get("section")

    @property
    def section_id(self) -> Optional[str]:
        return self.provenance.get("section_id")

    @property
    def parent_id(self) -> Optional[str]:
        return self.provenance.get("parent_id")

    @property
    def clause_id(self) -> Optional[str]:
        return self.provenance.get("clause_id")

    @property
    def content_type(self) -> Optional[str]:
        return self.provenance.get("content_type")


class VerificationDecision(BaseModel):
    """Intermediate decision returned by a claim verifier implementation.

    Note on Confidence Semantics:
        The confidence value represents a deterministic confidence tier (e.g. 0.95, 0.90,
        0.85, 0.70) assigned based on rule-based evidence sufficiency and match criteria.
        It is NOT a statistical, Bayesian, or probabilistic calibration.
    """

    verdict: Verdict = Field(..., description="Determined verification verdict")
    confidence: float = Field(
        ...,
        description="Deterministic confidence tier bounded between 0.0 and 1.0 (not a statistical probability)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(..., description="Factual, evidence-grounded explanation of the verdict", min_length=1)
    is_conclusive: bool = Field(
        default=True,
        description="Whether the decision is conclusive vs inconclusive (insufficient_evidence)",
    )
    verifier_version: str = Field(..., description="Version identifier of the verifier implementation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic verifier metadata")

    def model_post_init(self, __context: Any) -> None:
        """Update is_conclusive flag based on verdict."""
        self.is_conclusive = self.verdict != Verdict.INSUFFICIENT_EVIDENCE

    @field_validator("confidence")
    @classmethod
    def validate_confidence_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return round(float(v), 4)


class VerificationResult(BaseModel):
    """Auditable result of citation and claim verification against corpus evidence.

    Note on Confidence Semantics:
        The confidence value represents a deterministic confidence tier (e.g. 0.95, 0.90,
        0.85, 0.70) assigned based on rule-based evidence sufficiency and match criteria.
        It is NOT a statistical, Bayesian, or probabilistic calibration.
    """

    verification_id: str = Field(..., description="Deterministic unique verification run identifier", min_length=1)
    claim_id: str = Field(..., description="Identifier of the claim being verified", min_length=1)
    document_id: str = Field(..., description="Document identifier containing the claim", min_length=1)
    domain: Domain = Field(..., description="Domain of the verified claim ('legal' or 'scientific')")

    claim_text: str = Field(..., description="Substantive factual statement being verified", min_length=1)

    reference_text: str = Field(..., description="Exact raw reference text as cited in document", min_length=1)
    normalized_reference: str = Field(..., description="Canonical reference identifier", min_length=1)
    reference_type: str = Field(..., description="Classification category of the reference")

    verdict: Verdict = Field(..., description="Final verification verdict")
    reference_status: ReferenceStatus = Field(..., description="Status of the cited reference in the corpus")

    confidence: float = Field(
        ...,
        description="Deterministic confidence tier between 0.0 and 1.0 (not a statistical probability)",
        ge=0.0,
        le=1.0,
    )

    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Ranked list of auditable evidence items used to reach the verdict",
    )

    reasoning: str = Field(..., description="Concise, factual reasoning explaining the verdict", min_length=1)

    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete provenance trace: claim -> citation -> retrieved chunks -> verdict",
    )

    verifier_version: str = Field(..., description="Version of the verifier engine")
    is_conclusive: bool = Field(
        default=True,
        description="Whether the verification produced a conclusive final verdict",
    )
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of verification execution")

    def model_post_init(self, __context: Any) -> None:
        """Update is_conclusive flag based on verdict."""
        self.is_conclusive = self.verdict != Verdict.INSUFFICIENT_EVIDENCE

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return round(float(v), 4)


class LLMVerificationDecision(BaseModel):
    """Strict structured response produced by semantic claim-evidence verifier.

    Enforces that the LLM only performs semantic evaluation (supported, unsupported, partially_correct)
    and strictly cites supplied evidence_ids. Reference-existence verdicts (not_found, wrong_reference)
    are strictly rejected.
    """

    verdict: Verdict = Field(
        ...,
        description="Semantic evaluation verdict: 'supported', 'unsupported', or 'partially_correct'",
    )
    confidence: float = Field(
        ...,
        description="Confidence in the semantic relationship between claim and supplied evidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Identifiers of supplied evidence items explicitly supporting the verdict",
    )
    reasoning: str = Field(
        ...,
        description="Evidence-grounded explanation of the semantic verdict",
        min_length=1,
    )

    @field_validator("verdict")
    @classmethod
    def validate_allowed_verdicts(cls, v: Verdict) -> Verdict:
        allowed = {
            Verdict.SUPPORTED,
            Verdict.UNSUPPORTED,
            Verdict.PARTIALLY_CORRECT,
            Verdict.INSUFFICIENT_EVIDENCE,
        }
        if v not in allowed:
            raise ValueError(
                f"LLM semantic verifier cannot produce '{v.value}'. "
                f"Allowed verdicts are: {[a.value for a in allowed]}. "
                f"Reference existence (not_found, wrong_reference) is deterministic."
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return round(float(v), 4)

    @field_validator("reasoning")
    @classmethod
    def normalize_reasoning_whitespace(cls, v: str) -> str:
        import re
        cleaned = re.sub(r"\s+", " ", v).strip()
        if not cleaned:
            raise ValueError("Reasoning must not be empty or whitespace only")
        return cleaned


class VerificationFailure(BaseModel):
    """Structured representation of a verification execution failure."""

    error_type: str = Field(
        ...,
        description="Controlled category: invalid_response, schema_validation_error, unknown_evidence_id, timeout, provider_error, rate_limited, empty_evidence, configuration_error",
    )
    message: str = Field(..., description="Sanitized, safe description of the failure (no credentials)")
    retryable: bool = Field(default=False, description="Whether the failure is transient and can be retried")
    stage: str = Field(default="semantic_verification", description="Pipeline stage where failure occurred")
    claim_id: Optional[str] = Field(default=None, description="Identifier of the claim that encountered failure")

