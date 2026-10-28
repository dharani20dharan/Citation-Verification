"""Data models, error taxonomy, and metrics schemas for Stage 11 Evaluation Benchmark."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.schemas import Domain
from src.verification.models import Verdict


class PipelineStage(str, Enum):
    """Pipeline stages for stage-level failure attribution."""
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    CLAIM_EXTRACTION = "CLAIM_EXTRACTION"
    REFERENCE_RESOLUTION = "REFERENCE_RESOLUTION"
    RETRIEVAL = "RETRIEVAL"
    SOURCE_ATTRIBUTION = "SOURCE_ATTRIBUTION"
    EVIDENCE_SELECTION = "EVIDENCE_SELECTION"
    SEMANTIC_VERIFICATION = "SEMANTIC_VERIFICATION"
    RESULT_COMPILATION = "RESULT_COMPILATION"


class ErrorType(str, Enum):
    """Deterministic error taxonomy for failure diagnosis."""
    WRONG_REFERENCE_RESOLUTION = "WRONG_REFERENCE_RESOLUTION"
    MISSED_CITATION = "MISSED_CITATION"
    INCORRECT_CLAIM_BOUNDARY = "INCORRECT_CLAIM_BOUNDARY"
    MISSED_SOURCE = "MISSED_SOURCE"
    WRONG_SOURCE = "WRONG_SOURCE"
    INELIGIBLE_EVIDENCE_SELECTED = "INELIGIBLE_EVIDENCE_SELECTED"
    RELEVANT_EVIDENCE_MISSED = "RELEVANT_EVIDENCE_MISSED"
    STRUCTURAL_EVIDENCE_LEAK = "STRUCTURAL_EVIDENCE_LEAK"
    SUPPORT_CLASSIFICATION_ERROR = "SUPPORT_CLASSIFICATION_ERROR"
    CONTRADICTION_CLASSIFICATION_ERROR = "CONTRADICTION_CLASSIFICATION_ERROR"
    PARTIAL_SUPPORT_CLASSIFICATION_ERROR = "PARTIAL_SUPPORT_CLASSIFICATION_ERROR"
    INSUFFICIENT_EVIDENCE_CLASSIFICATION_ERROR = "INSUFFICIENT_EVIDENCE_CLASSIFICATION_ERROR"
    PROVENANCE_LOSS = "PROVENANCE_LOSS"
    NONDETERMINISTIC_RESULT = "NONDETERMINISTIC_RESULT"


class BenchmarkCase(BaseModel):
    """Benchmark test case with explicit, ground-truth annotations."""

    case_id: str = Field(..., description="Unique deterministic case identifier")
    domain: Domain = Field(..., description="Domain: legal or scientific")
    document_id: str = Field(..., description="Target or citing document identifier")
    claim: str = Field(..., description="Proposition text to verify")
    reference: str = Field(..., description="Citation marker text")
    normalized_reference: Optional[str] = Field(default=None, description="Expected normalized reference string")
    reference_type: Optional[str] = Field(default=None, description="Expected ReferenceType category")

    expected_reference_status: str = Field(..., description="Expected reference discovery status: exact_match, not_found, semantic_candidate")
    expected_source_status: str = Field(..., description="Expected source availability: available, unavailable, unresolved_reference")
    expected_verdict: Verdict = Field(..., description="Expected ground-truth verdict")
    expected_conclusive: bool = Field(..., description="Whether verdict is expected to be conclusive")

    expected_evidence_behavior: Dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit evidence criteria: eligible_required, prohibited_chunks, required_chunks",
    )
    category: str = Field(..., description="Evaluation category: reference_correctness, source_availability, evidence_support, evidence_attribution")
    description: str = Field(..., description="Explanation of the test scenario and expected invariant")
    mock_response: Optional[Dict[str, Any]] = Field(default=None, description="Explicit mock verifier response if mock mode is used")


class StageFailureDiagnosis(BaseModel):
    """Detailed diagnosis identifying the earliest pipeline stage and error classification."""

    stage: PipelineStage = Field(..., description="Earliest pipeline stage where failure occurred")
    error_types: List[ErrorType] = Field(..., description="Classified error categories from taxonomy")
    reason: str = Field(..., description="Factual explanation of divergence from ground truth")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic state snapshots")


class CaseEvaluationResult(BaseModel):
    """Complete evaluation outcome for an individual benchmark case."""

    case_id: str
    domain: Domain
    category: str
    passed: bool

    expected_verdict: str
    actual_verdict: str

    expected_reference_status: str
    actual_reference_status: str

    expected_source_status: str
    actual_source_status: str

    expected_conclusive: bool
    actual_conclusive: bool

    confidence: float
    evidence_ids: List[str] = Field(default_factory=list)
    eligible_evidence_ids: List[str] = Field(default_factory=list)

    evidence_trace: Dict[str, Any] = Field(default_factory=dict)
    failure_diagnosis: Optional[StageFailureDiagnosis] = None
    reasoning: str = ""


class ConfusionMatrix(BaseModel):
    """Confusion matrix tracking expected vs actual verdicts across all valid Verdict types."""

    labels: List[str] = Field(
        default_factory=lambda: [
            Verdict.SUPPORTED.value,
            Verdict.UNSUPPORTED.value,
            Verdict.PARTIALLY_CORRECT.value,
            Verdict.NOT_FOUND.value,
            Verdict.INSUFFICIENT_EVIDENCE.value,
            Verdict.WRONG_REFERENCE.value,
        ]
    )
    matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Format confusion matrix as a markdown table."""
        header = "| Expected \\ Actual | " + " | ".join(self.labels) + " | Total |"
        sep = "|---|" + "|".join(["---" for _ in self.labels]) + "|---|"
        lines = [header, sep]
        for exp in self.labels:
            row_vals = [self.matrix.get(exp, {}).get(act, 0) for act in self.labels]
            row_str = f"| **{exp}** | " + " | ".join(str(v) for v in row_vals) + f" | {sum(row_vals)} |"
            lines.append(row_str)
        return "\n".join(lines)


class EvaluationMetrics(BaseModel):
    """Comprehensive performance and error attribution metrics."""

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0

    verdict_accuracy: float = 0.0
    reference_resolution_accuracy: float = 0.0
    source_availability_accuracy: float = 0.0
    conclusive_classification_accuracy: float = 0.0

    stage_error_counts: Dict[str, int] = Field(default_factory=dict)
    error_taxonomy_counts: Dict[str, int] = Field(default_factory=dict)
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)


class BenchmarkReport(BaseModel):
    """Top-level report containing metrics and individual case evaluations."""

    pipeline_version: str = "1.0.0-stage10c"
    benchmark_version: str = "1.0.0-stage11"
    timestamp: str
    metrics: EvaluationMetrics
    results: List[CaseEvaluationResult]
