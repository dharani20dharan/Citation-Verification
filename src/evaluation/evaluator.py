"""Benchmark Evaluator running test cases through Stage 10C pipeline and diagnosing failures."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain
from src.pipeline.models import PipelineClaimResult, PipelineConfiguration, PipelineResult
from src.pipeline.orchestrator import VerificationPipelineOrchestrator
from src.verification.models import Verdict
from src.evaluation.models import (
    BenchmarkCase,
    BenchmarkReport,
    CaseEvaluationResult,
    ConfusionMatrix,
    ErrorType,
    EvaluationMetrics,
    PipelineStage,
    StageFailureDiagnosis,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates the Stage 10C pipeline on benchmark cases with strict ground truth."""

    def __init__(self, orchestrator: Optional[VerificationPipelineOrchestrator] = None):
        self.orchestrator = orchestrator or VerificationPipelineOrchestrator()

    @staticmethod
    def load_cases_from_jsonl(path: Path) -> List[BenchmarkCase]:
        """Load benchmark cases from a JSONL file."""
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cases.append(BenchmarkCase.model_validate(data))
        return cases

    def evaluate_case(self, case: BenchmarkCase) -> CaseEvaluationResult:
        """Run a single benchmark case through the pipeline and evaluate against ground truth."""
        norm_ref = case.normalized_reference or case.reference.replace("[", "").replace("]", "").replace("Section", "").strip()
        ref_type = case.reference_type or (
            ReferenceType.LEGAL_SECTION.value if case.domain == Domain.LEGAL else ReferenceType.NUMERIC.value
        )

        claim = CitationClaim(
            claim_id=case.case_id,
            document_id=case.document_id,
            domain=case.domain,
            claim_text=case.claim,
            reference_text=case.reference,
            normalized_reference=norm_ref,
            normalized_references=[norm_ref],
            reference_type=ref_type,
        )

        # Configure pipeline
        verifier_mode = "mock" if case.mock_response is not None else "deterministic"
        config = PipelineConfiguration(
            verifier_mode=verifier_mode,
            custom_mock_response=case.mock_response,
        )

        # Run pipeline
        res: PipelineResult = self.orchestrator.run(
            input_document=[claim],
            domain=case.domain,
            config=config,
            document_id=case.document_id,
        )

        if not res.claims:
            # Complete execution failure
            diagnosis = StageFailureDiagnosis(
                stage=PipelineStage.RESULT_COMPILATION,
                error_types=[ErrorType.PROVENANCE_LOSS],
                reason=f"Pipeline produced 0 claim results for case {case.case_id}. Errors: {res.errors}",
            )
            return CaseEvaluationResult(
                case_id=case.case_id,
                domain=case.domain,
                category=case.category,
                passed=False,
                expected_verdict=case.expected_verdict.value,
                actual_verdict="none",
                expected_reference_status=case.expected_reference_status,
                actual_reference_status="none",
                expected_source_status=case.expected_source_status,
                actual_source_status="none",
                expected_conclusive=case.expected_conclusive,
                actual_conclusive=False,
                confidence=0.0,
                evidence_ids=[],
                eligible_evidence_ids=[],
                evidence_trace={},
                failure_diagnosis=diagnosis,
                reasoning="No claim results produced",
            )

        cr: PipelineClaimResult = res.claims[0]
        actual_ref_status = cr.reference_status
        actual_source_status = cr.source_attribution.get("source_availability", "unknown")
        actual_verdict = cr.verdict.value
        actual_conclusive = cr.is_conclusive

        selected_ev_ids = [e.chunk_id for e in cr.selected_evidence]
        eligible_ev_ids = [e.chunk_id for e in cr.selected_evidence if getattr(e, "is_eligible", True)]

        # Complete evidence trace
        evidence_trace = {
            "claim": cr.claim_text,
            "reference": cr.reference_text,
            "reference_resolution": cr.reference_resolution,
            "retrieved_candidates": cr.retrieval_candidates,
            "eligible_candidates": eligible_ev_ids,
            "selected_evidence": selected_ev_ids,
            "source_attribution": cr.source_attribution,
            "final_verdict": actual_verdict,
            "reasoning": cr.reasoning,
        }

        # Check facet matches
        ref_status_matches = (actual_ref_status == case.expected_reference_status)
        source_status_matches = (actual_source_status == case.expected_source_status)
        verdict_matches = (cr.verdict == case.expected_verdict)
        conclusive_matches = (actual_conclusive == case.expected_conclusive)

        # Check evidence behavior expectations
        evidence_behavior_matches = True
        ev_mismatch_reason = ""
        exp_behav = case.expected_evidence_behavior

        if exp_behav.get("prohibited_chunks"):
            for p_chunk in exp_behav["prohibited_chunks"]:
                if p_chunk in eligible_ev_ids:
                    evidence_behavior_matches = False
                    ev_mismatch_reason = f"Prohibited chunk '{p_chunk}' was incorrectly marked eligible"
                    break

        if exp_behav.get("required_chunks"):
            for r_chunk in exp_behav["required_chunks"]:
                if r_chunk not in selected_ev_ids:
                    evidence_behavior_matches = False
                    ev_mismatch_reason = f"Required chunk '{r_chunk}' was not selected in evidence"
                    break

        if exp_behav.get("eligible_required", False) and not eligible_ev_ids and case.expected_verdict == Verdict.SUPPORTED:
            evidence_behavior_matches = False
            ev_mismatch_reason = "Supported verdict requires at least one eligible substantive evidence chunk"

        overall_passed = (
            ref_status_matches
            and source_status_matches
            and verdict_matches
            and conclusive_matches
            and evidence_behavior_matches
        )

        diagnosis: Optional[StageFailureDiagnosis] = None
        if not overall_passed:
            diagnosis = self.diagnose_failure(
                case=case,
                cr=cr,
                ref_status_matches=ref_status_matches,
                source_status_matches=source_status_matches,
                verdict_matches=verdict_matches,
                evidence_behavior_matches=evidence_behavior_matches,
                ev_mismatch_reason=ev_mismatch_reason,
            )

        return CaseEvaluationResult(
            case_id=case.case_id,
            domain=case.domain,
            category=case.category,
            passed=overall_passed,
            expected_verdict=case.expected_verdict.value,
            actual_verdict=actual_verdict,
            expected_reference_status=case.expected_reference_status,
            actual_reference_status=actual_ref_status,
            expected_source_status=case.expected_source_status,
            actual_source_status=actual_source_status,
            expected_conclusive=case.expected_conclusive,
            actual_conclusive=actual_conclusive,
            confidence=cr.confidence,
            evidence_ids=selected_ev_ids,
            eligible_evidence_ids=eligible_ev_ids,
            evidence_trace=evidence_trace,
            failure_diagnosis=diagnosis,
            reasoning=cr.reasoning,
        )

    def diagnose_failure(
        self,
        case: BenchmarkCase,
        cr: PipelineClaimResult,
        ref_status_matches: bool,
        source_status_matches: bool,
        verdict_matches: bool,
        evidence_behavior_matches: bool,
        ev_mismatch_reason: str,
    ) -> StageFailureDiagnosis:
        """Trace earliest failure stage and assign error taxonomy tags."""
        # 1. Did reference resolution fail?
        if not ref_status_matches:
            return StageFailureDiagnosis(
                stage=PipelineStage.REFERENCE_RESOLUTION,
                error_types=[ErrorType.WRONG_REFERENCE_RESOLUTION],
                reason=(
                    f"Reference status diverged: expected '{case.expected_reference_status}', "
                    f"got '{cr.reference_status}'"
                ),
                details={"reference_resolution": cr.reference_resolution},
            )

        # 2. Did source attribution fail?
        if not source_status_matches:
            return StageFailureDiagnosis(
                stage=PipelineStage.SOURCE_ATTRIBUTION,
                error_types=[ErrorType.MISSED_SOURCE if case.expected_source_status == "available" else ErrorType.WRONG_SOURCE],
                reason=(
                    f"Source status diverged: expected '{case.expected_source_status}', "
                    f"got '{cr.source_attribution.get('source_availability')}'"
                ),
                details={"source_attribution": cr.source_attribution},
            )

        # 3. Did evidence selection / eligibility fail?
        if not evidence_behavior_matches:
            err_types = []
            if "Prohibited" in ev_mismatch_reason or "structural" in ev_mismatch_reason.lower():
                err_types.append(ErrorType.STRUCTURAL_EVIDENCE_LEAK)
                err_types.append(ErrorType.INELIGIBLE_EVIDENCE_SELECTED)
            elif "Required" in ev_mismatch_reason:
                err_types.append(ErrorType.RELEVANT_EVIDENCE_MISSED)
            else:
                err_types.append(ErrorType.INELIGIBLE_EVIDENCE_SELECTED)

            return StageFailureDiagnosis(
                stage=PipelineStage.EVIDENCE_SELECTION,
                error_types=err_types,
                reason=ev_mismatch_reason or "Evidence selection behavior diverged from expectation",
                details={
                    "selected_evidence": [e.chunk_id for e in cr.selected_evidence],
                    "eligible_evidence": [e.chunk_id for e in cr.selected_evidence if getattr(e, "is_eligible", True)],
                },
            )

        # 4. Did semantic verification fail?
        if not verdict_matches:
            err_type = ErrorType.SUPPORT_CLASSIFICATION_ERROR
            if case.expected_verdict == Verdict.UNSUPPORTED:
                err_type = ErrorType.CONTRADICTION_CLASSIFICATION_ERROR
            elif case.expected_verdict == Verdict.PARTIALLY_CORRECT:
                err_type = ErrorType.PARTIAL_SUPPORT_CLASSIFICATION_ERROR
            elif case.expected_verdict == Verdict.INSUFFICIENT_EVIDENCE:
                err_type = ErrorType.INSUFFICIENT_EVIDENCE_CLASSIFICATION_ERROR

            return StageFailureDiagnosis(
                stage=PipelineStage.SEMANTIC_VERIFICATION,
                error_types=[err_type],
                reason=f"Verdict diverged: expected '{case.expected_verdict.value}', got '{cr.verdict.value}'",
                details={"reasoning": cr.reasoning, "confidence": cr.confidence},
            )

        # Fallback
        return StageFailureDiagnosis(
            stage=PipelineStage.RESULT_COMPILATION,
            error_types=[ErrorType.PROVENANCE_LOSS],
            reason="Unclassified discrepancy",
        )

    def compile_report(self, cases: List[BenchmarkCase], results: List[CaseEvaluationResult]) -> BenchmarkReport:
        """Compile metrics, confusion matrix, and report from evaluation results."""
        matrix_data: Dict[str, Dict[str, int]] = {}

        # Initialize confusion matrix rows
        all_verdicts = [v.value for v in Verdict]
        for exp in all_verdicts:
            matrix_data[exp] = {act: 0 for act in all_verdicts}

        ref_matches = 0
        src_matches = 0
        verdict_matches = 0
        conclusive_matches = 0
        passed_count = 0

        stage_error_counts: Dict[str, int] = {}
        error_taxonomy_counts: Dict[str, int] = {}

        for case, res in zip(cases, results):
            exp_v = case.expected_verdict.value
            act_v = res.actual_verdict
            if exp_v in matrix_data and act_v in matrix_data[exp_v]:
                matrix_data[exp_v][act_v] += 1

            if res.actual_reference_status == case.expected_reference_status:
                ref_matches += 1
            if res.actual_source_status == case.expected_source_status:
                src_matches += 1
            if res.actual_verdict == case.expected_verdict.value:
                verdict_matches += 1
            if res.actual_conclusive == case.expected_conclusive:
                conclusive_matches += 1

            if res.passed:
                passed_count += 1
            else:
                if res.failure_diagnosis:
                    stg = res.failure_diagnosis.stage.value
                    stage_error_counts[stg] = stage_error_counts.get(stg, 0) + 1
                    for et in res.failure_diagnosis.error_types:
                        et_val = et.value
                        error_taxonomy_counts[et_val] = error_taxonomy_counts.get(et_val, 0) + 1

        total = len(cases)
        failed_count = total - passed_count
        metrics = EvaluationMetrics(
            total_cases=total,
            passed_cases=passed_count,
            failed_cases=failed_count,
            pass_rate=round(passed_count / total, 4) if total > 0 else 0.0,
            verdict_accuracy=round(verdict_matches / total, 4) if total > 0 else 0.0,
            reference_resolution_accuracy=round(ref_matches / total, 4) if total > 0 else 0.0,
            source_availability_accuracy=round(src_matches / total, 4) if total > 0 else 0.0,
            conclusive_classification_accuracy=round(conclusive_matches / total, 4) if total > 0 else 0.0,
            stage_error_counts=stage_error_counts,
            error_taxonomy_counts=error_taxonomy_counts,
            confusion_matrix=ConfusionMatrix(matrix=matrix_data),
        )

        return BenchmarkReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            results=results,
        )

    def evaluate_suite(self, cases: List[BenchmarkCase]) -> BenchmarkReport:
        """Run the full benchmark suite and compile metrics, confusion matrix, and reports."""
        results: List[CaseEvaluationResult] = []
        for case in cases:
            res = self.evaluate_case(case)
            results.append(res)
        return self.compile_report(cases, results)
