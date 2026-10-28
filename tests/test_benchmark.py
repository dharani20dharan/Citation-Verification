"""
Unit and regression tests for Stage 11 Evaluation Benchmark & Error Analysis.

Verifies:
1. Benchmark schema validity
2. Ground truth completeness and validity
3. Benchmark runner execution
4. Metric calculations & confusion matrix correctness
5. Stage-level error attribution & diagnostic taxonomy
6. Deterministic repeated execution
"""

import json
from pathlib import Path
import pytest

from src.models.schemas import Domain
from src.pipeline.models import PipelineClaimResult
from src.verification.models import Verdict
from src.evaluation.models import (
    BenchmarkCase,
    CaseEvaluationResult,
    PipelineStage,
    ErrorType,
    ConfusionMatrix,
    EvaluationMetrics,
    BenchmarkReport,
)
from src.evaluation.evaluator import BenchmarkEvaluator


BENCHMARK_PATH = Path("data/evaluation/benchmark.jsonl")
LEGAL_BENCHMARK_PATH = Path("data/evaluation/legal/legal_benchmark.jsonl")
SCIENTIFIC_BENCHMARK_PATH = Path("data/evaluation/scientific/scientific_benchmark.jsonl")


class TestBenchmarkSchemaAndData:
    """Verifies that the benchmark dataset files comply with the schema and explicit ground truth requirements."""

    def test_benchmark_file_exists(self):
        assert BENCHMARK_PATH.exists(), f"Benchmark file {BENCHMARK_PATH} missing"
        assert LEGAL_BENCHMARK_PATH.exists(), f"Legal benchmark file {LEGAL_BENCHMARK_PATH} missing"
        assert SCIENTIFIC_BENCHMARK_PATH.exists(), f"Scientific benchmark file {SCIENTIFIC_BENCHMARK_PATH} missing"

    def test_benchmark_schema_validation(self):
        evaluator = BenchmarkEvaluator()
        cases = evaluator.load_cases_from_jsonl(BENCHMARK_PATH)
        assert len(cases) >= 10, f"Expected at least 10 benchmark cases, found {len(cases)}"

        legal_cases = [c for c in cases if c.domain == Domain.LEGAL]
        scientific_cases = [c for c in cases if c.domain == Domain.SCIENTIFIC]

        assert len(legal_cases) >= 5, "Expected at least 5 legal benchmark cases"
        assert len(scientific_cases) >= 5, "Expected at least 5 scientific benchmark cases"

        valid_verdicts = {v.value for v in Verdict}

        case_ids = set()
        for case in cases:
            assert case.case_id, "case_id must not be empty"
            assert case.case_id not in case_ids, f"Duplicate case_id {case.case_id}"
            case_ids.add(case.case_id)

            assert case.domain in {Domain.LEGAL, Domain.SCIENTIFIC}, f"Invalid domain: {case.domain}"
            assert case.document_id, f"Document ID missing for {case.case_id}"
            assert case.claim, f"Claim missing for {case.case_id}"
            assert case.reference, f"Reference missing for {case.case_id}"
            assert case.expected_reference_status in {
                "exact_match",
                "not_found",
                "semantic_candidate",
            }, f"Invalid expected_reference_status for {case.case_id}"
            assert case.expected_source_status in {
                "available",
                "unavailable",
                "unresolved_reference",
                "citing_document_only",
                "unrelated_source",
            }, f"Invalid expected_source_status for {case.case_id}"
            assert case.expected_verdict.value in valid_verdicts, (
                f"Invalid expected_verdict '{case.expected_verdict}' for {case.case_id}"
            )
            assert isinstance(case.expected_conclusive, bool)
            assert isinstance(case.expected_evidence_behavior, dict)
            assert case.category, f"Missing category for {case.case_id}"
            assert case.description, f"Missing description for {case.case_id}"


class TestEvaluationMetricsAndConfusionMatrix:
    """Tests the metric calculation and confusion matrix generation."""

    def test_metrics_calculation_perfect_pass(self):
        evaluator = BenchmarkEvaluator()
        dummy_case = BenchmarkCase(
            case_id="case_1",
            domain=Domain.LEGAL,
            document_id="doc_1",
            claim="Novartis upheld Section 3(d)",
            reference="Novartis AG v. Union of India",
            expected_reference_status="exact_match",
            expected_source_status="available",
            expected_verdict=Verdict.SUPPORTED,
            expected_conclusive=True,
            expected_evidence_behavior={"eligible_required": True},
            category="evidence_support",
            description="Test case",
        )

        res = CaseEvaluationResult(
            case_id="case_1",
            domain=Domain.LEGAL,
            category="evidence_support",
            passed=True,
            expected_verdict="supported",
            actual_verdict="supported",
            expected_reference_status="exact_match",
            actual_reference_status="exact_match",
            expected_source_status="available",
            actual_source_status="available",
            expected_conclusive=True,
            actual_conclusive=True,
            confidence=0.9,
        )

        report = evaluator.compile_report([dummy_case], [res])
        assert report.metrics.total_cases == 1
        assert report.metrics.passed_cases == 1
        assert report.metrics.failed_cases == 0
        assert report.metrics.verdict_accuracy == 1.0
        assert report.metrics.reference_resolution_accuracy == 1.0
        assert report.metrics.source_availability_accuracy == 1.0
        assert report.metrics.conclusive_classification_accuracy == 1.0
        assert "supported" in report.metrics.confusion_matrix.labels
        assert report.metrics.confusion_matrix.matrix["supported"]["supported"] == 1

    def test_confusion_matrix_distinct_categories_not_collapsed(self):
        """Ensures unsupported, not_found, insufficient_evidence are preserved separately."""
        evaluator = BenchmarkEvaluator()
        cases = [
            BenchmarkCase(
                case_id="c1",
                domain=Domain.LEGAL,
                document_id="doc1",
                claim="Claim 1",
                reference="Ref 1",
                expected_reference_status="exact_match",
                expected_source_status="available",
                expected_verdict=Verdict.NOT_FOUND,
                expected_conclusive=False,
                expected_evidence_behavior={"eligible_required": False},
                category="reference_correctness",
                description="desc",
            ),
            BenchmarkCase(
                case_id="c2",
                domain=Domain.SCIENTIFIC,
                document_id="doc2",
                claim="Claim 2",
                reference="[1]",
                expected_reference_status="exact_match",
                expected_source_status="available",
                expected_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                expected_conclusive=False,
                expected_evidence_behavior={"eligible_required": False},
                category="evidence_support",
                description="desc",
            ),
        ]

        results = [
            CaseEvaluationResult(
                case_id="c1",
                domain=Domain.LEGAL,
                category="reference_correctness",
                passed=False,
                expected_verdict="not_found",
                actual_verdict="unsupported",
                expected_reference_status="exact_match",
                actual_reference_status="exact_match",
                expected_source_status="available",
                actual_source_status="available",
                expected_conclusive=False,
                actual_conclusive=True,
                confidence=0.8,
            ),
            CaseEvaluationResult(
                case_id="c2",
                domain=Domain.SCIENTIFIC,
                category="evidence_support",
                passed=True,
                expected_verdict="insufficient_evidence",
                actual_verdict="insufficient_evidence",
                expected_reference_status="exact_match",
                actual_reference_status="exact_match",
                expected_source_status="available",
                actual_source_status="available",
                expected_conclusive=False,
                actual_conclusive=False,
                confidence=0.85,
            ),
        ]

        report = evaluator.compile_report(cases, results)
        matrix = report.metrics.confusion_matrix.matrix

        # c1: expected not_found, actual unsupported
        assert matrix["not_found"]["unsupported"] == 1
        # c2: expected insufficient_evidence, actual insufficient_evidence
        assert matrix["insufficient_evidence"]["insufficient_evidence"] == 1
        # unsupported vs not_found should be 0
        assert matrix["unsupported"]["not_found"] == 0


class TestStageLevelErrorAttribution:
    """Verifies that diagnosis attributes failures to the earliest pipeline stage correctly."""

    def test_diagnosis_reference_resolution_failure(self):
        evaluator = BenchmarkEvaluator()
        case = BenchmarkCase(
            case_id="ref_fail_01",
            domain=Domain.LEGAL,
            document_id="doc_1",
            claim="Claim text",
            reference="Section 3(d)",
            expected_reference_status="exact_match",
            expected_source_status="available",
            expected_verdict=Verdict.SUPPORTED,
            expected_conclusive=True,
            expected_evidence_behavior={"eligible_required": True},
            category="reference_correctness",
            description="Reference failure",
        )

        cr = PipelineClaimResult(
            claim_id="ref_fail_01",
            document_id="doc_1",
            domain=Domain.LEGAL,
            claim_text="Claim text",
            reference_text="Section 3(d)",
            normalized_reference="Section 3(d)",
            reference_type="legal_section",
            reference_status="not_found",
            verdict=Verdict.NOT_FOUND,
            confidence=0.9,
            reasoning="Reference not found",
            verifier_version="1.0.0",
            source_attribution={"source_availability": "available"},
        )

        diag = evaluator.diagnose_failure(
            case=case,
            cr=cr,
            ref_status_matches=False,
            source_status_matches=True,
            verdict_matches=False,
            evidence_behavior_matches=True,
            ev_mismatch_reason="",
        )
        assert diag.stage == PipelineStage.REFERENCE_RESOLUTION
        assert ErrorType.WRONG_REFERENCE_RESOLUTION in diag.error_types

    def test_diagnosis_source_attribution_failure(self):
        evaluator = BenchmarkEvaluator()
        case = BenchmarkCase(
            case_id="source_fail_01",
            domain=Domain.SCIENTIFIC,
            document_id="doc_1",
            claim="Claim text",
            reference="[1]",
            expected_reference_status="exact_match",
            expected_source_status="available",
            expected_verdict=Verdict.SUPPORTED,
            expected_conclusive=True,
            expected_evidence_behavior={"eligible_required": True},
            category="source_availability",
            description="Source failure",
        )

        cr = PipelineClaimResult(
            claim_id="source_fail_01",
            document_id="doc_1",
            domain=Domain.SCIENTIFIC,
            claim_text="Claim text",
            reference_text="[1]",
            normalized_reference="1",
            reference_type="numeric",
            reference_status="exact_match",
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            confidence=0.8,
            reasoning="Source unavailable",
            verifier_version="1.0.0",
            source_attribution={"source_availability": "unavailable"},
        )

        diag = evaluator.diagnose_failure(
            case=case,
            cr=cr,
            ref_status_matches=True,
            source_status_matches=False,
            verdict_matches=False,
            evidence_behavior_matches=True,
            ev_mismatch_reason="",
        )
        assert diag.stage == PipelineStage.SOURCE_ATTRIBUTION
        assert ErrorType.MISSED_SOURCE in diag.error_types

    def test_diagnosis_semantic_verification_failure(self):
        evaluator = BenchmarkEvaluator()
        case = BenchmarkCase(
            case_id="verdict_fail_01",
            domain=Domain.LEGAL,
            document_id="doc_1",
            claim="Claim text",
            reference="Section 3(d)",
            expected_reference_status="exact_match",
            expected_source_status="available",
            expected_verdict=Verdict.UNSUPPORTED,
            expected_conclusive=True,
            expected_evidence_behavior={"eligible_required": True},
            category="evidence_support",
            description="Semantic failure",
        )

        cr = PipelineClaimResult(
            claim_id="verdict_fail_01",
            document_id="doc_1",
            domain=Domain.LEGAL,
            claim_text="Claim text",
            reference_text="Section 3(d)",
            normalized_reference="Section 3(d)",
            reference_type="legal_section",
            reference_status="exact_match",
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            reasoning="Claim supported",
            verifier_version="1.0.0",
            source_attribution={"source_availability": "available"},
        )

        diag = evaluator.diagnose_failure(
            case=case,
            cr=cr,
            ref_status_matches=True,
            source_status_matches=True,
            verdict_matches=False,
            evidence_behavior_matches=True,
            ev_mismatch_reason="",
        )
        assert diag.stage == PipelineStage.SEMANTIC_VERIFICATION
        assert ErrorType.CONTRADICTION_CLASSIFICATION_ERROR in diag.error_types


class TestBenchmarkExecutionAndDeterminism:
    """Verifies that benchmark cases execute through the pipeline and produce deterministic results."""

    def test_benchmark_subset_execution_and_determinism(self):
        evaluator = BenchmarkEvaluator()
        cases = evaluator.load_cases_from_jsonl(BENCHMARK_PATH)

        # Run a representative subset (first 2 cases covering legal and scientific)
        test_cases = cases[:2]

        report_1 = evaluator.evaluate_suite(test_cases)
        report_2 = evaluator.evaluate_suite(test_cases)

        # Verify exact same ordering
        assert [r.case_id for r in report_1.results] == [r.case_id for r in report_2.results]

        # Verify determinism across all fields
        for r1, r2 in zip(report_1.results, report_2.results):
            assert r1.case_id == r2.case_id
            assert r1.passed == r2.passed
            assert r1.actual_verdict == r2.actual_verdict
            assert r1.actual_reference_status == r2.actual_reference_status
            assert r1.actual_source_status == r2.actual_source_status
            assert r1.actual_conclusive == r2.actual_conclusive
            assert r1.evidence_ids == r2.evidence_ids

        # Metrics must match exactly
        assert report_1.metrics.total_cases == report_2.metrics.total_cases
        assert report_1.metrics.passed_cases == report_2.metrics.passed_cases
        assert report_1.metrics.failed_cases == report_2.metrics.failed_cases
        assert report_1.metrics.verdict_accuracy == report_2.metrics.verdict_accuracy
        assert report_1.metrics.reference_resolution_accuracy == report_2.metrics.reference_resolution_accuracy
        assert report_1.metrics.source_availability_accuracy == report_2.metrics.source_availability_accuracy
