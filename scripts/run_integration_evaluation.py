"""End-to-End Integration Evaluation Script for Stage 10C.

Executes canonical test corpus scenarios across legal and scientific domains,
validating complete verification chains from document/claim to final auditable verdict.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain, EvidenceChunk
from src.pipeline.models import PipelineConfiguration, PipelineResult
from src.pipeline.orchestrator import VerificationPipelineOrchestrator
from src.verification.llm_verifier import MockSemanticVerifier
from src.verification.models import Verdict

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("integration_evaluation")


def load_chunks_from_jsonl(path: Path) -> List[EvidenceChunk]:
    """Load EvidenceChunk objects from a JSONL file."""
    chunks: List[EvidenceChunk] = []
    if not path.exists():
        return chunks
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(EvidenceChunk.model_validate(json.loads(line)))
    return chunks


def build_evaluation_scenarios() -> List[Dict[str, Any]]:
    """Build the suite of canonical evaluation test scenarios."""
    scenarios = []

    # Scenario 1: Legal Supported Citation - Section 3(d)
    legal_chunks = load_chunks_from_jsonl(PROJECT_ROOT / "data" / "chunks" / "legal" / "legal_sample_fixture.jsonl")
    scenarios.append({
        "scenario_id": "legal_section_3d_supported",
        "name": "Legal Supported Citation (Section 3(d))",
        "domain": Domain.LEGAL,
        "input": legal_chunks,
        "document_id": "legal_sample_fixture",
        "config": PipelineConfiguration(verifier_mode="deterministic"),
    })

    # Scenario 2: Legal Wrong/Nonexistent Reference - Section 9
    scenarios.append({
        "scenario_id": "legal_section_9_not_found",
        "name": "Legal Nonexistent Reference (Section 9)",
        "domain": Domain.LEGAL,
        "input": [
            CitationClaim(
                claim_id="eval_legal_sec_9_claim",
                document_id="legal_eval_doc",
                domain=Domain.LEGAL,
                claim_text="Section 9 excludes certain provisional specifications from examination.",
                reference_text="Section 9",
                normalized_reference="9",
                normalized_references=["9"],
                reference_type=ReferenceType.LEGAL_SECTION.value,
            )
        ],
        "document_id": "legal_eval_doc",
        "config": PipelineConfiguration(verifier_mode="deterministic"),
    })

    # Scenario 3: Scientific Citation with Resolved Bibliography but Unavailable Source - [12]
    sci_chunks = load_chunks_from_jsonl(PROJECT_ROOT / "data" / "chunks" / "scientific" / "scientific_citations_fixture.jsonl")
    # Claim 1 in sci_chunks is [12]
    scenarios.append({
        "scenario_id": "sci_ref_12_source_unavailable",
        "name": "Scientific Resolved Reference with Source Unavailable ([12])",
        "domain": Domain.SCIENTIFIC,
        "input": [
            CitationClaim(
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
        ],
        "document_id": "sci_eval_01",
        "config": PipelineConfiguration(verifier_mode="deterministic"),
    })

    # Scenario 4: Scientific Citation with Available Source & Supporting Evidence - (Smith et al., 2024)
    scenarios.append({
        "scenario_id": "sci_smith_2024_supported",
        "name": "Scientific Available Source with Supporting Evidence (Smith et al., 2024)",
        "domain": Domain.SCIENTIFIC,
        "input": [
            CitationClaim(
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
        ],
        "document_id": "sci_eval_01",
        "config": PipelineConfiguration(verifier_mode="deterministic"),
    })

    # Scenario 5: Scientific Contradiction (source available, contradictory claim)
    scenarios.append({
        "scenario_id": "sci_smith_2024_unsupported",
        "name": "Scientific Contradiction Against Verified Source",
        "domain": Domain.SCIENTIFIC,
        "input": [
            CitationClaim(
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
        ],
        "document_id": "sci_eval_01",
        "config": PipelineConfiguration(
            verifier_mode="mock",
            custom_mock_response={
                "verdict": "unsupported",
                "confidence": 0.92,
                "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
                "reasoning": "The evidence directly refutes the claim: it demonstrates reduced latency and improved precision.",
            },
        ),
    })

    # Scenario 6: Partially Supported Compound Claim
    scenarios.append({
        "scenario_id": "sci_smith_2024_partial",
        "name": "Scientific Partially Supported Compound Claim",
        "domain": Domain.SCIENTIFIC,
        "input": [
            CitationClaim(
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
        ],
        "document_id": "sci_eval_01",
        "config": PipelineConfiguration(
            verifier_mode="mock",
            custom_mock_response={
                "verdict": "partially_correct",
                "confidence": 0.88,
                "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
                "reasoning": "Reduced latency is explicitly corroborated, but zero memory overhead is unmentioned.",
            },
        ),
    })

    # Scenario 7: Insufficient Evidence Scenario
    scenarios.append({
        "scenario_id": "sci_smith_2024_insufficient",
        "name": "Scientific Tangential Evidence (Insufficient Evidence)",
        "domain": Domain.SCIENTIFIC,
        "input": [
            CitationClaim(
                claim_id="sci_eval_insufficient_claim",
                document_id="sci_eval_01",
                domain=Domain.SCIENTIFIC,
                claim_text="The model architecture was trained on multimodal biomedical imagery datasets",
                reference_text="(Smith et al., 2024)",
                normalized_reference="Smith et al., 2024",
                normalized_references=["Smith et al., 2024"],
                reference_type=ReferenceType.AUTHOR_YEAR.value,
                page=2,
            )
        ],
        "document_id": "sci_eval_01",
        "config": PipelineConfiguration(
            verifier_mode="mock",
            custom_mock_response={
                "verdict": "insufficient_evidence",
                "confidence": 0.70,
                "evidence_ids": ["sci_doc_smith_2024_chunk_001"],
                "reasoning": "The empirical findings discuss latency and precision in citation verification, containing no evidence about multimodal imagery training.",
            },
        ),
    })

    return scenarios


def run_evaluation() -> None:
    """Execute evaluation across all canonical scenarios and output auditable report."""
    orchestrator = VerificationPipelineOrchestrator()
    scenarios = build_evaluation_scenarios()

    print("=" * 80)
    print("STAGE 10C: END-TO-END INTEGRATION & EVALUATION REPORT")
    print("=" * 80)
    print(f"Pipeline Version : {VerificationPipelineOrchestrator.VERSION}")
    print(f"Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Scenarios  : {len(scenarios)}")
    print("=" * 80)

    total_claims = 0
    verdict_distribution: Dict[str, int] = {}
    reference_status_distribution: Dict[str, int] = {}
    source_availability_distribution: Dict[str, int] = {}
    eligible_evidence_total = 0
    ineligible_evidence_total = 0
    pipeline_errors = 0
    failed_cases = []
    claim_records = []

    scenario_results = []
    for sc in scenarios:
        t_start = time.perf_counter()
        result: PipelineResult = orchestrator.run(
            input_document=sc["input"],
            domain=sc["domain"],
            config=sc["config"],
            document_id=sc["document_id"],
        )
        elapsed = time.perf_counter() - t_start
        scenario_results.append((sc, result, elapsed))

        if result.errors:
            pipeline_errors += len(result.errors)
            failed_cases.append(f"{sc['scenario_id']}: {result.errors}")

        for cl in result.claims:
            total_claims += 1
            v = cl.verdict.value
            verdict_distribution[v] = verdict_distribution.get(v, 0) + 1

            r = cl.reference_status
            reference_status_distribution[r] = reference_status_distribution.get(r, 0) + 1

            s = cl.source_attribution.get("source_availability", "unknown")
            source_availability_distribution[s] = source_availability_distribution.get(s, 0) + 1

            for ev in cl.selected_evidence:
                if getattr(ev, "is_eligible", True):
                    eligible_evidence_total += 1
                else:
                    ineligible_evidence_total += 1

            claim_records.append(cl)

    # 1. High-Level Summary
    print("\n1. EXECUTION SUMMARY")
    print("-" * 40)
    print(f"Total Documents/Scenarios Evaluated : {len(scenarios)}")
    print(f"Total Claims Evaluated              : {total_claims}")
    print(f"Pipeline Errors / Unhandled Crashes : {pipeline_errors}")

    # 2. Reference Resolution Results
    print("\n2. REFERENCE RESOLUTION RESULTS")
    print("-" * 40)
    for status, count in sorted(reference_status_distribution.items()):
        print(f"  {status:<22} : {count}")

    # 3. Source Availability Results
    print("\n3. SOURCE AVAILABILITY RESULTS")
    print("-" * 40)
    for status, count in sorted(source_availability_distribution.items()):
        print(f"  {status:<22} : {count}")

    # 4. Verdict Distribution
    print("\n4. VERDICT DISTRIBUTION")
    print("-" * 40)
    for verdict, count in sorted(verdict_distribution.items()):
        print(f"  {verdict:<22} : {count}")

    # 5. Evidence Eligibility Statistics
    print("\n5. EVIDENCE ELIGIBILITY STATISTICS")
    print("-" * 40)
    print(f"  Eligible Substantive Evidence Chunks   : {eligible_evidence_total}")
    print(f"  Ineligible / Heading / Unrelated Chunks : {ineligible_evidence_total}")

    # 6. Detailed Claim Traceability Table
    print("\n6. DETAILED CLAIM VERIFICATION TRACE")
    print("-" * 120)
    header = f"{'CLAIM ID':<28} | {'REFERENCE':<16} | {'REF STATUS':<16} | {'SOURCE STATUS':<14} | {'VERDICT':<21} | {'CONF':<5} | {'EVID IDs'}"
    print(header)
    print("-" * 120)

    for cl in claim_records:
        ev_ids_str = ", ".join(e.chunk_id for e in cl.selected_evidence[:2])
        if len(cl.selected_evidence) > 2:
            ev_ids_str += f" (+{len(cl.selected_evidence) - 2} more)"
        if not ev_ids_str:
            ev_ids_str = "(none)"

        row = (
            f"{cl.claim_id:<28} | "
            f"{cl.reference_text:<16} | "
            f"{cl.reference_status:<16} | "
            f"{cl.source_attribution.get('source_availability', 'unknown'):<14} | "
            f"{cl.verdict.value:<21} | "
            f"{cl.confidence:<5.2f} | "
            f"{ev_ids_str}"
        )
        print(row)

    print("-" * 120)

    # 7. Scenario Breakdown
    print("\n7. SCENARIO OUTCOMES")
    print("-" * 80)
    for sc, result, elapsed in scenario_results:
        status_tag = "[PASS]" if result.processing_status.value == "success" else "[FAIL]"
        claim_verdicts = [c.verdict.value for c in result.claims]
        print(f"{status_tag} {sc['name']:<55} | claims: {len(result.claims)} | verdicts: {claim_verdicts} | time: {elapsed:.2f}s")

    print("\n" + "=" * 80)
    print("END OF EVALUATION REPORT")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
