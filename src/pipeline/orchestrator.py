"""Integration Orchestrator executing the complete automated citation verification pipeline."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from src.chunking.legal_chunker import LegalChunker
from src.chunking.scientific_chunker import ScientificChunker
from src.extraction.extractor import ClaimCitationExtractor
from src.extraction.models import CitationClaim
from src.models.schemas import Domain, EvidenceChunk, DocumentParsingResult
from src.parsing.docling_parser import (
    DoclingDocumentParser,
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedFormatError,
)
from src.pipeline.models import (
    PipelineClaimResult,
    PipelineConfiguration,
    PipelineResult,
    PipelineStage,
    PipelineSummary,
    ProcessingStatus,
    StageStatus,
)
from src.references.reference_detector import ReferenceSectionDetector
from src.references.reference_parser import BibliographicParser
from src.references.resolution_engine import ReferenceResolutionEngine
from src.retrieval.retriever import Retriever
from src.verification.evidence_selector import EvidenceSelector
from src.verification.llm_verifier import (
    MockSemanticVerifier,
    OpenAISemanticVerifier,
)
from src.verification.models import EvidenceItem, Verdict, VerificationResult
from src.verification.source_verifier import SourceIdentityVerifier
from src.verification.verification_engine import VerificationEngine
from src.verification.verifier import (
    BaseClaimVerifier,
    DeterministicClaimVerifier,
)

logger = logging.getLogger(__name__)


class VerificationPipelineOrchestrator:
    """End-to-end integration orchestrator coordinating the verification pipeline:

    Document
      ↓
    Parsing (DoclingDocumentParser)
      ↓
    Domain Chunking (LegalChunker / ScientificChunker)
      ↓
    Citation & Claim Extraction (ClaimCitationExtractor)
      ↓
    Reference Detection & Resolution (ReferenceResolutionEngine)
      ↓
    Evidence Retrieval (Retriever)
      ↓
    Source Attribution (SourceIdentityVerifier)
      ↓
    Evidence Selection (EvidenceSelector)
      ↓
    Semantic Verification (VerificationEngine / ClaimVerifiers)
      ↓
    Auditable PipelineResult
    """

    VERSION = "1.0.0-stage10c"

    def __init__(
        self,
        parser: Optional[DoclingDocumentParser] = None,
        legal_chunker: Optional[LegalChunker] = None,
        scientific_chunker: Optional[ScientificChunker] = None,
        extractor: Optional[ClaimCitationExtractor] = None,
        reference_resolver: Optional[ReferenceResolutionEngine] = None,
        retriever: Optional[Retriever] = None,
        source_verifier: Optional[SourceIdentityVerifier] = None,
        evidence_selector: Optional[EvidenceSelector] = None,
        default_verifier: Optional[BaseClaimVerifier] = None,
    ):
        self.parser = parser or DoclingDocumentParser()
        self.legal_chunker = legal_chunker or LegalChunker()
        self.scientific_chunker = scientific_chunker or ScientificChunker()
        self.extractor = extractor or ClaimCitationExtractor()
        self.reference_resolver = reference_resolver or ReferenceResolutionEngine()
        self.retriever = retriever or Retriever()
        self.source_verifier = source_verifier or SourceIdentityVerifier()
        self.evidence_selector = evidence_selector or EvidenceSelector()
        self.default_verifier = default_verifier or DeterministicClaimVerifier()

    def run(
        self,
        input_document: Union[str, Path, DocumentParsingResult, List[EvidenceChunk], List[CitationClaim]],
        domain: Union[Domain, str],
        config: Optional[PipelineConfiguration] = None,
        document_id: Optional[str] = None,
        custom_verifier: Optional[BaseClaimVerifier] = None,
    ) -> PipelineResult:
        """Execute the end-to-end verification pipeline.

        Args:
            input_document: File path, pre-parsed DocumentParsingResult, pre-computed EvidenceChunks,
                            or pre-extracted CitationClaims.
            domain: Target domain ('legal' or 'scientific').
            config: Pipeline configuration settings.
            document_id: Optional explicit document identifier.
            custom_verifier: Optional verifier override for testing or specific evaluation.

        Returns:
            Auditable, deterministic PipelineResult.
        """
        config = config or PipelineConfiguration()
        domain_enum = Domain(domain) if isinstance(domain, str) else domain

        stage_status: Dict[str, str] = {
            PipelineStage.PARSING.value: StageStatus.NOT_STARTED.value,
            PipelineStage.CHUNKING.value: StageStatus.NOT_STARTED.value,
            PipelineStage.EXTRACTION.value: StageStatus.NOT_STARTED.value,
            PipelineStage.REFERENCE_RESOLUTION.value: StageStatus.NOT_STARTED.value,
            PipelineStage.RETRIEVAL.value: StageStatus.NOT_STARTED.value,
            PipelineStage.SOURCE_ATTRIBUTION.value: StageStatus.NOT_STARTED.value,
            PipelineStage.EVIDENCE_SELECTION.value: StageStatus.NOT_STARTED.value,
            PipelineStage.VERIFICATION.value: StageStatus.NOT_STARTED.value,
        }
        timings: Dict[str, float] = {}
        errors: List[str] = []
        warnings: List[str] = []
        overall_start = time.perf_counter()

        doc_id = document_id or "unknown_doc"
        parsed_doc: Optional[DocumentParsingResult] = None
        chunks: List[EvidenceChunk] = []
        claims: List[CitationClaim] = []

        # =====================================================================
        # Step 1: Parsing
        # =====================================================================
        t0 = time.perf_counter()
        if isinstance(input_document, (str, Path)):
            input_path = Path(input_document)
            try:
                parsed_doc = self.parser.parse(input_path, domain=domain_enum)
                doc_id = parsed_doc.document_id
                stage_status[PipelineStage.PARSING.value] = StageStatus.COMPLETED.value
            except (DocumentNotFoundError, UnsupportedFormatError, DocumentParsingError) as err:
                msg = f"Document parsing failed: {err}"
                logger.error(msg)
                stage_status[PipelineStage.PARSING.value] = StageStatus.FAILED.value
                errors.append(msg)
                timings["total"] = time.perf_counter() - overall_start
                return PipelineResult(
                    document_id=doc_id,
                    domain=domain_enum,
                    pipeline_version=self.VERSION,
                    processing_status=ProcessingStatus.FAILED,
                    stage_status=stage_status,
                    timings=timings,
                    errors=errors,
                    warnings=warnings,
                )
        elif isinstance(input_document, DocumentParsingResult):
            parsed_doc = input_document
            doc_id = parsed_doc.document_id
            stage_status[PipelineStage.PARSING.value] = StageStatus.COMPLETED.value
        elif isinstance(input_document, list) and input_document and isinstance(input_document[0], EvidenceChunk):
            chunks = input_document
            doc_id = chunks[0].document_id
            stage_status[PipelineStage.PARSING.value] = StageStatus.SKIPPED.value
            stage_status[PipelineStage.CHUNKING.value] = StageStatus.COMPLETED.value
        elif isinstance(input_document, list) and input_document and isinstance(input_document[0], CitationClaim):
            claims = input_document
            doc_id = claims[0].document_id
            stage_status[PipelineStage.PARSING.value] = StageStatus.SKIPPED.value
            stage_status[PipelineStage.CHUNKING.value] = StageStatus.SKIPPED.value
            stage_status[PipelineStage.EXTRACTION.value] = StageStatus.COMPLETED.value
        timings[PipelineStage.PARSING.value] = time.perf_counter() - t0

        # =====================================================================
        # Step 2: Domain Chunking
        # =====================================================================
        if parsed_doc is not None and not chunks:
            t0 = time.perf_counter()
            try:
                if domain_enum == Domain.LEGAL:
                    chunks = self.legal_chunker.chunk(parsed_doc)
                else:
                    chunks = self.scientific_chunker.chunk(parsed_doc)
                stage_status[PipelineStage.CHUNKING.value] = StageStatus.COMPLETED.value
            except Exception as err:
                msg = f"Chunking failed: {err}"
                logger.error(msg)
                stage_status[PipelineStage.CHUNKING.value] = StageStatus.FAILED.value
                errors.append(msg)
                timings["total"] = time.perf_counter() - overall_start
                return PipelineResult(
                    document_id=doc_id,
                    domain=domain_enum,
                    pipeline_version=self.VERSION,
                    processing_status=ProcessingStatus.FAILED,
                    stage_status=stage_status,
                    timings=timings,
                    errors=errors,
                    warnings=warnings,
                )
            timings[PipelineStage.CHUNKING.value] = time.perf_counter() - t0

        # =====================================================================
        # Step 3: Citation & Claim Extraction
        # =====================================================================
        if chunks and not claims:
            t0 = time.perf_counter()
            try:
                claims = self.extractor.extract_from_chunks(chunks)
                stage_status[PipelineStage.EXTRACTION.value] = StageStatus.COMPLETED.value
            except Exception as err:
                msg = f"Claim extraction failed: {err}"
                logger.error(msg)
                stage_status[PipelineStage.EXTRACTION.value] = StageStatus.FAILED.value
                errors.append(msg)
                timings["total"] = time.perf_counter() - overall_start
                return PipelineResult(
                    document_id=doc_id,
                    domain=domain_enum,
                    pipeline_version=self.VERSION,
                    processing_status=ProcessingStatus.FAILED,
                    stage_status=stage_status,
                    timings=timings,
                    errors=errors,
                    warnings=warnings,
                )
            timings[PipelineStage.EXTRACTION.value] = time.perf_counter() - t0

        # Ingest document bibliography if present in scientific chunks
        if domain_enum == Domain.SCIENTIFIC and chunks:
            try:
                sec_detector = ReferenceSectionDetector()
                ref_parser = BibliographicParser()
                ref_chunks = sec_detector.detect_reference_chunks(chunks)
                for rc in ref_chunks:
                    entries = ref_parser.parse_reference_chunk(rc)
                    self.reference_resolver.add_entries(entries)
            except Exception as e:
                warnings.append(f"Inline reference section indexing notice: {e}")

        # =====================================================================
        # Step 4-8: Verification Engine Coordination
        # =====================================================================
        # Determine verifier to use
        verifier = custom_verifier
        if verifier is None:
            if config.verifier_mode == "mock":
                verifier = MockSemanticVerifier(response=config.custom_mock_response)
            elif config.verifier_mode == "llm":
                verifier = OpenAISemanticVerifier()
            else:
                verifier = self.default_verifier

        verif_engine = VerificationEngine(
            retriever=self.retriever,
            evidence_selector=self.evidence_selector,
            verifier=verifier,
            reference_resolver=self.reference_resolver,
            source_verifier=self.source_verifier,
            top_k=config.top_k,
            max_evidence=config.max_evidence,
        )

        t_verif_start = time.perf_counter()
        claim_results: List[PipelineClaimResult] = []
        has_claim_failures = False

        for claim in claims:
            try:
                res: VerificationResult = verif_engine.verify(claim)

                # Extract audit trace elements
                prov = res.provenance or {}
                bib_res = prov.get("bibliographic_resolution", {})
                source_avail = prov.get("source_availability", "unknown")

                # Build retrieval candidates summary from provenance
                retrieval_summary = [
                    {
                        "chunk_id": e.chunk_id,
                        "match_type": e.match_type,
                        "score": e.score,
                        "document_id": e.document_id,
                    }
                    for e in res.evidence
                ]

                claim_record = PipelineClaimResult(
                    claim_id=res.claim_id,
                    document_id=res.document_id,
                    domain=res.domain,
                    claim_text=res.claim_text,
                    reference_text=res.reference_text,
                    normalized_reference=res.normalized_reference,
                    reference_type=res.reference_type,
                    page=claim.page,
                    section=claim.section,
                    reference_status=res.reference_status.value,
                    reference_resolution=bib_res,
                    retrieval_candidates=retrieval_summary,
                    source_attribution={
                        "source_availability": source_avail,
                        "eligible_count": len(prov.get("eligible_chunks", [])),
                    },
                    selected_evidence=res.evidence,
                    verdict=res.verdict,
                    confidence=res.confidence,
                    is_conclusive=res.is_conclusive,
                    reasoning=res.reasoning,
                    verifier_version=res.verifier_version,
                    provenance=prov,
                )
                claim_results.append(claim_record)

            except Exception as err:
                msg = f"Verification failed for claim '{claim.claim_id}': {err}"
                logger.error(msg)
                warnings.append(msg)
                has_claim_failures = True

        timings[PipelineStage.VERIFICATION.value] = time.perf_counter() - t_verif_start

        # Record stage completion
        stage_status[PipelineStage.REFERENCE_RESOLUTION.value] = StageStatus.COMPLETED.value
        stage_status[PipelineStage.RETRIEVAL.value] = StageStatus.COMPLETED.value
        stage_status[PipelineStage.SOURCE_ATTRIBUTION.value] = StageStatus.COMPLETED.value
        stage_status[PipelineStage.EVIDENCE_SELECTION.value] = StageStatus.COMPLETED.value
        stage_status[PipelineStage.VERIFICATION.value] = StageStatus.COMPLETED.value

        # =====================================================================
        # Step 9: Summary & Top-Level Result
        # =====================================================================
        verdict_counts: Dict[str, int] = {}
        ref_status_counts: Dict[str, int] = {}
        source_avail_counts: Dict[str, int] = {}
        conclusive = 0
        inconclusive = 0
        eligible_chunks_count = 0
        ineligible_chunks_count = 0

        for cr in claim_results:
            v_val = cr.verdict.value
            verdict_counts[v_val] = verdict_counts.get(v_val, 0) + 1

            r_val = cr.reference_status
            ref_status_counts[r_val] = ref_status_counts.get(r_val, 0) + 1

            s_val = cr.source_attribution.get("source_availability", "unknown")
            source_avail_counts[s_val] = source_avail_counts.get(s_val, 0) + 1

            if cr.is_conclusive:
                conclusive += 1
            else:
                inconclusive += 1

            for ev in cr.selected_evidence:
                if getattr(ev, "is_eligible", True):
                    eligible_chunks_count += 1
                else:
                    ineligible_chunks_count += 1

        summary = PipelineSummary(
            total_claims=len(claim_results),
            verdicts=verdict_counts,
            conclusive_count=conclusive,
            inconclusive_count=inconclusive,
            reference_status_counts=ref_status_counts,
            source_availability_counts=source_avail_counts,
            eligible_evidence_total=eligible_chunks_count,
            ineligible_evidence_total=ineligible_chunks_count,
        )

        overall_time = time.perf_counter() - overall_start
        timings["total"] = overall_time

        overall_status = ProcessingStatus.SUCCESS
        if has_claim_failures and claim_results:
            overall_status = ProcessingStatus.PARTIAL_FAILURE
        elif has_claim_failures and not claim_results:
            overall_status = ProcessingStatus.FAILED

        return PipelineResult(
            document_id=doc_id,
            domain=domain_enum,
            pipeline_version=self.VERSION,
            claims=claim_results,
            summary=summary,
            processing_status=overall_status,
            stage_status=stage_status,
            timings=timings,
            provenance={
                "input_type": type(input_document).__name__,
                "chunks_count": len(chunks),
                "claims_extracted": len(claims),
                "timestamp": time.time(),
            },
            errors=errors,
            warnings=warnings,
        )
