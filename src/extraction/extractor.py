"""High-level facade orchestrating citation detection, reference normalization, and claim extraction."""

from collections import defaultdict
import logging
from typing import Dict, List, Optional

from src.models.schemas import Domain, EvidenceChunk
from src.extraction.citation_detector import CitationDetector
from src.extraction.claim_extractor import ClaimExtractor
from src.extraction.models import CitationClaim, ReferenceType
from src.extraction.reference_normalizer import ReferenceNormalizer

logger = logging.getLogger(__name__)


class ClaimCitationExtractor:
    """High-level unified API for extracting verifiable claims with citations and complete provenance."""

    def __init__(
        self,
        normalizer: Optional[ReferenceNormalizer] = None,
        detector: Optional[CitationDetector] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
    ):
        self.normalizer = normalizer or ReferenceNormalizer()
        self.detector = detector or CitationDetector(normalizer=self.normalizer)
        self.claim_extractor = claim_extractor or ClaimExtractor(detector=self.detector)

    def extract_from_chunk(
        self,
        chunk: EvidenceChunk,
        start_claim_index: int = 1,
    ) -> List[CitationClaim]:
        # Skip heading-only chunks: headings are structural titles, not substantive claims
        content_type_val = (
            chunk.content_type.value
            if hasattr(chunk.content_type, "value")
            else str(chunk.content_type)
        ).lower()
        if content_type_val in {"heading", "title"}:
            logger.debug("Skipping heading chunk %s from claim extraction", chunk.chunk_id)
            return []

        return self.claim_extractor.extract_claims_from_text(
            text=chunk.text,
            domain=chunk.domain,
            document_id=chunk.document_id,
            source_chunk_id=chunk.chunk_id,
            page=chunk.page,
            section=chunk.section,
            section_id=chunk.section_id,
            start_claim_index=start_claim_index,
        )

    def extract_from_chunks(
        self,
        chunks: List[EvidenceChunk],
    ) -> List[CitationClaim]:
        """Extract all CitationClaim objects across a list of EvidenceChunks with sequential claim IDs per document.

        Args:
            chunks: Ordered list of EvidenceChunks.

        Returns:
            Aggregated list of CitationClaim instances.
        """
        doc_claim_counters: Dict[str, int] = defaultdict(lambda: 1)
        all_claims: List[CitationClaim] = []

        for chunk in chunks:
            start_idx = doc_claim_counters[chunk.document_id]
            extracted = self.extract_from_chunk(chunk, start_claim_index=start_idx)
            doc_claim_counters[chunk.document_id] += len(extracted)
            all_claims.extend(extracted)

        logger.info(
            "Extracted %d total citation claims across %d chunks",
            len(all_claims),
            len(chunks),
        )
        return all_claims

    def extract_from_text(
        self,
        text: str,
        domain: Domain,
        document_id: str,
        source_chunk_id: Optional[str] = None,
        page: Optional[int] = None,
        section: Optional[str] = None,
        section_id: Optional[str] = None,
        start_claim_index: int = 1,
    ) -> List[CitationClaim]:
        """Extract claims directly from raw text with specified provenance metadata.

        Args:
            text: Raw input text.
            domain: Domain enum.
            document_id: Document identifier.
            source_chunk_id: Optional source chunk identifier.
            page: Optional page number.
            section: Optional section name.
            section_id: Optional section identifier.
            start_claim_index: Starting sequential sequence number.

        Returns:
            List of extracted CitationClaim instances.
        """
        return self.claim_extractor.extract_claims_from_text(
            text=text,
            domain=domain,
            document_id=document_id,
            source_chunk_id=source_chunk_id,
            page=page,
            section=section,
            section_id=section_id,
            start_claim_index=start_claim_index,
        )
