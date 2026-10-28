"""Source identity verifier and evidence eligibility filter for Stage 10B.1."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.extraction.models import CitationClaim
from src.models.schemas import Domain
from src.references.models import BibliographicEntry
from src.verification.models import EvidenceItem

logger = logging.getLogger(__name__)


class SourceIdentityVerifier:
    """Verifies that substantive evidence belongs to the cited publication.

    Distinguishes three separate concepts:
        1. Bibliographic identity: The cited publication has been identified via Stage 10A.
        2. Source availability: The cited publication's actual content exists locally in the corpus.
        3. Claim support: Substantive evidence from the verified source supports the proposition.
    """

    def verify_source_eligibility(
        self,
        claim: CitationClaim,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
        """Filter and tag evidence items by verified source identity.

        Args:
            claim: CitationClaim under verification.
            evidence: List of candidate EvidenceItem objects retrieved from the corpus.
            resolved_references: List of resolved BibliographicEntry objects from Stage 10A.

        Returns:
            Tuple of:
                - Processed List[EvidenceItem] with is_eligible flag set.
                - Metadata dictionary detailing source availability and attribution status.
        """
        # 1. Legal domain: references route through statutory provisions in the legal corpus
        if claim.domain == Domain.LEGAL:
            for ev in evidence:
                # Structural headings are kept for provenance only, not factual support
                ev.is_eligible = not ev.is_structural
            return evidence, {
                "source_availability": "available",
                "attribution_domain": "legal_statute",
                "eligible_substantive_count": sum(1 for e in evidence if e.is_eligible and not e.is_structural),
            }

        # 2. Scientific domain: check bibliographic identity and source availability
        if not resolved_references:
            # Reference was not resolved to a bibliographic publication
            for ev in evidence:
                ev.is_eligible = False
            return evidence, {
                "source_availability": "unresolved_reference",
                "attribution_domain": "scientific",
                "eligible_substantive_count": 0,
            }

        entry = resolved_references[0]
        verified_source_doc_id = entry.provenance.get("source_document_id")
        entry_doi = (entry.doi or "").strip().lower()
        entry_norm = entry.normalized_label.strip().lower()
        entry_title = (entry.title or "").strip().lower()

        eligible_count = 0
        for ev in evidence:
            # Rule 1: Structural headings are kept for provenance only, never for factual support
            if ev.is_structural or ev.evidence_type == "structural":
                ev.is_eligible = False
                continue

            # Rule 2: The document containing the citation is NOT the cited publication
            if ev.document_id == claim.document_id:
                logger.debug(
                    "Excluding chunk '%s' because its document '%s' is the citing document, not the cited source",
                    ev.chunk_id,
                    ev.document_id,
                )
                ev.is_eligible = False
                continue

            # Rule 3: Check if chunk originates from the verified cited publication
            is_verified_source = False

            # Check explicit source document mapping
            if verified_source_doc_id and ev.document_id == verified_source_doc_id:
                is_verified_source = True
            # Check explicit naming convention e.g. sci_doc_12 or sci_doc_smith_2024
            elif ev.document_id.lower() in (
                f"sci_doc_{entry_norm}",
                f"sci_source_{entry_norm}",
                f"doc_{entry_norm}",
            ):
                is_verified_source = True
            # Check verified DOI match in provenance if present
            elif entry_doi and ev.provenance.get("doi", "").strip().lower() == entry_doi and ev.document_id != claim.document_id:
                is_verified_source = True

            ev.is_eligible = is_verified_source
            if is_verified_source:
                eligible_count += 1

        is_available = eligible_count > 0
        status_str = "available" if is_available else "unavailable"

        metadata = {
            "bibliographic_identity": entry.entry_id,
            "source_availability": status_str,
            "resolved_publication_title": entry.title,
            "resolved_publication_label": entry.reference_label,
            "eligible_substantive_count": eligible_count,
            "total_candidates": len(evidence),
        }

        logger.info(
            "Source attribution for claim '%s': source_availability=%s (eligible_substantive=%d/%d)",
            claim.claim_id,
            status_str,
            eligible_count,
            len(evidence),
        )

        return evidence, metadata
