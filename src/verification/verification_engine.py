"""Verification Engine orchestrating retrieval, evidence selection, and claim evaluation."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, List, Optional
from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain
from src.retrieval.models import MatchType, RetrievalRequest, RetrievalResult
from src.retrieval.retriever import Retriever
from src.verification.evidence_selector import EvidenceSelector
from src.verification.models import (
    EvidenceItem,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
    VerificationResult,
)
from src.verification.verifier import (
    BaseClaimVerifier,
    DeterministicClaimVerifier,
)

logger = logging.getLogger(__name__)

# Reference types that expect exact structural lookup in corpus metadata
EXACT_LOOKUP_TYPES = {
    ReferenceType.LEGAL_SECTION.value,
    ReferenceType.LEGAL_ARTICLE.value,
    ReferenceType.LEGAL_CLAUSE.value,
    ReferenceType.LEGAL_RULE.value,
    ReferenceType.NUMERIC.value,
    "legal_section",
    "legal_article",
    "legal_clause",
    "legal_rule",
    "numeric_citation",
}


def _generate_verification_id(claim_id: str, document_id: str, normalized_reference: str) -> str:
    """Generate a deterministic verification identifier."""
    raw = f"{claim_id}_{document_id}_{normalized_reference}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"verif_{digest}"


from src.verification.source_verifier import SourceIdentityVerifier


class VerificationEngine:
    """High-level engine executing the end-to-end verification pipeline:

    CitationClaim
          ↓
    Stage 10A Reference Resolution (if bibliography available)
          ↓
    Reference correctness check & guard
          ↓
    Stage 7 Retrieval API
          ↓
    Stage 9 Evidence selection
          ↓
    Stage 10B.1 Source identity & eligibility verification
          ↓
    Stage 10B Claim vs evidence semantic evaluation
          ↓
    Structured VerificationResult
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        evidence_selector: Optional[EvidenceSelector] = None,
        verifier: Optional[BaseClaimVerifier] = None,
        reference_resolver: Optional[Any] = None,
        source_verifier: Optional[SourceIdentityVerifier] = None,
        top_k: int = 5,
        max_evidence: int = 5,
    ):
        """Initialize the verification engine.

        Args:
            retriever: High-level Stage 7 Retriever instance.
            evidence_selector: EvidenceSelector instance.
            verifier: BaseClaimVerifier implementation (defaults to DeterministicClaimVerifier).
            reference_resolver: Optional Stage 10A ReferenceResolutionEngine instance.
            source_verifier: Optional SourceIdentityVerifier instance.
            top_k: Top candidates to fetch via Stage 7 semantic retrieval.
            max_evidence: Maximum evidence chunks to supply to the verifier.
        """
        self.retriever = retriever or Retriever()
        self.evidence_selector = evidence_selector or EvidenceSelector(max_evidence=max_evidence)
        self.verifier = verifier or DeterministicClaimVerifier()
        self.reference_resolver = reference_resolver
        self.source_verifier = source_verifier or SourceIdentityVerifier()
        self.top_k = top_k
        self.max_evidence = max_evidence

    def _load_source_publication_chunks(
        self,
        entry: Any,
        domain: Domain,
    ) -> List[RetrievalResult]:
        """Search local corpus files (data/chunks/<domain>/) for chunks belonging to the cited publication.

        Strictly adheres to:
        - Source document identity must be verified by explicit document ID or DOI mapping.
        - Unrelated local files or citing documents are NOT loaded.
        """
        results: List[RetrievalResult] = []
        chunks_dir = Path("data/chunks") / domain.value
        if not chunks_dir.exists():
            return results

        entry_norm = (getattr(entry, "normalized_label", "") or "").strip().lower()
        verified_source_doc_id = getattr(entry, "provenance", {}).get("source_document_id")
        entry_doi = (getattr(entry, "doi", "") or "").strip().lower()

        target_doc_ids = set()
        if verified_source_doc_id:
            target_doc_ids.add(str(verified_source_doc_id).lower())
        if entry_norm:
            target_doc_ids.add(f"sci_doc_{entry_norm}")
            target_doc_ids.add(f"sci_source_{entry_norm}")
            target_doc_ids.add(f"doc_{entry_norm}")

        for jsonl_file in chunks_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        doc_id = data.get("document_id", "").lower()
                        chunk_doi = (data.get("doi") or "").strip().lower()

                        is_match = False
                        if doc_id in target_doc_ids:
                            is_match = True
                        elif entry_doi and chunk_doi and chunk_doi == entry_doi:
                            is_match = True

                        if is_match:
                            candidate = RetrievalResult(
                                chunk_id=data["chunk_id"],
                                document_id=data["document_id"],
                                domain=domain,
                                text=data["text"],
                                score=1.0,
                                match_type=MatchType.EXACT.value,
                                title=data.get("title"),
                                source=data.get("source"),
                                page=data.get("page"),
                                section=data.get("section"),
                                section_id=data.get("section_id"),
                                content_type=data.get("content_type", "paragraph"),
                                rank=1,
                            )
                            results.append(candidate)
            except Exception as e:
                logger.warning("Error reading chunks from %s: %s", jsonl_file, e)

        return results

    def verify(self, claim: CitationClaim) -> VerificationResult:
        """Verify an extracted CitationClaim against the evidence corpus.

        Args:
            claim: Extracted CitationClaim with substantive statement and citation metadata.

        Returns:
            Structured, auditable VerificationResult.
        """
        ref_type = (
            claim.reference_type.value
            if hasattr(claim.reference_type, "value")
            else str(claim.reference_type)
        )
        supports_exact = ref_type in EXACT_LOOKUP_TYPES

        # 1. Stage 10A Reference Resolution (if reference resolver available)
        resolved_entries = None
        resolution_result = None
        if self.reference_resolver is not None:
            try:
                resolution_result = self.reference_resolver.resolve(claim)
                if hasattr(resolution_result, "status") and resolution_result.status.value == "resolved":
                    resolved_entries = resolution_result.matched_entries
            except Exception as e:
                logger.warning("Stage 10A reference resolution exception: %s", e)

        # 2. Prepare exact identifier if reference type supports structured lookup
        exact_id = claim.normalized_reference if supports_exact else None

        # 3. Query Stage 7 Retrieval API
        request = RetrievalRequest(
            query=claim.claim_text,
            domain=claim.domain,
            exact_identifier=exact_id,
            expand_parent_context=True,
            semantic_search_enabled=True,
            top_k=self.top_k,
        )
        retrieval_candidates = self.retriever.retrieve(request)

        # If reference resolution succeeded for a scientific citation, check if local source publication chunks exist
        if claim.domain == Domain.SCIENTIFIC and resolved_entries:
            for entry in resolved_entries:
                source_chunks = self._load_source_publication_chunks(entry, claim.domain)
                if source_chunks:
                    retrieval_candidates = source_chunks + retrieval_candidates
                    break

        # 4. Determine ReferenceStatus
        # CRITICAL RULE: Semantic similarity alone must NEVER be treated as proof that a cited identifier exists.
        has_exact_match = any(
            r.match_type == MatchType.EXACT.value for r in retrieval_candidates
        )

        if supports_exact:
            if has_exact_match:
                ref_status = ReferenceStatus.EXACT_MATCH
            elif resolved_entries:
                # Stage 10A uniquely resolved the bibliography entry for this document
                ref_status = ReferenceStatus.EXACT_MATCH
            else:
                ref_status = ReferenceStatus.NOT_FOUND
        else:
            if retrieval_candidates:
                ref_status = ReferenceStatus.SEMANTIC_CANDIDATE
            else:
                ref_status = ReferenceStatus.NOT_FOUND

        logger.info(
            "Claim '%s' reference '%s' resolved to reference_status=%s (exact_matched=%s, bib_resolved=%s)",
            claim.claim_id,
            claim.normalized_reference,
            ref_status.value,
            has_exact_match,
            bool(resolved_entries),
        )

        # 5. Select and prioritize evidence
        selected_evidence = self.evidence_selector.select_evidence(
            retrieval_candidates,
            max_evidence=self.max_evidence,
        )

        # 6. Stage 10B.1 Source Identity & Eligibility Verification
        selected_evidence, source_meta = self.source_verifier.verify_source_eligibility(
            claim=claim,
            evidence=selected_evidence,
            resolved_references=resolved_entries,
        )

        # 7. REFERENCE STATUS & SOURCE AVAILABILITY GUARD:
        is_llm_or_mock = hasattr(self.verifier, "model_name") or "llm" in getattr(self.verifier, "VERSION", "").lower()

        if ref_status == ReferenceStatus.NOT_FOUND and is_llm_or_mock:
            reasoning = (
                f"The cited reference '{claim.reference_text}' ({claim.normalized_reference}) "
                f"was not found in the evidence corpus. Semantic similarity candidates cannot "
                f"substitute for the missing cited reference. (LLM semantic verification skipped by reference guard)"
            )
            decision = VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=reasoning,
                verifier_version="1.0.0-reference-guard",
                metadata={"guard_applied": "not_found", "llm_bypassed": True},
            )
        elif (
            claim.domain == Domain.SCIENTIFIC
            and resolved_entries
            and source_meta.get("source_availability") == "unavailable"
        ):
            # Bibliographic identity confirmed, but cited publication's actual content is unavailable locally
            entry = resolved_entries[0]
            reasoning = (
                f"Bibliographic reference '{claim.reference_text}' was resolved to "
                f"'{entry.title or entry.reference_label}', but the cited publication's "
                f"substantive content is not available in the local corpus. Retrieved chunks "
                f"from the citing document ('{claim.document_id}') or unrelated documents "
                f"cannot establish claim support."
            )
            decision = VerificationDecision(
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.85,
                reasoning=reasoning,
                verifier_version="1.0.0-source-verifier",
                metadata={
                    "guard_applied": "source_unavailable",
                    "source_availability": "unavailable",
                    "resolved_entry_id": entry.entry_id,
                    **source_meta,
                },
            )
        else:
            # Evaluate claim against selected evidence
            decision = self.verifier.verify(
                claim=claim,
                reference_status=ref_status,
                evidence=selected_evidence,
                resolved_references=resolved_entries,
            )

        # 8. Build auditable provenance
        provenance = {
            "claim_id": claim.claim_id,
            "source_chunk_id": claim.source_chunk_id,
            "document_id": claim.document_id,
            "page": claim.page,
            "section": claim.section,
            "section_id": claim.section_id,
            "selected_chunks": [e.chunk_id for e in selected_evidence],
            "eligible_chunks": [e.chunk_id for e in selected_evidence if getattr(e, "is_eligible", True)],
            "source_availability": source_meta.get("source_availability"),
            "match_types": {e.chunk_id: e.match_type for e in selected_evidence},
            "scores": {e.chunk_id: e.score for e in selected_evidence},
            "total_candidates_retrieved": len(retrieval_candidates),
        }
        if resolution_result is not None:
            provenance["bibliographic_resolution"] = {
                "status": resolution_result.status.value,
                "matched_entry_ids": resolution_result.matched_entry_ids,
            }

        # 7. Generate deterministic VerificationResult
        verification_id = _generate_verification_id(
            claim.claim_id,
            claim.document_id,
            claim.normalized_reference,
        )

        return VerificationResult(
            verification_id=verification_id,
            claim_id=claim.claim_id,
            document_id=claim.document_id,
            domain=claim.domain,
            claim_text=claim.claim_text,
            reference_text=claim.reference_text,
            normalized_reference=claim.normalized_reference,
            reference_type=ref_type,
            verdict=decision.verdict,
            reference_status=ref_status,
            confidence=decision.confidence,
            evidence=selected_evidence,
            reasoning=decision.reasoning,
            provenance=provenance,
            verifier_version=decision.verifier_version,
        )
