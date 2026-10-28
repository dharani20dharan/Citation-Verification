"""Verifiers evaluating citation claims against retrieved corpus evidence."""

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator

from src.extraction.models import CitationClaim
from src.verification.models import (
    EvidenceItem,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
)

logger = logging.getLogger(__name__)

# Polarity contradiction pairs: (affirmative pattern, contradictory pattern)
CONTRADICTION_PATTERNS = [
    (r"\b(?:is|are|shall be)\s+patentable\b", r"\b(?:is|are|shall be)?\s*(?:not|never)\s+patentable\b|\bwhat are not inventions\b"),
    (r"\bpatentable\b", r"\bnot\s+patentable\b|\bunpatentable\b"),
    (r"\b(?:is|are)\s+permitted\b", r"\b(?:is|are)?\s*(?:not\s+permitted|prohibited|forbidden)\b"),
    (r"\b(?:is|are)\s+allowed\b", r"\b(?:is|are)?\s*(?:not\s+allowed|disallowed)\b"),
    (r"\bupheld\b", r"\b(?:struck down|overruled|quashed|dismissed|reversed)\b"),
    (r"\bvalid\b", r"\b(?:invalid|null and void|void)\b"),
    (r"\bconstitutional\b", r"\bunconstitutional\b"),
    (r"\bmandatory\b", r"\b(?:optional|discretionary|not mandatory)\b"),
    (r"\bviolates\b", r"\b(?:complies with|does not violate)\b"),
    (r"\binfringes\b", r"\b(?:does not infringe|non-infringing)\b"),
]

# Stop words ignored during substantive overlap evaluation
COMMON_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by",
    "of", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "this", "that", "these", "those", "it", "its", "their",
    "such", "which", "who", "whom", "will", "would", "shall", "should", "can", "could",
    "under", "into", "through", "during", "before", "after", "above", "below",
}


def _normalize_tokens(text: str) -> List[str]:
    """Tokenize and filter text into normalized substantive tokens."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t.strip() for t in clean.split() if t.strip()]
    return [t for t in tokens if t not in COMMON_STOP_WORDS and len(t) > 1]


def _clean_str(text: str) -> str:
    """Normalize whitespace and punctuation for substring checks."""
    s = re.sub(r"\s+", " ", text.strip().lower())
    s = re.sub(r"[^\w\s]", "", s)
    return s


class BaseClaimVerifier(ABC):
    """Abstract base verifier interface.

    Allows replacing or plugging in different verification strategies (e.g. deterministic,
    specialized, or future LLM-based) without altering the verification pipeline.
    """

    @abstractmethod
    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[Any]] = None,
    ) -> VerificationDecision:
        """Verify the claim against retrieved evidence and reference status.

        Args:
            claim: Extracted CitationClaim under review.
            reference_status: Status of the cited reference in the corpus.
            evidence: Selected and ranked EvidenceItem list.

        Returns:
            VerificationDecision with verdict, confidence, and reasoning.
        """
        pass


class DeterministicClaimVerifier(BaseClaimVerifier):
    """Conservative, deterministic baseline verifier.

    Confidence Semantics:
        Confidence values (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50) represent
        deterministic confidence tiers reflecting rule-based evidence sufficiency and match
        completeness. They are NOT calibrated statistical or Bayesian probabilities.

    Evaluates:
        - Case A: Exact reference exists + substantive evidence contains the claimed statement -> supported
        - Case B: Exact reference exists + substantive evidence contradicts the claim -> unsupported
        - Case C: Exact reference exists + substantive evidence contains only part of the claim -> partially_correct
        - Case D: Exact reference does not exist -> not_found
        - Case E: Retrieved evidence points to a different identifiable reference -> wrong_reference
        - Case F: Only structural heading evidence exists -> unsupported (structural headings establish
                  existence but cannot establish factual support)

    Maintains conservative confidence tiers and generates transparent, auditable reasoning.
    """

    VERSION = "0.9.1-deterministic"

    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[Any]] = None,
    ) -> VerificationDecision:
        """Execute deterministic verification."""
        # Case D: Reference does not exist in corpus
        if reference_status == ReferenceStatus.NOT_FOUND:
            reasoning = (
                f"The cited reference '{claim.reference_text}' ({claim.normalized_reference}) "
                f"was not found in the evidence corpus. Semantic similarity candidates cannot "
                f"substitute for the missing cited reference."
            )
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "D", "reference_status": reference_status.value},
            )

        # Empty evidence check
        if not evidence:
            reasoning = (
                f"No evidence could be retrieved for cited reference '{claim.reference_text}'."
            )
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "empty_evidence"},
            )

        # Case E: Check for wrong reference indicators
        wrong_ref_decision = self._check_wrong_reference(claim, evidence)
        if wrong_ref_decision:
            return wrong_ref_decision

        # Partition evidence into substantive factual evidence and structural reference evidence
        substantive_evidence = [e for e in evidence if not e.is_structural]
        structural_evidence = [e for e in evidence if e.is_structural]

        # Case F: Only structural heading evidence exists
        # Structural headings establish reference existence but cannot establish factual support
        if not substantive_evidence:
            reasoning = (
                f"The cited reference '{claim.reference_text}' was located structurally in the corpus "
                f"({len(structural_evidence)} structural heading chunk(s)), but no substantive factual "
                f"evidence was retrieved to verify the proposition. Structural headings confirm reference "
                f"existence but cannot establish that the claim is factually supported."
            )
            return VerificationDecision(
                verdict=Verdict.UNSUPPORTED,
                confidence=0.80,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={
                    "case": "F_structural_only",
                    "structural_chunks": [e.chunk_id for e in structural_evidence],
                },
            )

        # Aggregate substantive evidence texts for factual verification
        substantive_text = " ".join(e.text for e in substantive_evidence)

        # Case B: Contradiction check against substantive evidence
        contradiction_reasoning = self._check_contradiction(claim.claim_text, substantive_text)
        if contradiction_reasoning:
            return VerificationDecision(
                verdict=Verdict.UNSUPPORTED,
                confidence=0.90,
                reasoning=(
                    f"The cited reference '{claim.reference_text}' exists, but the retrieved "
                    f"substantive evidence contradicts the claim: {contradiction_reasoning}"
                ),
                verifier_version=self.VERSION,
                metadata={"case": "B", "contradiction": True},
            )

        # Case A / C: Substantive support evaluation against substantive evidence
        return self._evaluate_support(claim, reference_status, substantive_evidence, substantive_text)

    def _check_wrong_reference(
        self,
        claim: CitationClaim,
        evidence: List[EvidenceItem],
    ) -> Optional[VerificationDecision]:
        """Detect if evidence clearly attributes the asserted proposition to another identifiable reference."""
        # 1. Check if claim text explicitly mentions a section different from its cited reference
        norm_ref = claim.normalized_reference.strip().lower()

        # Find any explicitly cited section numbers in the claim text (e.g. "Section 3(e)")
        mentioned_sections = re.findall(r"\bsection\s+([0-9]+(?:\([a-z0-9]+\))*)", claim.claim_text, re.IGNORECASE)
        for m_sec in mentioned_sections:
            clean_m = m_sec.strip().lower()
            if clean_m and clean_m != norm_ref and f"section {clean_m}" != norm_ref:
                return VerificationDecision(
                    verdict=Verdict.WRONG_REFERENCE,
                    confidence=0.85,
                    reasoning=(
                        f"The claim text explicitly describes Section {m_sec}, but cites "
                        f"'{claim.reference_text}' ({claim.normalized_reference}). "
                        f"Retrieved evidence indicates a mismatch between the asserted proposition and cited identifier."
                    ),
                    verifier_version=self.VERSION,
                    metadata={"case": "E", "detected_reference": m_sec},
                )

        # 2. Check if all top matching evidence items explicitly point to a different clause_id
        exact_evidence = [e for e in evidence if e.match_type == "exact"]
        if not exact_evidence and evidence:
            top_clause_ids = {
                e.clause_id.strip().lower()
                for e in evidence
                if e.clause_id and e.clause_id.strip()
            }
            if top_clause_ids and norm_ref not in top_clause_ids:
                distinct_id = next(iter(top_clause_ids))
                return VerificationDecision(
                    verdict=Verdict.WRONG_REFERENCE,
                    confidence=0.85,
                    reasoning=(
                        f"The cited reference '{claim.reference_text}' was not matched. Retrieved evidence "
                        f"substantiating the subject matter belongs to a different reference '{distinct_id}'."
                    ),
                    verifier_version=self.VERSION,
                    metadata={"case": "E", "retrieved_different_clause": distinct_id},
                )

        return None

    def _check_contradiction(self, claim_text: str, evidence_text: str) -> Optional[str]:
        """Check if claim statement directly contradicts retrieved evidence text."""
        claim_lower = claim_text.lower()
        ev_lower = evidence_text.lower()

        for aff_pat, cont_pat in CONTRADICTION_PATTERNS:
            # Check if claim asserts affirmative and evidence asserts contradictory
            if re.search(aff_pat, claim_lower) and re.search(cont_pat, ev_lower):
                # Ensure the claim itself didn't already have the contradictory pattern
                if not re.search(cont_pat, claim_lower):
                    return (
                        f"The claim asserts affirmative status '{re.search(aff_pat, claim_lower).group(0)}', "
                        f"whereas the evidence provision specifies '{re.search(cont_pat, ev_lower).group(0)}'."
                    )

            # Check reverse: claim asserts negative and evidence affirms
            if re.search(cont_pat, claim_lower) and re.search(aff_pat, ev_lower):
                if not re.search(cont_pat, ev_lower):
                    return (
                        f"The claim asserts negative/restrictive status, whereas the evidence specifies affirmative."
                    )

        return None

    def _evaluate_support(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        combined_text: str,
    ) -> VerificationDecision:
        """Evaluate whether retrieved evidence supports, partially supports, or fails to support the claim."""
        claim_clean = _clean_str(claim.claim_text)
        evidence_clean = _clean_str(combined_text)

        # 1. Exact substring check (high confidence corroboration)
        if claim_clean in evidence_clean:
            reasoning = (
                f"The cited reference '{claim.reference_text}' was found exactly. "
                f"The retrieved provision verbatim corroborates the substantive statement made in the claim."
            )
            return VerificationDecision(
                verdict=Verdict.SUPPORTED,
                confidence=0.90 if reference_status == ReferenceStatus.EXACT_MATCH else 0.80,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "A", "match_type": "exact_verbatim"},
            )

        # 2. Token overlap analysis
        claim_tokens = _normalize_tokens(claim.claim_text)
        evidence_tokens = set(_normalize_tokens(combined_text))

        if not claim_tokens:
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning="Claim text contains insufficient substantive tokens for verification.",
                verifier_version=self.VERSION,
                metadata={"case": "C"},
            )

        matched_tokens = [t for t in claim_tokens if t in evidence_tokens]
        overlap_ratio = len(matched_tokens) / len(claim_tokens)

        logger.debug(
            "Claim '%s' token overlap: %d/%d (%.2f)",
            claim.claim_id,
            len(matched_tokens),
            len(claim_tokens),
            overlap_ratio,
        )

        # 3. Decision boundaries
        if overlap_ratio >= 0.80:
            confidence = 0.85 if reference_status == ReferenceStatus.EXACT_MATCH else 0.75
            reasoning = (
                f"The cited reference '{claim.reference_text}' was found. "
                f"The retrieved provision corroborates the substantive requirement described in the claim "
                f"({int(overlap_ratio * 100)}% substantive term match)."
            )
            return VerificationDecision(
                verdict=Verdict.SUPPORTED,
                confidence=confidence,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "A", "overlap_ratio": round(overlap_ratio, 3)},
            )
        elif overlap_ratio >= 0.40:
            confidence = 0.70
            missing_tokens = [t for t in claim_tokens if t not in evidence_tokens]
            missing_preview = ", ".join(missing_tokens[:3])
            reasoning = (
                f"The cited reference '{claim.reference_text}' was found, but the retrieved evidence "
                f"contains only part of the claim ({int(overlap_ratio * 100)}% term match). "
                f"Missing terms or conditions: {missing_preview}."
            )
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=confidence,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "C", "overlap_ratio": round(overlap_ratio, 3)},
            )
        else:
            confidence = 0.75
            reasoning = (
                f"The cited reference '{claim.reference_text}' was found, but the retrieved provision "
                f"does not contain the claimed substantive outcome (only {int(overlap_ratio * 100)}% match)."
            )
            return VerificationDecision(
                verdict=Verdict.UNSUPPORTED,
                confidence=confidence,
                reasoning=reasoning,
                verifier_version=self.VERSION,
                metadata={"case": "B_weak_overlap", "overlap_ratio": round(overlap_ratio, 3)},
            )


# ==============================================================================
# 8. FUTURE LLM VERIFIER ABSTRACTION & SCHEMAS
# ==============================================================================

class LLMVerificationPayload(BaseModel):
    """Structured input payload supplied to an LLM verifier."""

    claim_text: str = Field(..., description="Substantive claim statement to be verified")
    cited_reference: str = Field(..., description="Canonical reference identifier")
    reference_status: ReferenceStatus = Field(..., description="Reference presence status in corpus")
    evidence_items: List[Dict[str, Any]] = Field(..., description="List of retrieved evidence chunks with provenance")


class LLMStructuredOutput(BaseModel):
    """Pydantic schema enforcing structured response from LLM."""

    verdict: Verdict = Field(..., description="LLM decided verdict")
    confidence: float = Field(..., description="Bounded confidence score", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation grounded exclusively in claim and evidence", min_length=1)
    evidence_chunk_ids_used: List[str] = Field(
        default_factory=list,
        description="Chunk identifiers explicitly relied upon by LLM",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("LLM confidence must be between 0.0 and 1.0")
        return round(float(v), 4)


class LLMClaimVerifier(BaseClaimVerifier):
    """Abstraction for future LLM-based verification.

    Defines the contract, formats the structured prompt payload, and strictly validates
    the output with Pydantic. Does NOT perform unconfigured external network/API calls.
    """

    VERSION = "0.9.0-llm-interface"

    def __init__(
        self,
        llm_callable: Optional[Callable[[LLMVerificationPayload], LLMStructuredOutput]] = None,
        model_name: str = "mock-llm-verifier",
    ):
        """Initialize LLM verifier abstraction.

        Args:
            llm_callable: Optional callable executing the model call and returning structured output.
            model_name: Identifier of the target LLM.
        """
        self.llm_callable = llm_callable
        self.model_name = model_name

    def format_payload(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
    ) -> LLMVerificationPayload:
        """Construct validated structured verification payload."""
        evidence_dicts = [
            {
                "chunk_id": e.chunk_id,
                "text": e.text,
                "match_type": e.match_type,
                "score": e.score,
                "provenance": e.provenance,
            }
            for e in evidence
        ]
        return LLMVerificationPayload(
            claim_text=claim.claim_text,
            cited_reference=claim.normalized_reference,
            reference_status=reference_status,
            evidence_items=evidence_dicts,
        )

    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[Any]] = None,
    ) -> VerificationDecision:
        """Verify claim using LLM interface."""
        payload = self.format_payload(claim, reference_status, evidence)

        if self.llm_callable is not None:
            # Invoke caller-provided LLM callable
            structured_out = self.llm_callable(payload)
            # Validate through Pydantic
            validated = LLMStructuredOutput.model_validate(structured_out)
            return VerificationDecision(
                verdict=validated.verdict,
                confidence=validated.confidence,
                reasoning=validated.reasoning,
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"evidence_used": validated.evidence_chunk_ids_used},
            )

        # Fallback dry-run when no external API is configured
        logger.info(
            "LLMClaimVerifier called in dry-run mode (no external API configured) for claim '%s'",
            claim.claim_id,
        )
        # In dry-run mode without external API, delegate conservatively to deterministic logic
        fallback_verifier = DeterministicClaimVerifier()
        decision = fallback_verifier.verify(claim, reference_status, evidence)
        return VerificationDecision(
            verdict=decision.verdict,
            confidence=decision.confidence,
            reasoning=f"[LLM Dry-Run / Interface Mode] {decision.reasoning}",
            verifier_version=f"{self.VERSION}:{self.model_name}",
            metadata={"llm_dry_run": True, **decision.metadata},
        )
