"""Prompt builder constructing structured, evidence-grounded verification prompts."""

import json
from typing import Any, Dict, List, Optional

from src.extraction.models import CitationClaim
from src.references.models import BibliographicEntry
from src.verification.models import EvidenceItem, ReferenceStatus


SYSTEM_INSTRUCTION = """You are an evidence-grounded citation verification component.

Your task is to determine whether the supplied evidence supports the supplied claim.

Use ONLY the supplied evidence.
Do not rely on:
- prior knowledge
- external knowledge
- assumptions
- unstated facts
- web information

Do not determine whether the citation exists.
Reference existence has already been determined by the retrieval and reference-resolution pipeline.

Do not invent facts, sources, evidence IDs, page numbers, or bibliographic information.

If the evidence directly supports the complete proposition, return supported.
If the evidence directly contradicts the proposition, return unsupported.
If the evidence supports only part of the proposition or lacks enough information for the complete proposition, return partially_correct.

Return only the required structured output."""


class VerificationPromptBuilder:
    """Builds structured verification prompts adhering strictly to Stage 10B specifications."""

    def build_prompt(
        self,
        claim: CitationClaim,
        evidence: List[EvidenceItem],
        reference_status: ReferenceStatus,
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> str:
        """Construct the complete structured prompt for semantic verification.

        Args:
            claim: CitationClaim containing claim text and citation metadata.
            evidence: List of trusted, selected EvidenceItem objects.
            reference_status: Status of the reference in the corpus (e.g. EXACT_MATCH).
            resolved_references: Optional list of BibliographicEntry objects from Stage 10A.

        Returns:
            Structured prompt text with clearly separated sections.
        """
        sections: List[str] = []

        # 1. ROLE
        sections.append("### ROLE\n" + SYSTEM_INSTRUCTION)

        # 2. TASK
        sections.append(
            "### TASK\n"
            "Evaluate whether the substantive proposition in CLAIM is supported, contradicted, "
            "or partially supported by the provided EVIDENCE items."
        )

        # 3. CLAIM
        sections.append(f"### CLAIM\n\"{claim.claim_text.strip()}\"")

        # 4. CITED REFERENCE
        norm_ref = claim.normalized_reference
        raw_ref = claim.reference_text
        ref_type = (
            claim.reference_type.value
            if hasattr(claim.reference_type, "value")
            else str(claim.reference_type)
        )
        sections.append(
            f"### CITED REFERENCE\n"
            f"Reference Text: {raw_ref}\n"
            f"Normalized Reference: {norm_ref}\n"
            f"Reference Type: {ref_type}"
        )

        # 5. REFERENCE STATUS
        sections.append(
            f"### REFERENCE STATUS\n"
            f"Status: {reference_status.value}\n"
            f"Note: Reference presence and validity have already been verified deterministically. "
            f"Do not attempt to verify reference existence."
        )

        # 6. BIBLIOGRAPHIC CONTEXT
        if resolved_references:
            bib_lines = ["Bibliographic metadata identifies the cited source. It is not itself evidence that the claim is true:"]
            for idx, entry in enumerate(resolved_references, 1):
                bib_lines.append(f"[{idx}] Label: {entry.reference_label}")
                if entry.title:
                    bib_lines.append(f"    Title: {entry.title}")
                if entry.authors:
                    bib_lines.append(f"    Authors: {', '.join(entry.authors)}")
                if entry.year is not None:
                    bib_lines.append(f"    Year: {entry.year}")
                if entry.journal:
                    bib_lines.append(f"    Venue: {entry.journal}")
                if entry.doi:
                    bib_lines.append(f"    DOI: {entry.doi}")
            sections.append("### BIBLIOGRAPHIC CONTEXT\n" + "\n".join(bib_lines))
        else:
            sections.append(
                "### BIBLIOGRAPHIC CONTEXT\n"
                "(No bibliographic entry resolved or reference is a statutory legal section)"
            )

        # 7. EVIDENCE
        ev_lines = [
            "The following evidence items were retrieved from the local corpus. "
            "Structural headings establish document organization/reference identity but are not substantive factual evidence. "
            "Evaluate primarily substantive evidence items:"
        ]
        if not evidence:
            ev_lines.append("(No evidence items provided)")
        else:
            for ev in evidence:
                ev_id = ev.evidence_id or ev.chunk_id
                ev_type = ev.evidence_type or ("structural" if ev.is_structural else "substantive")
                ev_lines.append(
                    f"--- EVIDENCE ITEM [{ev_id}] ---\n"
                    f"Evidence ID  : {ev_id}\n"
                    f"Chunk ID     : {ev.chunk_id}\n"
                    f"Document ID  : {ev.document_id}\n"
                    f"Page         : {ev.page if ev.page is not None else 'N/A'}\n"
                    f"Section      : {ev.section or 'N/A'}\n"
                    f"Match Type   : {ev.match_type}\n"
                    f"Score        : {ev.score:.4f}\n"
                    f"Evidence Type: {ev_type}\n"
                    f"Content:\n{ev.text.strip()}\n"
                )
        sections.append("### EVIDENCE\n" + "\n".join(ev_lines))

        # 8. DECISION RULES
        decision_rules = (
            "### DECISION RULES\n"
            "Apply the following semantic rules strictly:\n"
            "1. supported:\n"
            "   The supplied evidence directly and completely supports the substantive proposition made in the claim.\n"
            "2. unsupported:\n"
            "   The supplied evidence directly contradicts the claim's proposition, or contradicts its polarity "
            "   (e.g., claim asserts 'increases' but evidence states 'decreases'; claim asserts 'improves' but evidence states 'does not improve').\n"
            "3. partially_correct:\n"
            "   - Only part of a compound claim containing multiple propositions is supported.\n"
            "   - The evidence supports the general qualitative proposition but fails to explicitly support a specific quantitative value or percentage (e.g. claim says '40%', evidence only says 'improves').\n"
            "   - The evidence is insufficient to verify the complete proposition.\n"
            "4. insufficient_evidence:\n"
            "   - The available evidence cannot establish support, contradiction, or partial support.\n"
            "   - Substantive evidence from the verified cited publication is missing or unavailable.\n"
            "5. Numerical / Quantitative Claims:\n"
            "   Do not infer numerical values or perform arithmetic. A specific number must be explicitly verified by the evidence text.\n"
            "6. Tables and Figures:\n"
            "   Tabular evidence can support quantitative claims only when the relevant value is explicitly present.\n"
            "7. Allowed Verdicts:\n"
            "   ONLY 'supported', 'unsupported', 'partially_correct', or 'insufficient_evidence'. Do NOT output 'not_found' or 'wrong_reference'."
        )
        sections.append(decision_rules)

        # 9. OUTPUT FORMAT
        output_format = (
            "### OUTPUT FORMAT\n"
            "Respond ONLY with a valid, raw JSON object (no markdown code fences, no extra text) with the following structure:\n"
            "{\n"
            "  \"verdict\": \"supported\" | \"unsupported\" | \"partially_correct\" | \"insufficient_evidence\",\n"
            "  \"confidence\": <float between 0.0 and 1.0 representing confidence in the semantic relationship>,\n"
            "  \"evidence_ids\": [<list of evidence_id strings from the supplied evidence items used>],\n"
            "  \"reasoning\": \"<concise, evidence-grounded explanation of the verdict>\"\n"
            "}\n"
            "CRITICAL: The evidence_ids array must contain ONLY identifiers present in the EVIDENCE section above. "
            "Do not cite chunk IDs or evidence IDs that were not supplied."
        )
        sections.append(output_format)

        return "\n\n".join(sections)

    def build_payload(
        self,
        claim: CitationClaim,
        evidence: List[EvidenceItem],
        reference_status: ReferenceStatus,
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON-serializable dictionary representation of the verification request."""
        bib_context = None
        if resolved_references:
            bib_context = [
                {
                    "label": e.reference_label,
                    "normalized_label": e.normalized_label,
                    "title": e.title,
                    "authors": e.authors,
                    "year": e.year,
                    "venue": e.journal,
                    "doi": e.doi,
                    "url": e.url,
                }
                for e in resolved_references
            ]

        evidence_payload = [
            {
                "evidence_id": e.evidence_id or e.chunk_id,
                "chunk_id": e.chunk_id,
                "document_id": e.document_id,
                "text": e.text,
                "page": e.page,
                "section": e.section,
                "match_type": e.match_type,
                "score": e.score,
                "evidence_type": e.evidence_type or ("structural" if e.is_structural else "substantive"),
            }
            for e in evidence
        ]

        return {
            "claim": claim.claim_text,
            "citation": claim.reference_text,
            "normalized_reference": claim.normalized_reference,
            "reference_type": (
                claim.reference_type.value
                if hasattr(claim.reference_type, "value")
                else str(claim.reference_type)
            ),
            "reference_status": reference_status.value,
            "bibliographic_references": bib_context,
            "evidence": evidence_payload,
        }
