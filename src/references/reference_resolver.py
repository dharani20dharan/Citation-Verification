"""Deterministic citation-to-bibliography resolver for Stage 10A."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.references.models import (
    BibliographicEntry,
    ReferenceResolutionResult,
    ResolutionStatus,
    generate_resolution_id,
)
from src.references.reference_normalizer import NormalizedReference, ReferenceNormalizer

logger = logging.getLogger(__name__)


class ReferenceResolver:
    """Resolves citation references against bibliographic entries deterministically.

    Resolution rules:
        - Numeric single: exact match on normalized_label
        - Numeric multiple: resolves each element; all resolved -> resolved, subset -> partially_resolved
        - Numeric range: expands range [5-7] -> 5, 6, 7; resolves each individually
        - Author-year: deterministic metadata lookup on lead surname and publication year
        - Legal: returns not_applicable, routing to Stage 7 structural legal retrieval
    """

    VERSION = "1.0.0-stage10a"

    def __init__(self, normalizer: Optional[ReferenceNormalizer] = None):
        self.normalizer = normalizer or ReferenceNormalizer()

    def resolve(
        self,
        reference_text: str,
        bibliography: List[BibliographicEntry],
        document_id: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> ReferenceResolutionResult:
        """Resolve a reference string against available bibliographic entries.

        Args:
            reference_text: Raw reference string (e.g. '[12]', '[3, 4]', '[5-7]', '(Smith et al., 2024)', 'Section 3(d)').
            bibliography: List of candidate BibliographicEntry objects.
            document_id: Optional source document ID.
            claim_id: Optional source claim ID.

        Returns:
            Structured ReferenceResolutionResult with status, matched entries, and provenance.
        """
        norm_ref = self.normalizer.normalize(reference_text)

        # 1. Legal references are not bibliographic entries
        if norm_ref.is_legal:
            return self._handle_legal_reference(norm_ref, document_id, claim_id)

        # Index bibliography by document_id if provided
        filtered_entries = (
            [e for e in bibliography if e.document_id == document_id]
            if document_id
            else bibliography
        )

        # 2. Author-year reference resolution
        if norm_ref.reference_type == "author_year":
            return self._resolve_author_year(norm_ref, filtered_entries, document_id, claim_id)

        # 3. Numeric single reference resolution
        if not norm_ref.is_range and len(norm_ref.lookup_targets) == 1:
            return self._resolve_numeric_single(norm_ref, filtered_entries, document_id, claim_id)

        # 4. Numeric multiple and range reference resolution
        return self._resolve_numeric_multiple_or_range(norm_ref, filtered_entries, document_id, claim_id)

    def _handle_legal_reference(
        self,
        norm_ref: NormalizedReference,
        document_id: Optional[str],
        claim_id: Optional[str],
    ) -> ReferenceResolutionResult:
        """Handle legal structured references (not applicable for bibliography resolution)."""
        res_id = generate_resolution_id(norm_ref.raw_reference, document_id, claim_id)
        reasoning = (
            f"The reference '{norm_ref.raw_reference}' is a structured legal reference "
            f"({norm_ref.reference_type}), not a scientific bibliography entry. "
            f"Legal references route through Stage 7 exact structural retrieval."
        )
        return ReferenceResolutionResult(
            resolution_id=res_id,
            claim_id=claim_id,
            document_id=document_id,
            reference_text=norm_ref.raw_reference,
            normalized_reference=norm_ref.normalized_string,
            reference_type=norm_ref.reference_type,
            status=ResolutionStatus.NOT_APPLICABLE,
            confidence=1.0,
            reasoning=reasoning,
            provenance={"legal_routing": "stage_7_structural_retrieval"},
            resolver_version=self.VERSION,
        )

    def _resolve_numeric_single(
        self,
        norm_ref: NormalizedReference,
        entries: List[BibliographicEntry],
        document_id: Optional[str],
        claim_id: Optional[str],
    ) -> ReferenceResolutionResult:
        """Resolve a single numeric reference identifier (e.g. '[12]')."""
        target = norm_ref.lookup_targets[0]
        matches = [e for e in entries if e.normalized_label == target]

        res_id = generate_resolution_id(norm_ref.raw_reference, document_id, claim_id)

        if len(matches) == 1:
            entry = matches[0]
            reasoning = (
                f"Numeric citation '{norm_ref.raw_reference}' uniquely resolved to "
                f"bibliographic entry '{entry.entry_id}' (label: {entry.reference_label})."
            )
            provenance = {
                "matched_entry_id": entry.entry_id,
                "target": target,
                "entry_page": entry.page,
                "entry_section": entry.section,
                "entry_chunk_id": entry.chunk_id,
            }
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.RESOLVED,
                matched_entry_ids=[entry.entry_id],
                matched_entries=[entry],
                confidence=0.95,
                reasoning=reasoning,
                provenance=provenance,
                resolver_version=self.VERSION,
            )
        elif len(matches) > 1:
            entry_ids = [e.entry_id for e in matches]
            reasoning = (
                f"Numeric citation '{norm_ref.raw_reference}' is ambiguous: "
                f"matched {len(matches)} distinct bibliography entries {entry_ids}."
            )
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.AMBIGUOUS,
                matched_entry_ids=entry_ids,
                matched_entries=matches,
                confidence=0.70,
                reasoning=reasoning,
                provenance={"conflicting_entry_ids": entry_ids},
                resolver_version=self.VERSION,
            )
        else:
            reasoning = (
                f"Numeric citation '{norm_ref.raw_reference}' was not found in the "
                f"document's bibliography (searched label: '{target}')."
            )
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.NOT_FOUND,
                confidence=0.95,
                reasoning=reasoning,
                provenance={"searched_target": target},
                resolver_version=self.VERSION,
            )

    def _resolve_numeric_multiple_or_range(
        self,
        norm_ref: NormalizedReference,
        entries: List[BibliographicEntry],
        document_id: Optional[str],
        claim_id: Optional[str],
    ) -> ReferenceResolutionResult:
        """Resolve grouped numeric citations (e.g. '[3, 4]') or ranges (e.g. '[5-7]')."""
        targets = norm_ref.lookup_targets
        individual_results: Dict[str, List[BibliographicEntry]] = {}

        for target in targets:
            matches = [e for e in entries if e.normalized_label == target]
            individual_results[target] = matches

        resolved_targets = [t for t, m in individual_results.items() if len(m) == 1]
        missing_targets = [t for t, m in individual_results.items() if len(m) == 0]
        ambiguous_targets = [t for t, m in individual_results.items() if len(m) > 1]

        all_matched_entries: List[BibliographicEntry] = []
        for t in targets:
            all_matched_entries.extend(individual_results.get(t, []))

        matched_entry_ids = [e.entry_id for e in all_matched_entries]
        res_id = generate_resolution_id(norm_ref.raw_reference, document_id, claim_id)

        target_kind = "Range" if norm_ref.is_range else "Grouped citation"

        # Check resolution outcome
        if len(resolved_targets) == len(targets) and not ambiguous_targets:
            reasoning = (
                f"{target_kind} '{norm_ref.raw_reference}' successfully resolved all "
                f"{len(targets)} targets ({', '.join(targets)}) to distinct entries."
            )
            status = ResolutionStatus.RESOLVED
            confidence = 0.95
        elif resolved_targets and (missing_targets or ambiguous_targets):
            parts = []
            if resolved_targets:
                parts.append(f"resolved: {', '.join(resolved_targets)}")
            if missing_targets:
                parts.append(f"missing: {', '.join(missing_targets)}")
            if ambiguous_targets:
                parts.append(f"ambiguous: {', '.join(ambiguous_targets)}")

            reasoning = (
                f"{target_kind} '{norm_ref.raw_reference}' partially resolved ({'; '.join(parts)})."
            )
            status = ResolutionStatus.PARTIALLY_RESOLVED
            confidence = 0.80
        elif ambiguous_targets and not missing_targets:
            reasoning = (
                f"{target_kind} '{norm_ref.raw_reference}' produced ambiguous matches for: "
                f"{', '.join(ambiguous_targets)}."
            )
            status = ResolutionStatus.AMBIGUOUS
            confidence = 0.70
        else:
            reasoning = (
                f"{target_kind} '{norm_ref.raw_reference}' targets ({', '.join(targets)}) "
                f"were not found in the bibliography."
            )
            status = ResolutionStatus.NOT_FOUND
            confidence = 0.95

        provenance = {
            "is_range": norm_ref.is_range,
            "targets": targets,
            "resolved_targets": resolved_targets,
            "missing_targets": missing_targets,
            "ambiguous_targets": ambiguous_targets,
            "matched_entries_count": len(all_matched_entries),
        }

        return ReferenceResolutionResult(
            resolution_id=res_id,
            claim_id=claim_id,
            document_id=document_id,
            reference_text=norm_ref.raw_reference,
            normalized_reference=norm_ref.normalized_string,
            reference_type=norm_ref.reference_type,
            status=status,
            matched_entry_ids=matched_entry_ids,
            matched_entries=all_matched_entries,
            confidence=confidence,
            reasoning=reasoning,
            provenance=provenance,
            resolver_version=self.VERSION,
        )

    def _resolve_author_year(
        self,
        norm_ref: NormalizedReference,
        entries: List[BibliographicEntry],
        document_id: Optional[str],
        claim_id: Optional[str],
    ) -> ReferenceResolutionResult:
        """Resolve author-year citation (e.g. '(Smith et al., 2024)') using deterministic metadata."""
        target_surname = (norm_ref.author_surname or "").strip().lower()
        target_year = norm_ref.publication_year

        matches: List[BibliographicEntry] = []
        for e in entries:
            # Year check
            year_matches = e.year == target_year if (e.year and target_year) else False
            if not year_matches and str(target_year) in e.raw_text:
                year_matches = True

            if not year_matches:
                continue

            # Author surname check
            surname_matches = False
            if target_surname:
                # Check authors list
                for a in e.authors:
                    if target_surname in a.lower():
                        surname_matches = True
                        break
                # Fallback to normalized_label or raw_text
                if not surname_matches:
                    if target_surname in e.normalized_label.lower() or target_surname in e.raw_text.lower():
                        surname_matches = True

            if surname_matches:
                matches.append(e)

        res_id = generate_resolution_id(norm_ref.raw_reference, document_id, claim_id)

        if len(matches) == 1:
            entry = matches[0]
            reasoning = (
                f"Author-year citation '{norm_ref.raw_reference}' uniquely matched "
                f"bibliographic entry '{entry.entry_id}' (Author: {target_surname.capitalize()}, Year: {target_year})."
            )
            provenance = {
                "matched_entry_id": entry.entry_id,
                "author_surname": target_surname,
                "publication_year": target_year,
                "entry_page": entry.page,
                "entry_section": entry.section,
                "entry_chunk_id": entry.chunk_id,
            }
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.RESOLVED,
                matched_entry_ids=[entry.entry_id],
                matched_entries=[entry],
                confidence=0.90,
                reasoning=reasoning,
                provenance=provenance,
                resolver_version=self.VERSION,
            )
        elif len(matches) > 1:
            entry_ids = [e.entry_id for e in matches]
            reasoning = (
                f"Author-year citation '{norm_ref.raw_reference}' is ambiguous: "
                f"matched {len(matches)} publications for author '{target_surname.capitalize()}' in {target_year} ({entry_ids})."
            )
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.AMBIGUOUS,
                matched_entry_ids=entry_ids,
                matched_entries=matches,
                confidence=0.70,
                reasoning=reasoning,
                provenance={"conflicting_entry_ids": entry_ids},
                resolver_version=self.VERSION,
            )
        else:
            reasoning = (
                f"Author-year citation '{norm_ref.raw_reference}' was not found in the bibliography "
                f"(no entry matching author '{target_surname.capitalize()}' and year {target_year})."
            )
            return ReferenceResolutionResult(
                resolution_id=res_id,
                claim_id=claim_id,
                document_id=document_id,
                reference_text=norm_ref.raw_reference,
                normalized_reference=norm_ref.normalized_string,
                reference_type=norm_ref.reference_type,
                status=ResolutionStatus.NOT_FOUND,
                confidence=0.95,
                reasoning=reasoning,
                provenance={"author_surname": target_surname, "publication_year": target_year},
                resolver_version=self.VERSION,
            )
