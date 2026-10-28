"""Evidence selector prioritizing exact reference matches and structural context."""

import logging
from typing import Any, Dict, List, Optional, Set
from src.retrieval.models import MatchType, RetrievalResult
from src.verification.models import EvidenceItem

logger = logging.getLogger(__name__)


class EvidenceSelector:
    """Selects and prioritizes the most relevant evidence candidates deterministically.

    Priority order:
        1. Exact reference matches
        2. Child / supporting chunks belonging to the exact reference
        3. Parent context
        4. Semantic matches
    """

    def __init__(self, max_evidence: int = 5):
        """Initialize selector with configurable evidence limit.

        Args:
            max_evidence: Maximum number of evidence items to select (default: 5).
        """
        if max_evidence < 1:
            raise ValueError(f"max_evidence must be >= 1, got {max_evidence}")
        self.max_evidence = max_evidence

    def select_evidence(
        self,
        retrieval_results: List[RetrievalResult],
        max_evidence: Optional[int] = None,
    ) -> List[EvidenceItem]:
        """Select, rank, deduplicate, and convert retrieval results to EvidenceItems.

        Args:
            retrieval_results: Ranked list of RetrievalResult candidates from Stage 7.
            max_evidence: Optional override for the maximum evidence count.

        Returns:
            Deterministic, prioritized List of auditable EvidenceItem objects.
        """
        limit = max_evidence if max_evidence is not None else self.max_evidence
        if not retrieval_results:
            return []

        # 1. Identify exact matches to resolve child relationships
        exact_chunk_ids: Set[str] = set()
        exact_section_ids: Set[str] = set()
        for res in retrieval_results:
            if res.match_type == MatchType.EXACT.value:
                exact_chunk_ids.add(res.chunk_id)
                if res.section_id:
                    exact_section_ids.add(res.section_id)

        # 2. Partition into 4 strict priority tiers
        tier_exact: List[RetrievalResult] = []
        tier_children: List[RetrievalResult] = []
        tier_parent: List[RetrievalResult] = []
        tier_semantic: List[RetrievalResult] = []

        for res in retrieval_results:
            if res.match_type == MatchType.EXACT.value:
                tier_exact.append(res)
            elif (
                (res.parent_id and res.parent_id in exact_chunk_ids)
                or (res.section_id and res.section_id in exact_section_ids and res.chunk_id not in exact_chunk_ids)
            ):
                tier_children.append(res)
            elif res.match_type == MatchType.PARENT_CONTEXT.value:
                tier_parent.append(res)
            else:
                tier_semantic.append(res)

        # Helper to determine if a retrieval result is purely structural (heading/title)
        def is_structural_result(res: RetrievalResult) -> bool:
            ct = res.content_type.value if hasattr(res.content_type, "value") else str(res.content_type)
            return ct.lower() in ("heading", "title", "section")

        # Sort each tier deterministically: substantive first (0 before 1), score DESC, chunk_id ASC
        def sort_tier(items: List[RetrievalResult]) -> List[RetrievalResult]:
            return sorted(
                items,
                key=lambda x: (1 if is_structural_result(x) else 0, -x.score, x.chunk_id),
            )

        ordered_candidates = (
            sort_tier(tier_exact)
            + sort_tier(tier_children)
            + sort_tier(tier_parent)
            + sort_tier(tier_semantic)
        )

        # 3. Deduplicate by chunk_id preserving highest priority appearance
        selected_evidence: List[EvidenceItem] = []
        seen_chunk_ids: Set[str] = set()

        for res in ordered_candidates:
            if res.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(res.chunk_id)

            is_struct = is_structural_result(res)
            provenance: Dict[str, Any] = {
                "document_id": res.document_id,
                "title": res.title,
                "source": res.source,
                "page": res.page,
                "section": res.section,
                "section_id": res.section_id,
                "parent_id": res.parent_id,
                "content_type": res.content_type.value if hasattr(res.content_type, "value") else str(res.content_type),
                "clause_id": res.clause_id,
                "subsection_id": res.subsection_id,
                "paragraph_id": res.paragraph_id,
                "case_name": res.case_name,
                "statute_name": res.statute_name,
                "table_id": res.table_id,
                "figure_id": res.figure_id,
                "retrieval_rank": res.rank,
                "match_type": res.match_type,
                "retrieval_score": res.score,
                "distance": res.distance,
                "similarity": res.similarity,
                "is_structural": is_struct,
            }

            item = EvidenceItem(
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                text=res.text,
                match_type=res.match_type,
                score=res.score,
                similarity=res.similarity,
                is_structural=is_struct,
                evidence_type="structural" if is_struct else "substantive",
                provenance=provenance,
            )
            selected_evidence.append(item)

            if len(selected_evidence) >= limit:
                break

        logger.debug(
            "Selected %d evidence items from %d candidate results (limit=%d)",
            len(selected_evidence),
            len(retrieval_results),
            limit,
        )
        return selected_evidence
