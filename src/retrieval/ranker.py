"""Deterministic ranking and deduplication policy for retrieval candidates."""

import logging
from typing import Dict, List

from src.retrieval.models import MatchType, RetrievalResult

logger = logging.getLogger(__name__)

# Priority ordering for match types
MATCH_TYPE_PRIORITY = {
    MatchType.EXACT.value: 3,
    MatchType.SEMANTIC.value: 2,
    MatchType.PARENT_CONTEXT.value: 1,
}


class DeterministicRanker:
    """Ranks and deduplicates retrieval candidates with strict deterministic tie-breaking."""

    def rank(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate candidates by chunk_id and sort into final ranked order.

        Deduplication rule:
            Keeps the strongest match type (exact > semantic > parent_context).

        Ranking order:
            1. Match type category: exact > semantic > parent_context
            2. Score: higher score first
            3. Tie-breaking: document_id -> page -> chunk_id

        Args:
            candidates: Unfiltered collection of candidate RetrievalResults.

        Returns:
            Deduplicated, sorted list of RetrievalResult objects with sequential rank 1..N.
        """
        if not candidates:
            return []

        # 1. Deduplicate by chunk_id, preserving the highest priority match type
        deduped: Dict[str, RetrievalResult] = {}
        for cand in candidates:
            cid = cand.chunk_id
            if cid not in deduped:
                deduped[cid] = cand
            else:
                existing = deduped[cid]
                existing_prio = MATCH_TYPE_PRIORITY.get(existing.match_type, 0)
                cand_prio = MATCH_TYPE_PRIORITY.get(cand.match_type, 0)

                if cand_prio > existing_prio or (cand_prio == existing_prio and cand.score > existing.score):
                    deduped[cid] = cand

        unique_list = list(deduped.values())

        # 2. Deterministic sort key
        def sort_key(res: RetrievalResult):
            prio = MATCH_TYPE_PRIORITY.get(res.match_type, 0)
            page_val = res.page if res.page is not None else 0
            # Sort:
            # -prio (descending: 3, 2, 1)
            # -score (descending)
            # document_id (ascending string)
            # page_val (ascending int)
            # chunk_id (ascending string)
            return (
                -prio,
                -round(res.score, 6),
                res.document_id,
                page_val,
                res.chunk_id,
            )

        sorted_list = sorted(unique_list, key=sort_key)

        # 3. Assign 1-indexed sequential ranks
        final_results: List[RetrievalResult] = []
        for idx, item in enumerate(sorted_list, start=1):
            ranked_item = item.model_copy(update={"rank": idx})
            final_results.append(ranked_item)

        return final_results
