"""Context expander traversing structural parent-child relationships in ChromaDB."""

import logging
from typing import Dict, List, Set
from chromadb.api.models.Collection import Collection

from src.models.schemas import ContentType, Domain
from src.retrieval.models import MatchType, RetrievalResult

logger = logging.getLogger(__name__)


class ContextExpander:
    """Ascends parent_id chains to retrieve surrounding structural context."""

    def expand(
        self,
        collection: Collection,
        domain: Domain,
        candidate_results: List[RetrievalResult],
        max_depth: int = 2,
    ) -> List[RetrievalResult]:
        """Retrieve parent structural chunks for the given candidates.

        Args:
            collection: ChromaDB persistent collection.
            domain: Domain enum.
            candidate_results: Already retrieved exact/semantic results.
            max_depth: Maximum levels of parent hierarchy to ascend (default: 2).

        Returns:
            List of newly retrieved parent RetrievalResult instances.
        """
        if max_depth <= 0 or not candidate_results:
            return []

        # Track IDs already known to avoid duplicates
        existing_ids: Set[str] = {c.chunk_id for c in candidate_results}
        expanded_parents: Dict[str, RetrievalResult] = {}

        for candidate in candidate_results:
            curr_parent_id = candidate.parent_id
            depth = 0

            while curr_parent_id and depth < max_depth:
                if curr_parent_id in existing_ids:
                    # Already in initial candidates; look up its parent if any to continue climbing
                    matching = [c for c in candidate_results if c.chunk_id == curr_parent_id]
                    if matching and matching[0].parent_id:
                        curr_parent_id = matching[0].parent_id
                        depth += 1
                        continue
                    break

                if curr_parent_id in expanded_parents:
                    # Already fetched in this expansion round; continue climbing its parent
                    curr_parent_id = expanded_parents[curr_parent_id].parent_id
                    depth += 1
                    continue

                # Fetch parent record from collection
                try:
                    res = collection.get(
                        ids=[curr_parent_id],
                        include=["documents", "metadatas"],
                    )
                except Exception as exc:
                    logger.debug("Failed to fetch parent %s: %s", curr_parent_id, exc)
                    break

                if not res or not res.get("ids"):
                    break

                meta = res["metadatas"][0] if res.get("metadatas") else {}
                doc = res["documents"][0] if res.get("documents") else ""

                parent_result = RetrievalResult(
                    chunk_id=curr_parent_id,
                    document_id=meta.get("document_id", candidate.document_id),
                    domain=domain,
                    text=doc,
                    score=round(candidate.score * 0.9, 4),  # Scaled relative to child
                    match_type=MatchType.PARENT_CONTEXT.value,
                    distance=None,
                    similarity=None,
                    title=meta.get("title"),
                    source=meta.get("source"),
                    page=meta.get("page"),
                    section=meta.get("section"),
                    section_id=meta.get("section_id"),
                    parent_id=meta.get("parent_id"),
                    content_type=meta.get("content_type", ContentType.SECTION.value),
                    clause_id=meta.get("clause_id"),
                    subsection_id=meta.get("subsection_id"),
                    paragraph_id=meta.get("paragraph_id"),
                    case_name=meta.get("case_name"),
                    statute_name=meta.get("statute_name"),
                    table_id=meta.get("table_id"),
                    figure_id=meta.get("figure_id"),
                    rank=999,
                )
                expanded_parents[curr_parent_id] = parent_result
                curr_parent_id = meta.get("parent_id")
                depth += 1

        logger.info(
            "Context expander retrieved %d parent context chunks (max_depth=%d)",
            len(expanded_parents),
            max_depth,
        )
        return list(expanded_parents.values())
