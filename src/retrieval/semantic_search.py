"""Dense semantic search using domain-specific embeddings and ChromaDB."""

import logging
from typing import List, Optional
from chromadb.api.models.Collection import Collection

from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.factory import get_embedder
from src.models.schemas import ContentType, Domain
from src.retrieval.models import MatchType, RetrievalResult

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Executes dense vector similarity search over persistent ChromaDB collections."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        """Initialize with optional pre-configured embedder (useful for testing)."""
        self._embedder = embedder

    def search(
        self,
        collection: Collection,
        domain: Domain,
        query: str,
        top_k: int = 5,
        embedder_override: Optional[BaseEmbedder] = None,
    ) -> List[RetrievalResult]:
        """Perform semantic search against the domain ChromaDB collection.

        Args:
            collection: ChromaDB persistent collection.
            domain: Target corpus domain ('legal' or 'scientific').
            query: Query text string.
            top_k: Maximum number of candidates to retrieve.
            embedder_override: Optional embedder override.

        Returns:
            List of semantic RetrievalResult objects ordered by cosine similarity.
        """
        if not query or not query.strip():
            return []

        active_embedder = embedder_override or self._embedder or get_embedder(domain)

        # 1. Generate query embedding (dense vector)
        query_vector = active_embedder.embed_text(query.strip())
        if not query_vector:
            logger.warning("Empty query embedding generated for '%s'", query)
            return []

        # 2. Query collection with distance preservation
        # Ensure we do not request more results than exist in collection
        col_count = collection.count()
        if col_count == 0:
            logger.info("Collection %s is empty; returning 0 semantic results.", collection.name)
            return []

        actual_n = min(top_k, col_count)
        raw_results = collection.query(
            query_embeddings=[query_vector],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        results: List[RetrievalResult] = []
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for i, (cid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
            distance_val = float(dist)
            # ChromaDB uses cosine distance: d = 1 - cosine_similarity.
            # Thus: cosine_similarity = 1.0 - d
            similarity_val = max(-1.0, min(1.0, 1.0 - distance_val))

            res = RetrievalResult(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                domain=domain,
                text=doc,
                score=similarity_val,
                match_type=MatchType.SEMANTIC.value,
                distance=distance_val,
                similarity=similarity_val,
                title=meta.get("title"),
                source=meta.get("source"),
                page=meta.get("page"),
                section=meta.get("section"),
                section_id=meta.get("section_id"),
                parent_id=meta.get("parent_id"),
                content_type=meta.get("content_type", ContentType.PARAGRAPH.value),
                clause_id=meta.get("clause_id"),
                subsection_id=meta.get("subsection_id"),
                paragraph_id=meta.get("paragraph_id"),
                case_name=meta.get("case_name"),
                statute_name=meta.get("statute_name"),
                table_id=meta.get("table_id"),
                figure_id=meta.get("figure_id"),
                rank=i,
            )
            results.append(res)

        logger.info(
            "Semantic search retrieved %d candidates from %s for query '%s'",
            len(results),
            collection.name,
            query[:40],
        )
        return results
