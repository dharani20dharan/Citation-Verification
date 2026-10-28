"""Unified Retrieval API and engine for legal and scientific evidence corpora."""

import logging
from typing import List, Optional

from src.config.settings import Settings, get_settings
from src.embeddings.base_embedder import BaseEmbedder
from src.retrieval.context_expander import ContextExpander
from src.retrieval.exact_match import ExactMatcher
from src.retrieval.models import RetrievalRequest, RetrievalResult
from src.retrieval.ranker import DeterministicRanker
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval engine integrating exact matching, semantic search, and context expansion."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        settings: Optional[Settings] = None,
        embedder_override: Optional[BaseEmbedder] = None,
    ):
        """Initialize the retrieval engine.

        Args:
            vector_store: ChromaVectorStore instance (defaults to persistent store).
            settings: Project Settings singleton.
            embedder_override: Optional embedder override (useful for fast unit testing).
        """
        self.settings = settings or get_settings()
        self.vector_store = vector_store or ChromaVectorStore(settings=self.settings)
        self._embedder_override = embedder_override

        self.exact_matcher = ExactMatcher()
        self.semantic_searcher = SemanticSearcher(embedder=self._embedder_override)
        self.context_expander = ContextExpander()
        self.ranker = DeterministicRanker()

    def retrieve(self, request: RetrievalRequest) -> List[RetrievalResult]:
        """Execute structured retrieval for a given request.

        Flow:
            1. Validate request
            2. Exact-match lookup (if exact_identifier provided)
            3. Semantic search (if semantic_search_enabled)
            4. Parent-context expansion (if expand_parent_context)
            5. Deterministic ranking and deduplication

        Args:
            request: Validated RetrievalRequest specification.

        Returns:
            List of ranked, provenance-preserving RetrievalResult objects.
        """
        collection = self.vector_store.get_or_create_collection(request.domain)

        exact_candidates: List[RetrievalResult] = []
        if request.exact_identifier:
            exact_candidates = self.exact_matcher.match(
                collection=collection,
                identifier=request.exact_identifier,
                domain=request.domain,
            )

        semantic_candidates: List[RetrievalResult] = []
        if request.semantic_search_enabled:
            semantic_candidates = self.semantic_searcher.search(
                collection=collection,
                domain=request.domain,
                query=request.query,
                top_k=request.top_k,
                embedder_override=self._embedder_override,
            )

        initial_pool = exact_candidates + semantic_candidates

        # Parent-context expansion
        parent_candidates: List[RetrievalResult] = []
        if request.expand_parent_context and request.max_parent_depth > 0:
            parent_candidates = self.context_expander.expand(
                collection=collection,
                domain=request.domain,
                candidate_results=initial_pool,
                max_depth=request.max_parent_depth,
            )

        all_candidates = initial_pool + parent_candidates

        # Deterministic ranking and deduplication
        ranked_results = self.ranker.rank(all_candidates)

        logger.info(
            "Retrieved %d total candidates (exact: %d, semantic: %d, parent: %d) for query '%s'",
            len(ranked_results),
            len(exact_candidates),
            len(semantic_candidates),
            len(parent_candidates),
            request.query[:40],
        )
        return ranked_results
