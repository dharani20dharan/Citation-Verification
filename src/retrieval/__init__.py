"""Retrieval engine module for legal and scientific evidence corpora."""

from src.retrieval.context_expander import ContextExpander
from src.retrieval.exact_match import ExactMatcher
from src.retrieval.models import MatchType, RetrievalRequest, RetrievalResult
from src.retrieval.ranker import DeterministicRanker
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_search import SemanticSearcher

__all__ = [
    "MatchType",
    "RetrievalRequest",
    "RetrievalResult",
    "ExactMatcher",
    "SemanticSearcher",
    "ContextExpander",
    "DeterministicRanker",
    "Retriever",
]
