"""Domain-specific embeddings module for scientific and legal literature."""

from src.embeddings.base_embedder import BaseEmbedder, resolve_device
from src.embeddings.factory import get_embedder
from src.embeddings.legal_embedder import LegalEmbedder
from src.embeddings.scientific_embedder import ScientificEmbedder

__all__ = [
    "BaseEmbedder",
    "resolve_device",
    "ScientificEmbedder",
    "LegalEmbedder",
    "get_embedder",
]
