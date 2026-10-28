"""Chunking module for scientific and legal documents."""

from src.chunking.base_chunker import (
    BaseChunker,
    clean_section_slug,
    normalize_source_identifier,
    pack_sentences,
    split_into_sentences,
)
from src.chunking.legal_chunker import LegalChunker
from src.chunking.scientific_chunker import ScientificChunker

__all__ = [
    "BaseChunker",
    "LegalChunker",
    "ScientificChunker",
    "clean_section_slug",
    "normalize_source_identifier",
    "pack_sentences",
    "split_into_sentences",
]
