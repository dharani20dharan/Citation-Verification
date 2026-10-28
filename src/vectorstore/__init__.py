"""Vector store module for persistent ChromaDB storage."""

from src.vectorstore.chroma_store import ChromaVectorStore, IngestionReport
from src.vectorstore.metadata import serialize_chunk_metadata_for_chroma

__all__ = [
    "ChromaVectorStore",
    "IngestionReport",
    "serialize_chunk_metadata_for_chroma",
]
