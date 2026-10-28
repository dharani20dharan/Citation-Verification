"""Serialization and adaptation of EvidenceChunk metadata for ChromaDB storage."""

import json
from typing import Any, Dict

from src.models.schemas import EvidenceChunk


def serialize_chunk_metadata_for_chroma(chunk: EvidenceChunk) -> Dict[str, Any]:
    """Convert an EvidenceChunk into a Chroma-compatible metadata dictionary.

    ChromaDB metadata supports strings, integers, floats, and booleans.
    None values must be omitted. Lists are converted to deterministic JSON strings.

    Args:
        chunk: Canonical EvidenceChunk instance.

    Returns:
        Dictionary containing only ChromaDB-safe scalar types.
    """
    metadata: Dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "domain": chunk.domain.value,
        "title": chunk.title,
        "source": chunk.source,
        "page": int(chunk.page),
        "section": chunk.section,
        "section_id": chunk.section_id,
        "content_type": chunk.content_type.value,
        "metadata_version": chunk.metadata_version,
    }

    if chunk.parent_id is not None:
        metadata["parent_id"] = chunk.parent_id

    # Optional bibliographic / scientific attributes
    if chunk.authors:
        metadata["authors"] = json.dumps(chunk.authors)
    if chunk.year is not None:
        metadata["year"] = int(chunk.year)
    if chunk.doi:
        metadata["doi"] = chunk.doi

    # Optional legal attributes
    if chunk.case_name:
        metadata["case_name"] = chunk.case_name
    if chunk.statute_name:
        metadata["statute_name"] = chunk.statute_name
    if chunk.clause_id:
        metadata["clause_id"] = chunk.clause_id
    if chunk.subsection_id:
        metadata["subsection_id"] = chunk.subsection_id
    if chunk.paragraph_id:
        metadata["paragraph_id"] = chunk.paragraph_id

    # Optional tabular / figure attributes
    if chunk.figure_id:
        metadata["figure_id"] = chunk.figure_id
    if chunk.table_id:
        metadata["table_id"] = chunk.table_id

    return metadata
