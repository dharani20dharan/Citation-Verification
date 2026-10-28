"""Canonical data models for the retrieval engine."""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator

from src.models.schemas import ContentType, Domain


class MatchType(str, Enum):
    """Classification of how a retrieval candidate was matched."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    PARENT_CONTEXT = "parent_context"


class RetrievalRequest(BaseModel):
    """Specification of a retrieval query across the evidence store."""

    query: str = Field(
        ...,
        description="Search text or claim to be verified",
        min_length=1,
    )
    domain: Domain = Field(
        ...,
        description="Target corpus domain ('legal' or 'scientific')",
    )
    top_k: int = Field(
        default=5,
        description="Maximum semantic candidates to retrieve",
        ge=1,
    )
    exact_identifier: Optional[str] = Field(
        default=None,
        description="Specific structural identifier (e.g. '3(d)', 'Section 3(e)', '3(d)(i)')",
    )
    expand_parent_context: bool = Field(
        default=True,
        description="Whether to walk parent_id links and retrieve enclosing section contexts",
    )
    semantic_search_enabled: bool = Field(
        default=True,
        description="Whether to perform dense vector similarity search",
    )
    max_parent_depth: int = Field(
        default=2,
        description="Maximum parent hierarchy levels to ascend",
        ge=0,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")
        return v.strip()


class RetrievalResult(BaseModel):
    """Ranked evidence candidate preserving complete provenance and auditability."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document identifier")
    domain: Domain = Field(..., description="Corpus domain")
    text: str = Field(..., description="Canonical evidence text from source chunk")

    score: float = Field(
        ...,
        description="Normalized ranking score (1.0 for exact matches, cosine similarity for semantic)",
    )
    match_type: str = Field(
        ...,
        description="Classification: 'exact', 'semantic', or 'parent_context'",
    )
    distance: Optional[float] = Field(
        default=None,
        description="Raw vector distance from ChromaDB (if retrieved semantically)",
    )
    similarity: Optional[float] = Field(
        default=None,
        description="Calculated cosine similarity: max(0.0, 1.0 - distance)",
    )

    # Provenance fields
    title: Optional[str] = Field(default=None, description="Document title")
    source: Optional[str] = Field(default=None, description="Source file or relative path")
    page: Optional[int] = Field(default=None, description="Page number in original document")

    section: Optional[str] = Field(default=None, description="Section heading")
    section_id: Optional[str] = Field(default=None, description="Deterministic section slug")
    parent_id: Optional[str] = Field(default=None, description="Parent chunk identifier")

    content_type: Union[ContentType, str] = Field(..., description="Structural content type")

    # Legal attributes
    clause_id: Optional[str] = Field(default=None, description="Exact statutory clause identifier")
    subsection_id: Optional[str] = Field(default=None, description="Exact subsection identifier")
    paragraph_id: Optional[str] = Field(default=None, description="Numbered paragraph identifier")
    case_name: Optional[str] = Field(default=None, description="Associated case title")
    statute_name: Optional[str] = Field(default=None, description="Enacting statute name")

    # Scientific / visual attributes
    table_id: Optional[str] = Field(default=None, description="Table identifier")
    figure_id: Optional[str] = Field(default=None, description="Figure identifier")

    rank: int = Field(default=1, description="Deterministic final ranking position", ge=1)
