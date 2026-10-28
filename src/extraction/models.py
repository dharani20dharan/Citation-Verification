"""Data models for citation and claim extraction."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from src.models.schemas import Domain


class ReferenceType(str, Enum):
    """Supported reference and citation categories."""
    NUMERIC = "numeric_citation"
    AUTHOR_YEAR = "author_year"
    LEGAL_SECTION = "legal_section"
    LEGAL_ARTICLE = "legal_article"
    LEGAL_CLAUSE = "legal_clause"
    LEGAL_RULE = "legal_rule"


class DetectedReference(BaseModel):
    """Internal model for an identified citation or statutory reference within text."""
    reference_text: str = Field(..., description="Exact raw reference text as found in document")
    normalized_reference: str = Field(..., description="Normalized canonical identifier")
    normalized_references: List[str] = Field(
        default_factory=list,
        description="Structured list of normalized identifiers (e.g. ['3', '4'] for [3, 4])",
    )
    reference_type: str = Field(..., description="Classification category from ReferenceType")
    start: int = Field(..., description="Start character offset in source text")
    end: int = Field(..., description="End character offset in source text")

    @field_validator("reference_text", "normalized_reference")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reference value must not be empty or whitespace")
        return v.strip()


class CitationClaim(BaseModel):
    """Structured extraction unit pairing a substantive claim with its citation/reference."""

    claim_id: str = Field(
        ...,
        description="Deterministic unique claim identifier: <document_id>_claim_<seq>",
        min_length=1,
    )
    document_id: str = Field(
        ...,
        description="Source document identifier",
        min_length=1,
    )
    domain: Domain = Field(
        ...,
        description="Domain: legal or scientific",
    )

    claim_text: str = Field(
        ...,
        description="Substantive statement text with citation marker removed",
        min_length=1,
    )

    reference_text: str = Field(
        ...,
        description="Exact raw citation/reference text as it appeared in source",
        min_length=1,
    )
    normalized_reference: str = Field(
        ...,
        description="Canonical reference identifier without prefixes (e.g. '3(d)', '12', '5-7')",
        min_length=1,
    )
    normalized_references: List[str] = Field(
        default_factory=list,
        description="List of individual normalized references when multiple are cited",
    )

    reference_type: str = Field(
        ...,
        description="Category classification (e.g. numeric_citation, legal_section)",
    )

    # Provenance tracking
    source_chunk_id: Optional[str] = Field(
        default=None,
        description="Originating EvidenceChunk identifier",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in source document",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section title or heading in source document",
    )
    section_id: Optional[str] = Field(
        default=None,
        description="Normalized section identifier",
    )

    # Character offsets in source chunk text
    character_start: Optional[int] = Field(
        default=None,
        description="Start character index of claim in source chunk text",
    )
    character_end: Optional[int] = Field(
        default=None,
        description="End character index of claim in source chunk text",
    )

    # Context window
    context_before: Optional[str] = Field(
        default=None,
        description="Surrounding text immediately before the claim (up to 100 chars)",
    )
    context_after: Optional[str] = Field(
        default=None,
        description="Surrounding text immediately after the claim (up to 100 chars)",
    )

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("claim_text must not be empty or whitespace only")
        return cleaned

    @field_validator("reference_text", "normalized_reference")
    @classmethod
    def validate_reference_fields(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Reference fields must not be empty or whitespace only")
        return cleaned
