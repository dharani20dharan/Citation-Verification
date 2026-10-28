"""Data models and schemas for Stage 10A: Bibliography and Reference Resolution."""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.models.schemas import Domain


class ResolutionStatus(str, Enum):
    """Controlled status enumeration for reference resolution outcomes."""
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class BibliographicEntry(BaseModel):
    """Structured bibliographic entry extracted from a document's reference list."""

    entry_id: str = Field(..., description="Deterministic unique identifier, e.g. {document_id}_ref_{norm_label}", min_length=1)
    document_id: str = Field(..., description="Identifier of the document containing this entry", min_length=1)
    domain: Domain = Field(default=Domain.SCIENTIFIC, description="Corpus domain ('scientific' or 'legal')")

    reference_label: str = Field(..., description="Original raw citation label, e.g. '[12]' or '1.' or '(Smith, 2024)'")
    normalized_label: str = Field(..., description="Canonical reference identifier, e.g. '12' or 'Smith_2024'")

    raw_text: str = Field(..., description="Complete, unmodified original text of the reference entry", min_length=1)

    title: Optional[str] = Field(default=None, description="Extracted title of the cited work")
    authors: List[str] = Field(default_factory=list, description="Extracted list of author names")
    year: Optional[int] = Field(default=None, description="Extracted publication year")

    journal: Optional[str] = Field(default=None, description="Journal or publication venue")
    conference: Optional[str] = Field(default=None, description="Conference proceedings name")
    publisher: Optional[str] = Field(default=None, description="Publisher name")

    doi: Optional[str] = Field(default=None, description="Digital Object Identifier (DOI)")
    url: Optional[str] = Field(default=None, description="Direct URL to cited work")

    page: Optional[int] = Field(default=None, description="Source document page number")
    section: Optional[str] = Field(default=None, description="Source section title (e.g. 'References')")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auditable metadata preserving source, page, section, chunk_id, etc.",
    )

    metadata_version: str = Field(default="1.0", description="Schema metadata version")

    @field_validator("entry_id", "document_id")
    @classmethod
    def validate_non_empty_ids(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Identifier must not be empty or whitespace only")
        return cleaned

    @property
    def chunk_id(self) -> Optional[str]:
        return self.provenance.get("chunk_id")


class ReferenceResolutionResult(BaseModel):
    """Structured, auditable result of resolving a citation identifier to bibliography."""

    resolution_id: str = Field(..., description="Deterministic resolution identifier", min_length=1)
    claim_id: Optional[str] = Field(default=None, description="Claim identifier if resolving from a CitationClaim")
    document_id: Optional[str] = Field(default=None, description="Source document identifier")

    reference_text: str = Field(..., description="Exact reference text as cited in text")
    normalized_reference: str = Field(..., description="Canonical reference identifier or range")
    reference_type: str = Field(..., description="Classification category of the reference")

    status: ResolutionStatus = Field(..., description="Resolution outcome status")

    matched_entry_ids: List[str] = Field(
        default_factory=list,
        description="List of entry_ids for uniquely matched bibliographic entries",
    )
    matched_entries: List[BibliographicEntry] = Field(
        default_factory=list,
        description="List of matched BibliographicEntry objects",
    )

    confidence: float = Field(
        ...,
        description="Deterministic confidence tier between 0.0 and 1.0 (not a statistical probability)",
        ge=0.0,
        le=1.0,
    )

    reasoning: str = Field(..., description="Transparent explanation of resolution outcome", min_length=1)

    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete provenance trace: claim -> citation -> bibliography -> source chunk",
    )

    resolver_version: str = Field(default="1.0.0-stage10a", description="Version of the reference resolver")

    @field_validator("confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return round(float(v), 4)


def generate_entry_id(document_id: str, normalized_label: str) -> str:
    """Generate a deterministic, stable identifier for a bibliographic entry."""
    clean_doc = document_id.strip()
    clean_label = normalized_label.strip().lower().replace(" ", "_").replace("-", "_")
    return f"{clean_doc}_ref_{clean_label}"


def generate_resolution_id(
    reference_text: str,
    document_id: Optional[str] = None,
    claim_id: Optional[str] = None,
) -> str:
    """Generate a deterministic resolution run identifier."""
    raw = f"{document_id or 'doc'}_{claim_id or 'claim'}_{reference_text.strip()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"res_{digest}"
