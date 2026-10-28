"""Canonical schemas for the citation verification offline corpus-building pipeline."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Domain(str, Enum):
    """Allowed corpus domains."""
    LEGAL = "legal"
    SCIENTIFIC = "scientific"


class ContentType(str, Enum):
    """Allowed structural content types."""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    CLAUSE = "clause"
    SUBSECTION = "subsection"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADING = "heading"
    ABSTRACT = "abstract"
    REFERENCE = "reference"


class EvidenceChunk(BaseModel):
    """Canonical Evidence Chunk schema.

    Represents a self-contained unit of searchable evidence with complete
    traceability back to original source, page, section, and hierarchy.
    """

    # Primary identification
    chunk_id: str = Field(
        ...,
        description="Globally unique chunk identifier: <document_id>_<structural_identifier>_<sequence>",
        min_length=1,
    )
    document_id: str = Field(
        ...,
        description="Deterministic stable document identifier: <domain>_<stable_identifier>",
        min_length=1,
    )
    domain: Domain = Field(
        ...,
        description="Domain classification: 'legal' or 'scientific'",
    )

    # Document & Source provenance
    title: str = Field(
        ...,
        description="Title of the source document",
        min_length=1,
    )
    source: str = Field(
        ...,
        description="Source identifier (e.g. filename, filepath, citation identifier)",
        min_length=1,
    )

    # Structural location
    page: int = Field(
        ...,
        description="1-based page number where the chunk appears",
        ge=1,
    )
    section: str = Field(
        ...,
        description="Name or title of the section containing this chunk",
    )
    section_id: str = Field(
        ...,
        description="Structural identifier of the section (e.g. 'sec_3_d', 'section_abstract')",
    )
    parent_id: Optional[str] = Field(
        None,
        description="chunk_id of the parent section/block if child chunk, or None if root",
    )

    # Content
    content_type: ContentType = Field(
        ...,
        description="Structural type of the content",
    )
    text: str = Field(
        ...,
        description="Raw clean text content of the chunk to be embedded",
        min_length=1,
    )

    # Versioning
    metadata_version: str = Field(
        default="1.0",
        description="Schema metadata version",
    )

    # Optional bibliographic / scientific attributes
    authors: Optional[List[str]] = Field(
        default=None,
        description="Author list for scientific publications",
    )
    year: Optional[int] = Field(
        default=None,
        description="Publication or enactment year",
    )
    doi: Optional[str] = Field(
        default=None,
        description="DOI for scientific documents",
    )

    # Optional legal attributes
    case_name: Optional[str] = Field(
        default=None,
        description="Case title or judgment citation name",
    )
    statute_name: Optional[str] = Field(
        default=None,
        description="Name of the statute or act",
    )
    clause_id: Optional[str] = Field(
        default=None,
        description="Exact clause identifier (e.g. 'clause_d')",
    )
    subsection_id: Optional[str] = Field(
        default=None,
        description="Exact subsection identifier (e.g. 'subsection_3')",
    )
    paragraph_id: Optional[str] = Field(
        default=None,
        description="Exact numbered paragraph identifier",
    )

    # Optional visual / tabular attributes
    figure_id: Optional[str] = Field(
        default=None,
        description="Figure identifier (e.g. 'Fig 1')",
    )
    table_id: Optional[str] = Field(
        default=None,
        description="Table identifier (e.g. 'Table 1')",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("EvidenceChunk text cannot be empty or whitespace only.")
        return stripped

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_identifier_format(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifiers cannot be empty.")
        return v.strip()

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to a flat, Chroma-compatible dictionary.

        ChromaDB metadata supports strings, integers, floats, and booleans.
        Lists are serialized as comma-separated strings or JSON.
        """
        metadata: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "domain": self.domain.value,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "section_id": self.section_id,
            "parent_id": self.parent_id if self.parent_id is not None else "",
            "content_type": self.content_type.value,
            "metadata_version": self.metadata_version,
        }

        # Optional fields
        if self.authors:
            metadata["authors"] = ", ".join(self.authors)
        if self.year is not None:
            metadata["year"] = self.year
        if self.doi:
            metadata["doi"] = self.doi
        if self.case_name:
            metadata["case_name"] = self.case_name
        if self.statute_name:
            metadata["statute_name"] = self.statute_name
        if self.clause_id:
            metadata["clause_id"] = self.clause_id
        if self.subsection_id:
            metadata["subsection_id"] = self.subsection_id
        if self.paragraph_id:
            metadata["paragraph_id"] = self.paragraph_id
        if self.figure_id:
            metadata["figure_id"] = self.figure_id
        if self.table_id:
            metadata["table_id"] = self.table_id

        return metadata


class ParsedElement(BaseModel):
    """Structured element extracted from document during parsing."""
    element_id: str = Field(..., description="Deterministic element identifier")
    element_type: str = Field(..., description="Element type (heading, paragraph, table, figure, caption, etc.)")
    text: str = Field(..., description="Clean text representation of this element")
    page: Optional[int] = Field(None, description="1-based page number where element appears")
    level: Optional[int] = Field(None, description="Hierarchical heading level (1 for H1, 2 for H2, etc.)")
    parent_element_id: Optional[str] = Field(None, description="element_id of parent structural node if known")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional item-specific metadata")


class ParsedTable(BaseModel):
    """Structured table representation extracted from document."""
    table_id: str = Field(..., description="Unique table identifier")
    page: Optional[int] = Field(None, description="1-based page number")
    markdown: str = Field(..., description="Markdown representation of the table")
    num_rows: int = Field(default=0, description="Total rows")
    num_cols: int = Field(default=0, description="Total columns")
    caption: Optional[str] = Field(None, description="Table caption if present")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentParsingResult(BaseModel):
    """Structured intermediate representation from Docling parsing."""
    document_id: str = Field(..., description="Deterministic stable document ID")
    domain: Domain = Field(..., description="Document domain")
    source_path: str = Field(..., description="Relative or normalized source path")
    title: str = Field(..., description="Document title")
    pages: int = Field(..., description="Total pages in document", ge=1)
    elements: List[ParsedElement] = Field(
        default_factory=list,
        description="Ordered sequence of structured document elements"
    )
    tables: List[ParsedTable] = Field(
        default_factory=list,
        description="Structured tables with markdown representations"
    )
    raw_structure: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full raw Docling JSON/dict export"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata"
    )


class ProcessingReport(BaseModel):
    """Machine-readable pipeline run execution report."""
    domain: str
    documents_discovered: int = 0
    documents_successfully_parsed: int = 0
    documents_failed: int = 0
    pages_processed: int = 0
    chunks_generated: int = 0
    chunks_rejected: int = 0
    chunks_embedded: int = 0
    chunks_indexed: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    chroma_collection: str = ""
    processing_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
