"""Structure-aware chunker for scientific documents.

Preserves section hierarchy, parent-child linking, exact page provenance,
tables as Markdown, figures/captions, and reference sections.
"""

import logging
from typing import List, Optional

from src.chunking.base_chunker import (
    BaseChunker,
    clean_section_slug,
    normalize_source_identifier,
    pack_sentences,
    split_into_sentences,
)
from src.config.settings import Settings, get_settings
from src.models.schemas import (
    ContentType,
    Domain,
    DocumentParsingResult,
    EvidenceChunk,
    ParsedElement,
)

logger = logging.getLogger(__name__)


class ScientificChunker(BaseChunker):
    """Structure-aware chunker for scientific research papers and reports."""

    def __init__(
        self,
        max_characters: int = 1500,
        min_characters: int = 30,
        overlap_characters: int = 150,
        settings: Optional[Settings] = None,
    ):
        """Initialize ScientificChunker with configurable size boundaries.

        Args:
            max_characters: Maximum characters per child chunk before sentence-splitting.
            min_characters: Minimum characters for a standalone paragraph chunk.
            overlap_characters: Overlap characters between sentence-split sub-chunks.
            settings: Project settings singleton.
        """
        self.max_characters = max_characters
        self.min_characters = min_characters
        self.overlap_characters = overlap_characters
        self.settings = settings or get_settings()

    def chunk_document(self, parsed_doc: DocumentParsingResult) -> List[EvidenceChunk]:
        """Convert a parsed scientific document into validated EvidenceChunks.

        Maintains parent-child relationships where sections serve as parents
        and paragraphs, tables, and figures serve as child evidence chunks.
        """
        if parsed_doc.domain != Domain.SCIENTIFIC:
            logger.warning(
                "Document %s has domain %s, but ScientificChunker was invoked.",
                parsed_doc.document_id,
                parsed_doc.domain.value,
            )

        chunks: List[EvidenceChunk] = []

        doc_id = parsed_doc.document_id
        title = parsed_doc.title
        source = normalize_source_identifier(parsed_doc.source_path, domain=parsed_doc.domain.value)

        # State tracking for sections
        current_section_name = "Document Title"
        current_section_id = "sec_000_title"
        current_parent_chunk_id: Optional[str] = None

        section_seq = 0
        chunk_seq = 0

        # Optional metadata attributes from parsed document
        authors = parsed_doc.metadata.get("authors")
        year = parsed_doc.metadata.get("year")
        doi = parsed_doc.metadata.get("doi")

        for elem in parsed_doc.elements:
            elem_text = elem.text.strip()
            if not elem_text:
                continue

            elem_page = elem.page if elem.page is not None and elem.page >= 1 else 1

            # Determine if this element is a section heading or title
            if elem.element_type in {"title", "heading"}:
                section_seq += 1
                current_section_name = elem_text
                slug = clean_section_slug(elem_text)
                current_section_id = f"sec_{section_seq:03d}_{slug}"
                current_parent_chunk_id = f"{doc_id}_section_{section_seq:03d}"

                # Decide content_type for section
                lowered_sec = elem_text.lower()
                if "abstract" in lowered_sec:
                    sec_content_type = ContentType.ABSTRACT
                elif "reference" in lowered_sec or "bibliography" in lowered_sec:
                    sec_content_type = ContentType.REFERENCE
                elif elem.element_type == "title":
                    sec_content_type = ContentType.HEADING
                else:
                    sec_content_type = ContentType.SECTION

                # Create the Parent Section EvidenceChunk
                parent_chunk = EvidenceChunk(
                    chunk_id=current_parent_chunk_id,
                    document_id=doc_id,
                    domain=Domain.SCIENTIFIC,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=None,
                    content_type=sec_content_type,
                    text=elem_text,
                    authors=authors,
                    year=year,
                    doi=doi,
                )
                chunks.append(parent_chunk)
                continue

            # Ensure we have an active parent section; if none yet, create a default root
            if current_parent_chunk_id is None:
                section_seq += 1
                current_section_name = "General"
                current_section_id = f"sec_{section_seq:03d}_general"
                current_parent_chunk_id = f"{doc_id}_section_{section_seq:03d}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=current_parent_chunk_id,
                        document_id=doc_id,
                        domain=Domain.SCIENTIFIC,
                        title=title,
                        source=source,
                        page=elem_page,
                        section=current_section_name,
                        section_id=current_section_id,
                        parent_id=None,
                        content_type=ContentType.SECTION,
                        text=current_section_name,
                        authors=authors,
                        year=year,
                        doi=doi,
                    )
                )

            # Handle Tables
            if elem.element_type == "table":
                chunk_seq += 1
                tbl_id = elem.metadata.get("table_id") or f"{doc_id}_table_{chunk_seq:03d}"
                chunk_id = f"{doc_id}_{current_section_id}_table_{chunk_seq:03d}"
                table_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.SCIENTIFIC,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=current_parent_chunk_id,
                    content_type=ContentType.TABLE,
                    table_id=tbl_id,
                    text=elem_text,
                    authors=authors,
                    year=year,
                    doi=doi,
                )
                chunks.append(table_chunk)
                continue

            # Handle Figures / Captions
            if elem.element_type in {"figure", "caption"}:
                chunk_seq += 1
                fig_id = elem.metadata.get("figure_id") or f"Fig_{chunk_seq:03d}"
                chunk_id = f"{doc_id}_{current_section_id}_fig_{chunk_seq:03d}"
                content_type = ContentType.FIGURE if elem.element_type == "figure" else ContentType.CAPTION
                fig_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.SCIENTIFIC,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=current_parent_chunk_id,
                    content_type=content_type,
                    figure_id=fig_id,
                    text=elem_text,
                    authors=authors,
                    year=year,
                    doi=doi,
                )
                chunks.append(fig_chunk)
                continue

            # Filter out micro-chunks that do not carry semantic content
            if len(elem_text) < self.min_characters and elem.element_type not in {"reference"}:
                logger.debug(
                    "Skipping micro element of length %d: %s",
                    len(elem_text),
                    elem_text[:30],
                )
                continue

            # Determine content_type for text paragraphs / references
            lowered_sec = current_section_name.lower()
            if "abstract" in lowered_sec:
                child_content_type = ContentType.ABSTRACT
            elif "reference" in lowered_sec or "bibliography" in lowered_sec or elem.element_type == "reference":
                child_content_type = ContentType.REFERENCE
            else:
                child_content_type = ContentType.PARAGRAPH

            # Check if paragraph is oversized and needs sentence-level packing
            if len(elem_text) > self.max_characters:
                sentences = split_into_sentences(elem_text)
                packed_chunks = pack_sentences(
                    sentences,
                    max_chars=self.max_characters,
                    overlap_chars=self.overlap_characters,
                )
                for sub_idx, sub_text in enumerate(packed_chunks, 1):
                    chunk_seq += 1
                    chunk_id = f"{doc_id}_{current_section_id}_chunk_{chunk_seq:03d}"
                    child_chunk = EvidenceChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        domain=Domain.SCIENTIFIC,
                        title=title,
                        source=source,
                        page=elem_page,
                        section=current_section_name,
                        section_id=current_section_id,
                        parent_id=current_parent_chunk_id,
                        content_type=child_content_type,
                        text=sub_text,
                        authors=authors,
                        year=year,
                        doi=doi,
                    )
                    chunks.append(child_chunk)
            else:
                # Keep reasonably-sized paragraph intact
                chunk_seq += 1
                chunk_id = f"{doc_id}_{current_section_id}_chunk_{chunk_seq:03d}"
                child_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.SCIENTIFIC,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=current_parent_chunk_id,
                    content_type=child_content_type,
                    text=elem_text,
                    authors=authors,
                    year=year,
                    doi=doi,
                )
                chunks.append(child_chunk)

        # Validate complete chunk collection
        self.validate_chunks(chunks)

        logger.info(
            "Scientific chunking complete: generated %d chunks from %s",
            len(chunks),
            doc_id,
        )
        return chunks
