"""Structure-aware chunker for legal documents.

Preserves legal document hierarchy: Act -> Chapter -> Section -> Subsection -> Clause -> Subclause.
Ensures exact structural identifiers (e.g. Section 3(d) vs Section 3(e), 3(d)(i) vs 3(d)(ii))
remain distinctly identifiable and traceable to their parent context.
"""

import logging
import re
from typing import List, Optional, Tuple

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


# Regular expressions for legal structure detection
SECTION_HEADER_RE = re.compile(
    r"^(?:(?:Section|Sec\.)\s+(\d+[A-Za-z]?(?:\([0-9a-z]+\))*)|(?:(\d+[A-Za-z]?)\.\s+(?:Section|Sec\.)?))(.*)$",
    re.IGNORECASE,
)
EXACT_SECTION_NUM_RE = re.compile(
    r"(?:Section|Sec\.)\s+(\d+[A-Za-z]?(?:\([0-9a-z]+\))*)",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+)(?:\s*[:.-]\s*(.*))?$", re.IGNORECASE)
PART_RE = re.compile(r"^(?:PART|Part)\s+([IVXLCDM\d]+)(?:\s*[:.-]\s*(.*))?$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(?:ARTICLE|Article)\s+(\d+[A-Za-z]?)(?:\s*[:.-]\s*(.*))?$", re.IGNORECASE)

ROMAN_SUBCLAUSE_RE = re.compile(
    r"^\(((?:x{0,3}(?:ix|iv|v?i{1,3})|xl|l))\)\s+(.*)$",
    re.IGNORECASE,
)
CLAUSE_START_RE = re.compile(r"^\(([a-z]{1,2})\)\s+(.*)$", re.IGNORECASE)
SUBSECTION_START_RE = re.compile(r"^\((\d+)\)\s+(.*)$")


class LegalChunker(BaseChunker):
    """Structure-aware chunker for statutes, regulations, acts, and judgments."""

    def __init__(
        self,
        max_characters: int = 1500,
        min_characters: int = 20,
        overlap_characters: int = 150,
        settings: Optional[Settings] = None,
    ):
        """Initialize LegalChunker with configurable size boundaries.

        Args:
            max_characters: Maximum characters per chunk before sentence-splitting.
            min_characters: Minimum characters for a standalone paragraph chunk.
            overlap_characters: Overlap characters between sentence-split sub-chunks.
            settings: Project settings singleton.
        """
        self.max_characters = max_characters
        self.min_characters = min_characters
        self.overlap_characters = overlap_characters
        self.settings = settings or get_settings()

    def chunk_document(self, parsed_doc: DocumentParsingResult) -> List[EvidenceChunk]:
        """Convert a parsed legal document into validated EvidenceChunks.

        Maintains exact structural identifiers (e.g. Section 3(d) vs Section 3(e))
        and preserves the structural parent-child hierarchy.
        """
        if parsed_doc.domain != Domain.LEGAL:
            logger.warning(
                "Document %s has domain %s, but LegalChunker was invoked.",
                parsed_doc.document_id,
                parsed_doc.domain.value,
            )

        chunks: List[EvidenceChunk] = []

        doc_id = parsed_doc.document_id
        title = parsed_doc.title
        source = normalize_source_identifier(parsed_doc.source_path, domain=parsed_doc.domain.value)

        # Hierarchy state
        current_statute_name = parsed_doc.metadata.get("statute_name") or title
        current_case_name = parsed_doc.metadata.get("case_name")

        current_chapter: Optional[str] = None
        current_section_name: str = "General Provisions"
        current_section_id: str = "sec_000_general"
        current_section_chunk_id: Optional[str] = None

        current_clause_id: Optional[str] = None
        current_subsection_id: Optional[str] = None

        section_seq = 0
        chunk_seq = 0

        for elem in parsed_doc.elements:
            elem_text = elem.text.strip()
            if not elem_text:
                continue

            elem_page = elem.page if elem.page is not None and elem.page >= 1 else 1

            # 1. Check for Chapters / Parts
            chapter_match = CHAPTER_RE.match(elem_text)
            if chapter_match:
                current_chapter = f"Chapter {chapter_match.group(1)}"
                continue

            part_match = PART_RE.match(elem_text)
            if part_match:
                current_chapter = f"Part {part_match.group(1)}"
                continue

            # 2. Check for Headings / Section boundaries
            is_heading = elem.element_type in {"heading", "title"}
            sec_header_match = SECTION_HEADER_RE.match(elem_text)
            article_match = ARTICLE_RE.match(elem_text)

            if is_heading or sec_header_match or article_match:
                section_seq += 1

                # Extract exact section or article reference
                exact_id = self._extract_exact_section_identifier(elem_text)
                if exact_id:
                    current_section_name = f"Section {exact_id}"
                    current_section_id = f"sec_{exact_id.lower().replace('(', '_').replace(')', '')}"
                elif article_match:
                    art_num = article_match.group(1)
                    current_section_name = f"Article {art_num}"
                    current_section_id = f"art_{art_num.lower()}"
                else:
                    current_section_name = elem_text
                    current_section_id = f"sec_{section_seq:03d}_{clean_section_slug(elem_text)}"

                current_section_chunk_id = f"{doc_id}_{current_section_id}"
                current_clause_id = exact_id if (exact_id and "(" in exact_id) else None
                current_subsection_id = None

                # Create the Parent Section EvidenceChunk
                parent_chunk = EvidenceChunk(
                    chunk_id=current_section_chunk_id,
                    document_id=doc_id,
                    domain=Domain.LEGAL,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=None,
                    content_type=ContentType.SECTION if not is_heading else ContentType.HEADING,
                    text=elem_text,
                    statute_name=current_statute_name,
                    case_name=current_case_name,
                    clause_id=current_clause_id,
                    subsection_id=current_subsection_id,
                )
                chunks.append(parent_chunk)
                continue

            # Ensure we have an active parent section; if none, create a default root
            if current_section_chunk_id is None:
                section_seq += 1
                current_section_name = "General"
                current_section_id = f"sec_{section_seq:03d}_general"
                current_section_chunk_id = f"{doc_id}_{current_section_id}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=current_section_chunk_id,
                        document_id=doc_id,
                        domain=Domain.LEGAL,
                        title=title,
                        source=source,
                        page=elem_page,
                        section=current_section_name,
                        section_id=current_section_id,
                        parent_id=None,
                        content_type=ContentType.SECTION,
                        text=current_section_name,
                        statute_name=current_statute_name,
                        case_name=current_case_name,
                    )
                )

            # 3. Handle Tables
            if elem.element_type == "table":
                chunk_seq += 1
                tbl_id = elem.metadata.get("table_id") or f"{doc_id}_table_{chunk_seq:03d}"
                chunk_id = f"{doc_id}_{current_section_id}_table_{chunk_seq:03d}"
                table_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.LEGAL,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=current_section_chunk_id,
                    content_type=ContentType.TABLE,
                    table_id=tbl_id,
                    text=elem_text,
                    statute_name=current_statute_name,
                    case_name=current_case_name,
                    clause_id=current_clause_id,
                )
                chunks.append(table_chunk)
                continue

            # 4. Handle Figures
            if elem.element_type in {"figure", "caption"}:
                chunk_seq += 1
                fig_id = elem.metadata.get("figure_id") or f"Fig_{chunk_seq:03d}"
                chunk_id = f"{doc_id}_{current_section_id}_fig_{chunk_seq:03d}"
                fig_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.LEGAL,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=current_section_name,
                    section_id=current_section_id,
                    parent_id=current_section_chunk_id,
                    content_type=ContentType.FIGURE if elem.element_type == "figure" else ContentType.CAPTION,
                    figure_id=fig_id,
                    text=elem_text,
                    statute_name=current_statute_name,
                    case_name=current_case_name,
                    clause_id=current_clause_id,
                )
                chunks.append(fig_chunk)
                continue

            # 5. Skip micro elements
            if len(elem_text) < self.min_characters:
                continue

            # 6. Analyze paragraph for exact clause / subsection patterns
            (
                content_type,
                detected_clause_id,
                detected_subsection_id,
                effective_sec_id,
                effective_sec_name,
                is_subclause,
            ) = self._classify_legal_paragraph(
                elem_text,
                current_section_id,
                current_section_name,
                current_clause_id,
                current_subsection_id,
            )

            # Update hierarchy state tracking
            if content_type == ContentType.SUBSECTION:
                current_subsection_id = detected_subsection_id
                current_clause_id = None
            elif content_type == ContentType.CLAUSE:
                if not is_subclause:
                    current_clause_id = detected_clause_id

            # Handle oversized legal provisions with sentence packing
            if len(elem_text) > self.max_characters:
                sentences = split_into_sentences(elem_text)
                packed_chunks = pack_sentences(
                    sentences,
                    max_chars=self.max_characters,
                    overlap_chars=self.overlap_characters,
                )
                for sub_text in packed_chunks:
                    chunk_seq += 1
                    chunk_id = f"{doc_id}_{effective_sec_id}_chunk_{chunk_seq:03d}"
                    child_chunk = EvidenceChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        domain=Domain.LEGAL,
                        title=title,
                        source=source,
                        page=elem_page,
                        section=effective_sec_name,
                        section_id=effective_sec_id,
                        parent_id=current_section_chunk_id,
                        content_type=content_type,
                        text=sub_text,
                        statute_name=current_statute_name,
                        case_name=current_case_name,
                        clause_id=detected_clause_id,
                        subsection_id=detected_subsection_id,
                    )
                    chunks.append(child_chunk)
            else:
                chunk_seq += 1
                chunk_id = f"{doc_id}_{effective_sec_id}_chunk_{chunk_seq:03d}"
                child_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    domain=Domain.LEGAL,
                    title=title,
                    source=source,
                    page=elem_page,
                    section=effective_sec_name,
                    section_id=effective_sec_id,
                    parent_id=current_section_chunk_id,
                    content_type=content_type,
                    text=elem_text,
                    statute_name=current_statute_name,
                    case_name=current_case_name,
                    clause_id=detected_clause_id,
                    subsection_id=detected_subsection_id,
                )
                chunks.append(child_chunk)

        # Validate complete chunk collection
        self.validate_chunks(chunks)

        logger.info(
            "Legal chunking complete: generated %d chunks from %s",
            len(chunks),
            doc_id,
        )
        return chunks

    def _extract_exact_section_identifier(self, text: str) -> Optional[str]:
        """Extract exact section identifier like '3', '3(d)', '3(1)(a)' from text."""
        match = EXACT_SECTION_NUM_RE.search(text)
        if match:
            return match.group(1).strip()

        # Check for leading "3(d)" or "3."
        numbered_match = re.match(r"^(\d+[A-Za-z]?(?:\([0-9a-z]+\))*)\.?", text)
        if numbered_match:
            candidate = numbered_match.group(1)
            # Ensure it's not a generic year or huge number
            if len(candidate) <= 10 and not candidate.startswith("19") and not candidate.startswith("20"):
                return candidate
        return None

    def _classify_legal_paragraph(
        self,
        text: str,
        current_sec_id: str,
        current_sec_name: str,
        current_clause: Optional[str],
        current_sub: Optional[str],
    ) -> Tuple[ContentType, Optional[str], Optional[str], str, str, bool]:
        """Classify legal paragraph and extract exact clause/subsection identifiers.

        Returns:
            Tuple of:
            (content_type, clause_id, subsection_id, effective_sec_id, effective_sec_name, is_subclause)
        """
        sec_num_match = re.search(r"\d+[A-Za-z]?", current_sec_name)
        sec_num = sec_num_match.group(0) if sec_num_match else "sec"

        # 1. Check for Roman subclause like (i), (ii), (iv), (ix)
        roman_match = ROMAN_SUBCLAUSE_RE.match(text)
        if roman_match:
            roman = roman_match.group(1).lower()
            if current_clause:
                clause_id = f"{current_clause}({roman})"
                slug_clause = current_clause.lower().replace("(", "_").replace(")", "")
                eff_sec_id = f"sec_{slug_clause}_{roman}"
                eff_sec_name = f"Section {clause_id}"
            elif current_sub:
                clause_id = f"{current_sub}({roman})"
                slug_sub = current_sub.lower().replace("(", "_").replace(")", "")
                eff_sec_id = f"sec_{slug_sub}_{roman}"
                eff_sec_name = f"Section {clause_id}"
            else:
                clause_id = f"({roman})"
                eff_sec_id = f"{current_sec_id}_{roman}"
                eff_sec_name = f"{current_sec_name}({roman})"
            return ContentType.CLAUSE, clause_id, current_sub, eff_sec_id, eff_sec_name, True

        # 2. Check for subsection like (1), (2)
        sub_match = SUBSECTION_START_RE.match(text)
        if sub_match:
            num = sub_match.group(1)
            sub_id = f"{sec_num}({num})"
            eff_sec_id = f"{current_sec_id}_{num}"
            eff_sec_name = f"Section {sub_id}"
            return ContentType.SUBSECTION, current_clause, sub_id, eff_sec_id, eff_sec_name, False

        # 3. Check for clause like (d), (e), (a)
        clause_match = CLAUSE_START_RE.match(text)
        if clause_match:
            letter = clause_match.group(1).lower()
            if current_sub:
                clause_id = f"{current_sub}({letter})"
                eff_sec_name = f"Section {clause_id}"
            else:
                clause_id = f"{sec_num}({letter})"
                eff_sec_name = f"Section {clause_id}"
            eff_sec_id = f"{current_sec_id}_{letter}"
            return ContentType.CLAUSE, clause_id, current_sub, eff_sec_id, eff_sec_name, False

        # 4. Check for references / case citation
        if "v." in text or "vs." in text or "AIR " in text or "SCC " in text or "SCR " in text:
            return ContentType.REFERENCE, current_clause, current_sub, current_sec_id, current_sec_name, False

        # Default standard paragraph within the section
        return ContentType.PARAGRAPH, current_clause, current_sub, current_sec_id, current_sec_name, False
