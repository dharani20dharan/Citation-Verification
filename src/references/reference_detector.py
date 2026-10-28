"""Reference section detector identifying bibliographic sections in parsed scientific documents."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from src.models.schemas import Domain, EvidenceChunk

logger = logging.getLogger(__name__)

# Common reference section heading patterns (case-insensitive)
REFERENCE_HEADING_RE = re.compile(
    r"^(?:\d+[\.\s]+)?(?:references|bibliography|literature\s+cited|works\s+cited|reference\s+list)\b",
    re.IGNORECASE,
)

# Structural heading patterns that terminate a reference section
TERMINATING_HEADING_RE = re.compile(
    r"^(?:\d+[\.\s]+)?(?:appendix|appendices|supplementary|author\s+contributions|acknowledgements?|conflict\s+of\s+interest|funding)\b",
    re.IGNORECASE,
)


class ReferenceSectionDetector:
    """Detects and extracts raw bibliographic entries from reference sections.

    Supports:
        - ParsedDocument / dict with elements
        - EvidenceChunk collections
        - Raw text blocks
    """

    def is_reference_heading(self, text: str) -> bool:
        """Check if heading text indicates a reference/bibliography section."""
        cleaned = text.strip()
        return bool(REFERENCE_HEADING_RE.search(cleaned))

    def is_terminating_heading(self, text: str) -> bool:
        """Check if heading text indicates the end of a reference section."""
        cleaned = text.strip()
        return bool(TERMINATING_HEADING_RE.search(cleaned))

    def detect_from_elements(
        self,
        elements: List[Union[Any, Dict[str, Any]]],
        document_id: str,
        domain: Domain = Domain.SCIENTIFIC,
    ) -> List[Dict[str, Any]]:
        """Detect and collect raw reference entries from a sequence of parsed document elements.

        Args:
            elements: List of ParsedElement or parsed element dicts from Docling.
            document_id: Identifier of the source document.
            domain: Corpus domain.

        Returns:
            List of raw candidate reference blocks with provenance.
        """
        inside_ref_section = False
        current_section_title = "References"
        raw_entries: List[Dict[str, Any]] = []

        for elem in elements:
            if isinstance(elem, dict):
                elem_type = elem.get("element_type", "paragraph")
                elem_text = elem.get("text", "").strip()
                elem_page = elem.get("page")
                elem_id = elem.get("element_id")
            else:
                elem_type = elem.element_type.value if hasattr(elem.element_type, "value") else str(elem.element_type)
                elem_text = elem.text.strip()
                elem_page = elem.page
                elem_id = elem.element_id

            if not elem_text:
                continue

            # Check if this element is a heading
            is_heading = elem_type.lower() in ("heading", "title", "section", "section_header")

            if is_heading:
                if self.is_reference_heading(elem_text):
                    inside_ref_section = True
                    current_section_title = elem_text
                    logger.debug("Entering reference section '%s' in %s", elem_text, document_id)
                    continue
                elif inside_ref_section and self.is_terminating_heading(elem_text):
                    logger.debug("Terminating reference section at '%s' in %s", elem_text, document_id)
                    break
                elif inside_ref_section:
                    # Check if heading clearly indicates another distinct section
                    logger.debug("Ending reference section at unexpected heading '%s' in %s", elem_text, document_id)
                    break

            if inside_ref_section:
                # Discard empty or trivial metadata entries
                if len(elem_text) < 5:
                    continue
                raw_entries.append({
                    "raw_text": elem_text,
                    "document_id": document_id,
                    "domain": domain,
                    "page": elem_page,
                    "section": current_section_title,
                    "element_id": elem_id,
                    "chunk_id": None,
                })

        logger.info(
            "Detected %d raw reference items in document '%s'",
            len(raw_entries),
            document_id,
        )
        return raw_entries

    def detect_from_chunks(
        self,
        chunks: List[EvidenceChunk],
        document_id: str,
        domain: Domain = Domain.SCIENTIFIC,
    ) -> List[Dict[str, Any]]:
        """Detect and collect raw reference entries from a collection of EvidenceChunks.

        Args:
            chunks: List of EvidenceChunk objects.
            document_id: Identifier of the source document.
            domain: Corpus domain.

        Returns:
            List of raw candidate reference blocks with chunk provenance.
        """
        inside_ref_section = False
        current_section_title = "References"
        raw_entries: List[Dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue

            c_type = chunk.content_type.value if hasattr(chunk.content_type, "value") else str(chunk.content_type)
            is_heading = c_type.lower() in ("heading", "title", "section")

            # Check section metadata or chunk text for reference heading
            chunk_section = chunk.section or ""
            if self.is_reference_heading(chunk_section) or (is_heading and self.is_reference_heading(text)):
                inside_ref_section = True
                if is_heading:
                    current_section_title = text
                    continue

            if is_heading and inside_ref_section and not self.is_reference_heading(text):
                if self.is_terminating_heading(text):
                    break

            if inside_ref_section or self.is_reference_heading(chunk_section):
                # If chunk contains multiple references on separate lines, split them
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                for line in lines:
                    if self.is_reference_heading(line):
                        continue
                    if len(line) < 5:
                        continue
                    raw_entries.append({
                        "raw_text": line,
                        "document_id": document_id,
                        "domain": domain,
                        "page": chunk.page,
                        "section": chunk_section or current_section_title,
                        "chunk_id": chunk.chunk_id,
                        "element_id": None,
                    })

        return raw_entries
