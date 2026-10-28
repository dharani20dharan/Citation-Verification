"""Docling-based document parser for legal and scientific documents.

Preserves structural hierarchy, headings, page traceability, tables,
figures, and element ordering into a canonical DocumentParsingResult.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)

from src.config.settings import Settings, get_settings
from src.models.schemas import (
    Domain,
    DocumentParsingResult,
    ParsedElement,
    ParsedTable,
)

logger = logging.getLogger(__name__)


class DocumentParsingError(Exception):
    """Base exception for document parsing failures."""
    pass


class DocumentNotFoundError(DocumentParsingError):
    """Raised when the specified document file does not exist."""
    pass


class UnsupportedFormatError(DocumentParsingError):
    """Raised when the document extension is not supported."""
    pass


class DoclingDocumentParser:
    """Parser using Docling to extract structured documents for downstream chunking."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._converter: Optional[DocumentConverter] = None

    @property
    def converter(self) -> DocumentConverter:
        """Lazy initialization of DocumentConverter."""
        if self._converter is None:
            logger.info("Initializing Docling DocumentConverter...")
            self._converter = DocumentConverter()
        return self._converter

    @staticmethod
    def generate_document_id(domain: Domain, file_path: Path) -> str:
        """Generate a deterministic, stable document identifier.

        Format: <domain>_<sanitized_stem>
        """
        clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", file_path.stem).lower().strip("_")
        if not clean_stem:
            clean_stem = "doc"
        return f"{domain.value}_{clean_stem}"

    def parse(
        self,
        file_path: Union[str, Path],
        domain: Domain,
        custom_doc_id: Optional[str] = None,
    ) -> DocumentParsingResult:
        """Parse a PDF or DOCX file into a structured DocumentParsingResult.

        Args:
            file_path: Path to the input document.
            domain: Domain enum (LEGAL or SCIENTIFIC).
            custom_doc_id: Optional custom deterministic document ID.

        Returns:
            DocumentParsingResult containing ordered elements, tables, and metadata.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the extension is not .pdf or .docx.
            DocumentParsingError: If Docling fails during parsing.
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise DocumentNotFoundError(f"Document file not found at: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        doc_id = custom_doc_id or self.generate_document_id(domain, path)
        logger.info("Starting Docling parsing for document_id=%s, path=%s", doc_id, path)

        try:
            conversion_res = self.converter.convert(path)
            docling_doc: DoclingDocument = conversion_res.document
        except Exception as e:
            self._converter = None
            logger.error("Docling failed to convert %s: %s", path, str(e), exc_info=True)
            raise DocumentParsingError(f"Failed to parse document '{path.name}': {str(e)}") from e

        return self._extract_structured_result(docling_doc, path, domain, doc_id)

    def _extract_structured_result(
        self,
        doc: DoclingDocument,
        path: Path,
        domain: Domain,
        doc_id: str,
    ) -> DocumentParsingResult:
        """Extract structured elements, tables, pages, and metadata from DoclingDocument."""
        # 1. Determine title
        title = self._extract_title(doc, path)

        # 2. Number of pages
        num_pages_fn = getattr(doc, "num_pages", None)
        if callable(num_pages_fn):
            try:
                pages = int(num_pages_fn())
            except Exception:
                pages = 1
        elif hasattr(doc, "pages"):
            pages = len(doc.pages)
        else:
            pages = 1
        pages = max(1, pages)

        elements: List[ParsedElement] = []
        tables: List[ParsedTable] = []

        # Hierarchy tracking: stack of (level, element_id, heading_text)
        heading_stack: List[Tuple[int, str, str]] = []

        element_seq = 0
        table_seq = 0

        for item, level in doc.iterate_items():
            element_seq += 1
            elem_id = f"{doc_id}_elem_{element_seq:04d}"

            # Extract page number if available from provenance
            page_no = self._extract_page_number(item)

            label_str = getattr(item, "label", None)
            if hasattr(label_str, "value"):
                label_str = label_str.value
            else:
                label_str = str(label_str) if label_str else "text"

            # Handle Headings / Titles
            if isinstance(item, (SectionHeaderItem, TitleItem)) or label_str in {"section_header", "title"}:
                heading_text = getattr(item, "text", "").strip()
                h_level = level if level and level > 0 else 1

                # Pop items in stack with level >= current heading level
                while heading_stack and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                parent_id = heading_stack[-1][1] if heading_stack else None
                heading_stack.append((h_level, elem_id, heading_text))

                elements.append(
                    ParsedElement(
                        element_id=elem_id,
                        element_type="title" if isinstance(item, TitleItem) or label_str == "title" else "heading",
                        text=heading_text,
                        page=page_no,
                        level=h_level,
                        parent_element_id=parent_id,
                        metadata={"raw_label": label_str},
                    )
                )
                continue

            parent_id = heading_stack[-1][1] if heading_stack else None
            enclosing_section = heading_stack[-1][2] if heading_stack else ""

            # Handle Tables
            if isinstance(item, TableItem) or label_str == "table":
                table_seq += 1
                tbl_id = f"{doc_id}_table_{table_seq:03d}"

                try:
                    markdown = item.export_to_markdown(doc=doc)
                except Exception:
                    markdown = getattr(item, "text", "")

                num_rows = 0
                num_cols = 0
                if hasattr(item, "data") and item.data is not None:
                    num_rows = getattr(item.data, "num_rows", 0)
                    num_cols = getattr(item.data, "num_cols", 0)

                caption = None
                if hasattr(item, "caption_text"):
                    try:
                        caption = item.caption_text(doc=doc)
                    except Exception:
                        pass

                parsed_table = ParsedTable(
                    table_id=tbl_id,
                    page=page_no,
                    markdown=markdown,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    caption=caption,
                    metadata={"enclosing_section": enclosing_section},
                )
                tables.append(parsed_table)

                # Also add as a parsed element to preserve document stream ordering
                elements.append(
                    ParsedElement(
                        element_id=elem_id,
                        element_type="table",
                        text=markdown,
                        page=page_no,
                        level=None,
                        parent_element_id=parent_id,
                        metadata={
                            "table_id": tbl_id,
                            "num_rows": num_rows,
                            "num_cols": num_cols,
                            "caption": caption or "",
                            "enclosing_section": enclosing_section,
                        },
                    )
                )
                continue

            # Handle Figures / Pictures
            if isinstance(item, PictureItem) or label_str in {"picture", "figure"}:
                caption = None
                if hasattr(item, "caption_text"):
                    try:
                        caption = item.caption_text(doc=doc)
                    except Exception:
                        pass
                fig_text = caption or f"[Figure {element_seq}]"

                elements.append(
                    ParsedElement(
                        element_id=elem_id,
                        element_type="figure",
                        text=fig_text,
                        page=page_no,
                        level=None,
                        parent_element_id=parent_id,
                        metadata={
                            "caption": caption or "",
                            "enclosing_section": enclosing_section,
                        },
                    )
                )
                continue

            # Handle General Text / Paragraphs / Clauses / Lists
            item_text = getattr(item, "text", "").strip()
            if not item_text:
                continue

            elements.append(
                ParsedElement(
                    element_id=elem_id,
                    element_type="paragraph" if label_str in {"paragraph", "text"} else label_str,
                    text=item_text,
                    page=page_no,
                    level=None,
                    parent_element_id=parent_id,
                    metadata={
                        "raw_label": label_str,
                        "enclosing_section": enclosing_section,
                    },
                )
            )

        # Ensure page count accounts for max detected page
        detected_pages = [e.page for e in elements if e.page is not None]
        if detected_pages:
            pages = max(pages, max(detected_pages))

        # Retain full raw dictionary export
        try:
            raw_structure = doc.export_to_dict()
        except Exception:
            raw_structure = {}

        return DocumentParsingResult(
            document_id=doc_id,
            domain=domain,
            source_path=str(path),
            title=title,
            pages=pages,
            elements=elements,
            tables=tables,
            raw_structure=raw_structure,
            metadata={
                "filename": path.name,
                "file_extension": path.suffix.lower(),
                "element_count": len(elements),
                "table_count": len(tables),
            },
        )

    def _extract_title(self, doc: DoclingDocument, path: Path) -> str:
        """Extract title from doc items or fallback to clean filename stem."""
        for item, _ in doc.iterate_items():
            if isinstance(item, TitleItem) or getattr(item, "label", None) == DocItemLabel.TITLE:
                t = getattr(item, "text", "").strip()
                if t:
                    return t
        # Fallback to first section header if available
        for item, _ in doc.iterate_items():
            if isinstance(item, SectionHeaderItem) or getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                t = getattr(item, "text", "").strip()
                if t:
                    return t
        return path.stem.replace("_", " ").replace("-", " ").title()

    def _extract_page_number(self, item: Any) -> Optional[int]:
        """Extract 1-based page number from DocItem provenance if available."""
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            first_prov = prov[0]
            if hasattr(first_prov, "page_no") and first_prov.page_no is not None:
                return int(first_prov.page_no)
        return None

    def save_parsed_result(
        self,
        result: DocumentParsingResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save DocumentParsingResult to a JSON file under data/parsed/<domain>/."""
        if output_dir is None:
            if result.domain == Domain.LEGAL:
                output_dir = self.settings.parsed_legal_dir
            else:
                output_dir = self.settings.parsed_scientific_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"{result.document_id}.json"

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        logger.info("Saved parsed document result to %s", target_file)
        return target_file

    @staticmethod
    def load_parsed_result(file_path: Union[str, Path]) -> DocumentParsingResult:
        """Load DocumentParsingResult from a JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentParsingResult.model_validate(data)
