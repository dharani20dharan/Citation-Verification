"""Bibliographic entry parser converting raw reference strings into structured BibliographicEntry objects."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.schemas import Domain
from src.references.models import BibliographicEntry, generate_entry_id

logger = logging.getLogger(__name__)

# Label extraction patterns
# Bracketed numeric: [1], [12], [3]
BRACKET_NUMERIC_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$", re.DOTALL)

# Dotted numeric: 1. Smith, 12. Brown
DOTTED_NUMERIC_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.DOTALL)

# Year patterns: (2024), 2024.
YEAR_RE = re.compile(r"\b([12]\d{3})[a-z]?\b")

# DOI pattern: 10.xxxx/...
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")

# URL pattern: http:// or https://
URL_RE = re.compile(r"\b(https?://[^\s]+)\b")

# Author pattern for author-year style: "Smith, J., Doe, A. 2024..." or "Smith, J. et al. (2024)"
AUTHOR_YEAR_ENTRY_RE = re.compile(
    r"^\s*([A-Z][a-zA-Z\s\-]+?(?:et\s+al\.?|and\s+[A-Za-z\s\-]+|&\s+[A-Za-z\s\-]+|,\s*[A-Z]\.?\s*[A-Z]?\.?)?)[,\s]+\(?([12]\d{3}[a-z]?)\)?[\.:\s]+(.*)$",
    re.DOTALL,
)


class BibliographicParser:
    """Parses raw text reference blocks into validated BibliographicEntry objects."""

    def parse_entry(
        self,
        raw_text: str,
        document_id: str,
        domain: Domain = Domain.SCIENTIFIC,
        page: Optional[int] = None,
        section: Optional[str] = None,
        chunk_id: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> BibliographicEntry:
        """Parse a single raw reference string into a structured BibliographicEntry.

        Args:
            raw_text: Raw reference text.
            document_id: Source document ID.
            domain: Corpus domain.
            page: Optional page number.
            section: Optional section title.
            chunk_id: Optional chunk ID.
            provenance: Optional additional provenance dictionary.

        Returns:
            Structured BibliographicEntry with deterministic ID and preserved raw_text.
        """
        clean_raw = raw_text.strip()
        label, normalized_label, body = self._extract_label(clean_raw)

        # Baseline metadata extraction
        year = self._extract_year(clean_raw)
        doi = self._extract_doi(clean_raw)
        url = self._extract_url(clean_raw)
        authors = self._extract_authors(body if body else clean_raw, year)
        title, venue = self._extract_title_and_venue(body if body else clean_raw, authors, year)

        entry_id = generate_entry_id(document_id, normalized_label)

        merged_provenance: Dict[str, Any] = {
            "document_id": document_id,
            "page": page,
            "section": section,
            "chunk_id": chunk_id,
            **(provenance or {}),
        }

        return BibliographicEntry(
            entry_id=entry_id,
            document_id=document_id,
            domain=domain,
            reference_label=label,
            normalized_label=normalized_label,
            raw_text=clean_raw,
            title=title,
            authors=authors,
            year=year,
            journal=venue,
            doi=doi,
            url=url,
            page=page,
            section=section,
            provenance=merged_provenance,
        )

    def _extract_label(self, raw_text: str) -> Tuple[str, str, str]:
        """Extract reference label, normalized label, and remaining body text.

        Returns:
            Tuple of (reference_label, normalized_label, remaining_body)
        """
        # Form 1: Bracketed numeric e.g. "[12] Smith..."
        m_bracket = BRACKET_NUMERIC_RE.match(raw_text)
        if m_bracket:
            num = m_bracket.group(1)
            return f"[{num}]", num, m_bracket.group(2).strip()

        # Form 2: Dotted numeric e.g. "12. Smith..."
        m_dotted = DOTTED_NUMERIC_RE.match(raw_text)
        if m_dotted:
            num = m_dotted.group(1)
            return f"{num}.", num, m_dotted.group(2).strip()

        # Form 3: Author-Year style e.g. "Smith, J., Johnson, K., and Williams, R. (2024). Precision..."
        m_paren_year = re.search(r"\(([12]\d{3}[a-z]?)\)", raw_text)
        if m_paren_year:
            year_str = m_paren_year.group(1)
            prefix = raw_text[:m_paren_year.start()].strip().rstrip(",")
            surname_match = re.match(r"^\s*([A-Z][a-zA-Z\-]+)", prefix)
            if surname_match and len(prefix.split()) <= 15:
                surname = surname_match.group(1)
                norm_label = f"{surname}_{year_str}"
                body = raw_text[m_paren_year.end():].strip().lstrip(".").strip()
                return f"{surname} et al., {year_str}", norm_label, body

        # Fallback: Hash-based or first 3 words
        words = raw_text.split()
        first_few = "_".join(words[:2]) if len(words) >= 2 else "ref"
        clean_first = re.sub(r"[^\w]", "", first_few) or "ref"
        return clean_first, clean_first, raw_text

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year from text, ignoring DOIs and URLs."""
        # Strip DOIs and URLs first so numbers in IDs (e.g. arXiv 2005.14165) are not mistaken for years
        clean_text = DOI_RE.sub(" ", text)
        clean_text = URL_RE.sub(" ", clean_text)

        # Check parenthesized years first (e.g. (2024))
        m_paren = re.search(r"\(([12]\d{3})[a-z]?\)", clean_text)
        if m_paren:
            return int(m_paren.group(1))

        # Check for standalone 4-digit year
        matches = YEAR_RE.findall(clean_text)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                return None
        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI string from text."""
        m = DOI_RE.search(text)
        return m.group(1).rstrip(".") if m else None

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract HTTP/HTTPS URL from text."""
        m = URL_RE.search(text)
        return m.group(1).rstrip(".,;") if m else None

    def _extract_authors(self, text: str, year: Optional[int]) -> List[str]:
        """Heuristic author list extraction from the prefix of reference text."""
        if not text:
            return []

        # If year exists, look at text before year
        prefix = text
        if year is not None:
            year_idx = text.find(str(year))
            if year_idx > 5:
                prefix = text[:year_idx]

        # Strip trailing parentheses or dots
        prefix = re.sub(r"[\(\.\,]+$", "", prefix).strip()

        # Split authors on 'and', '&', or ';'
        chunks = re.split(r"\s+(?:and|&)\s+|;\s*", prefix)
        authors: List[str] = []
        for c in chunks:
            cleaned = c.strip()
            # If comma separated multiple authors: e.g. "Vaswani, A., Shazeer, N."
            if "," in cleaned and not any(title_word in cleaned.lower() for title_word in ["title", "proceedings", "conference"]):
                sub_parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                if len(sub_parts) == 2 and len(sub_parts[1].split()) <= 2:
                    # Surname, Initial
                    authors.append(f"{sub_parts[0]}, {sub_parts[1]}")
                else:
                    authors.append(cleaned)
            elif cleaned:
                authors.append(cleaned)

        return [a for a in authors if len(a) > 2][:8]

    def _extract_title_and_venue(
        self,
        text: str,
        authors: List[str],
        year: Optional[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract title and venue/journal heuristics."""
        if not text:
            return None, None

        # Look for quoted title: "Title of Paper"
        m_quote = re.search(r'["\u201c\u201d\']([^"\u201c\u201d\']{8,150})["\u201c\u201d\']', text)
        if m_quote:
            title = m_quote.group(1).strip()
            venue = text[m_quote.end() :].strip().strip("., ")
            return title, (venue if len(venue) > 3 else None)

        # Sentence heuristic: split by period
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) >= 2:
            # First or second sentence after authors
            candidate_title = sentences[1] if len(sentences) >= 3 else sentences[0]
            venue = sentences[-1] if len(sentences) >= 3 else None
            if len(candidate_title) >= 10:
                return candidate_title, venue

        return None, None
