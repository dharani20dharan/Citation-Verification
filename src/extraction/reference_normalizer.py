"""Reference normalizer preserving exact legal structural identifiers and scientific citations."""

import logging
import re
from typing import List, Tuple

from src.extraction.models import ReferenceType

logger = logging.getLogger(__name__)


class ReferenceNormalizer:
    """Normalizes citations and statutory references without loss of structural identity."""

    @staticmethod
    def normalize_numeric_citation(raw_citation: str) -> Tuple[str, List[str]]:
        """Normalize numeric citation brackets like [12], [3, 4], [5–7].

        Args:
            raw_citation: Raw matched text including brackets (e.g. '[3, 4]').

        Returns:
            Tuple of (canonical_normalized_str, list_of_individual_references).
        """
        # Strip outer brackets and surrounding whitespace
        content = raw_citation.strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()

        # Normalize unicode dashes (en-dash \u2013, em-dash \u2014) to standard ASCII hyphen
        content = re.sub(r"[\u2013\u2014\u2212]", "-", content)

        # Split multiple references by comma
        parts = [p.strip() for p in content.split(",") if p.strip()]

        normalized_list: List[str] = []
        for part in parts:
            # Normalize internal spaces around hyphen in ranges: '5 - 7' -> '5-7'
            part_norm = re.sub(r"\s*-\s*", "-", part)
            if part_norm:
                normalized_list.append(part_norm)

        normalized_str = ", ".join(normalized_list) if normalized_list else content
        return normalized_str, normalized_list

    @staticmethod
    def normalize_author_year_citation(raw_citation: str) -> Tuple[str, List[str]]:
        """Normalize author-year citation parentheses like (Smith et al., 2024).

        Args:
            raw_citation: Raw matched text including parentheses.

        Returns:
            Tuple of (canonical_normalized_str, list_of_references).
        """
        content = raw_citation.strip()
        if content.startswith("(") and content.endswith(")"):
            content = content[1:-1].strip()

        # Normalize multiple spaces and standard formatting
        content = re.sub(r"\s+", " ", content)
        return content, [content]

    @staticmethod
    def normalize_legal_reference(raw_ref: str, ref_type: str) -> Tuple[str, List[str]]:
        """Normalize legal section, article, clause, or rule references.

        CRITICAL LEGAL REQUIREMENT:
        - Must preserve exact subclauses, parentheses, roman numerals, and letters.
        - Section 3(d) must NOT become 3.
        - Section 3(d)(i) must NOT become 3(d).
        - 3(d) != 3(e) and 3(d)(i) != 3(d)(ii).

        Args:
            raw_ref: Raw reference text (e.g. 'Section 3(d)(i)').
            ref_type: ReferenceType value (e.g. legal_section, legal_article).

        Returns:
            Tuple of (normalized_id, [normalized_id]).
        """
        cleaned = raw_ref.strip()

        # Match and strip prefixes only, keeping exact identifier
        if ref_type == ReferenceType.LEGAL_SECTION.value:
            # Matches 'Section 3(d)', 'Sec. 3(d)', '§§ 3(d)', '§ 3(d)'
            cleaned = re.sub(r"^(?:Section|Sec\.|§§?)\s*", "", cleaned, flags=re.IGNORECASE).strip()
        elif ref_type == ReferenceType.LEGAL_ARTICLE.value:
            # Matches 'Article 14(2)', 'Art. 14'
            cleaned = re.sub(r"^(?:Article|Art\.)\s*", "", cleaned, flags=re.IGNORECASE).strip()
        elif ref_type == ReferenceType.LEGAL_CLAUSE.value:
            # Matches 'Clause 7', 'Clause 3(a)'
            cleaned = re.sub(r"^Clause\s*", "", cleaned, flags=re.IGNORECASE).strip()
        elif ref_type == ReferenceType.LEGAL_RULE.value:
            # Matches 'Rule 12', 'Rule 12(1)'
            cleaned = re.sub(r"^Rule\s*", "", cleaned, flags=re.IGNORECASE).strip()

        # Ensure no leading/trailing punctuation left over except closing/opening parentheses
        cleaned = cleaned.strip()

        return cleaned, [cleaned]

    def normalize(self, raw_ref: str, ref_type: str) -> Tuple[str, List[str]]:
        """Dispatch normalization according to reference type.

        Args:
            raw_ref: Raw reference string.
            ref_type: ReferenceType classification string.

        Returns:
            Tuple of (normalized_reference, list_of_normalized_references).
        """
        if ref_type == ReferenceType.NUMERIC.value:
            return self.normalize_numeric_citation(raw_ref)
        elif ref_type == ReferenceType.AUTHOR_YEAR.value:
            return self.normalize_author_year_citation(raw_ref)
        elif ref_type in {
            ReferenceType.LEGAL_SECTION.value,
            ReferenceType.LEGAL_ARTICLE.value,
            ReferenceType.LEGAL_CLAUSE.value,
            ReferenceType.LEGAL_RULE.value,
        }:
            return self.normalize_legal_reference(raw_ref, ref_type)
        else:
            cleaned = raw_ref.strip()
            return cleaned, [cleaned]
