"""Citation and reference detector identifying citations and statutory markers in text."""

import logging
import re
from typing import List, Optional

from src.models.schemas import Domain
from src.extraction.models import DetectedReference, ReferenceType
from src.extraction.reference_normalizer import ReferenceNormalizer

logger = logging.getLogger(__name__)


class CitationDetector:
    """Detects scientific citations and legal structural references using robust regex rules."""

    # Scientific citation patterns
    # Matches [1], [12], [3, 4], [5-7], [5–7], [12, 14]
    PATTERN_NUMERIC = re.compile(
        r"\[\s*(\d+(?:\s*[\u2013\u2014\-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[\u2013\u2014\-]\s*\d+)?)*)\s*\]"
    )

    # Matches (Smith et al., 2024), (Smith, 2023), (Smith & Jones, 2021)
    PATTERN_AUTHOR_YEAR = re.compile(
        r"\(([A-Z][a-zA-Z\s\-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-zA-Z\s\-]+|&\s+[A-Z][a-zA-Z\s\-]+))?),\s*([12]\d{3}[a-z]?)\)"
    )

    # Legal reference patterns
    # Matches Section 3(d), Section 3(e), Section 3(d)(i), Section 3(d)(ii), § 3(d), Section 3
    PATTERN_LEGAL_SECTION = re.compile(
        r"(?:\bSection\b|\bSec\.\b|§§?)\s*(\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
    )

    # Matches Article 14, Article 14(2), Art. 14
    PATTERN_LEGAL_ARTICLE = re.compile(
        r"(?:\bArticle\b|\bArt\.\b)\s*(\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
    )

    # Matches Clause 7, Clause 3(a)
    PATTERN_LEGAL_CLAUSE = re.compile(
        r"\bClause\s*(\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
    )

    # Matches Rule 12, Rule 12(1)
    PATTERN_LEGAL_RULE = re.compile(
        r"\bRule\s*(\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)"
    )

    def __init__(self, normalizer: Optional[ReferenceNormalizer] = None):
        self.normalizer = normalizer or ReferenceNormalizer()

    def detect_references(
        self,
        text: str,
        domain: Optional[Domain] = None,
    ) -> List[DetectedReference]:
        """Detect all citations and legal references in text, sorted by position.

        Args:
            text: Input text string.
            domain: Optional Domain filter (Domain.LEGAL, Domain.SCIENTIFIC).

        Returns:
            List of non-overlapping DetectedReference objects sorted by start offset.
        """
        if not text or not text.strip():
            return []

        raw_candidates: List[DetectedReference] = []

        # 1. Scientific citations
        if domain is None or domain == Domain.SCIENTIFIC:
            for match in self.PATTERN_NUMERIC.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.NUMERIC.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.NUMERIC.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

            for match in self.PATTERN_AUTHOR_YEAR.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.AUTHOR_YEAR.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.AUTHOR_YEAR.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # 2. Legal references
        if domain is None or domain == Domain.LEGAL:
            for match in self.PATTERN_LEGAL_SECTION.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.LEGAL_SECTION.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.LEGAL_SECTION.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

            for match in self.PATTERN_LEGAL_ARTICLE.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.LEGAL_ARTICLE.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.LEGAL_ARTICLE.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

            for match in self.PATTERN_LEGAL_CLAUSE.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.LEGAL_CLAUSE.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.LEGAL_CLAUSE.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

            for match in self.PATTERN_LEGAL_RULE.finditer(text):
                raw_ref = match.group(0)
                norm_ref, norm_list = self.normalizer.normalize(raw_ref, ReferenceType.LEGAL_RULE.value)
                raw_candidates.append(
                    DetectedReference(
                        reference_text=raw_ref,
                        normalized_reference=norm_ref,
                        normalized_references=norm_list,
                        reference_type=ReferenceType.LEGAL_RULE.value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # Sort by start offset
        raw_candidates.sort(key=lambda r: (r.start, -(r.end - r.start)))

        # Deduplicate and remove overlapping spans (keep longer match)
        non_overlapping: List[DetectedReference] = []
        last_end = -1
        for cand in raw_candidates:
            if cand.start >= last_end:
                non_overlapping.append(cand)
                last_end = cand.end
            else:
                # Overlap: if this candidate is strictly longer, replace previous
                if non_overlapping and (cand.end - cand.start) > (non_overlapping[-1].end - non_overlapping[-1].start):
                    non_overlapping[-1] = cand
                    last_end = cand.end

        return non_overlapping
