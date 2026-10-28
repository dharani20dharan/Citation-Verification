"""Deterministic reference normalizer for Stage 10A bibliographic resolution."""

import re
from typing import List, NamedTuple, Optional


class NormalizedReference(NamedTuple):
    """Normalized structured representation of a citation reference for resolution."""
    raw_reference: str
    normalized_string: str
    reference_type: str
    lookup_targets: List[str]
    author_surname: Optional[str] = None
    publication_year: Optional[int] = None
    is_range: bool = False
    is_legal: bool = False


# Numeric single: [1], [12]
NUMERIC_SINGLE_RE = re.compile(r"^\[\s*(\d+)\s*\]$")

# Numeric multiple: [3, 4], [3,4], [1, 2, 5]
NUMERIC_MULTIPLE_RE = re.compile(r"^\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]$")

# Numeric range: [5-7], [5–7], [5—7]
NUMERIC_RANGE_RE = re.compile(r"^\[\s*(\d+)\s*[\u2013\u2014\-]\s*(\d+)\s*\]$")

# Author-Year: (Smith et al., 2024), (Smith, 2024), (Smith & Jones, 2023)
AUTHOR_YEAR_RE = re.compile(
    r"^\(?([A-Z][a-zA-Z\s\-]+?(?:\s+et\s+al\.?|\s+and\s+[A-Za-z\s\-]+|\s+&\s+[A-Za-z\s\-]+)?),\s*([12]\d{3}[a-z]?)\)?$"
)

# Legal structured patterns: Section 3(d), Article 14, Clause 7, Rule 12, § 3(d)
LEGAL_RE = re.compile(
    r"^(?:Section|Sec\.|§§?|Article|Art\.|Clause|Rule)\s*(\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)$",
    re.IGNORECASE,
)


class ReferenceNormalizer:
    """Normalizes citation references into canonical lookup targets for resolution."""

    def normalize(self, reference_text: str) -> NormalizedReference:
        """Parse and normalize any reference into structured lookup targets.

        Args:
            reference_text: Raw reference string (e.g. '[12]', '[3, 4]', '[5-7]', '(Smith et al., 2024)').

        Returns:
            NormalizedReference with individual target identifiers and metadata.
        """
        raw = reference_text.strip()

        # 1. Check Legal reference
        m_legal = LEGAL_RE.match(raw)
        if m_legal:
            norm_legal = m_legal.group(1)
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=norm_legal,
                reference_type="legal_section",
                lookup_targets=[norm_legal],
                is_legal=True,
            )

        # 2. Check Numeric Range e.g. [5-7] or [5–7]
        m_range = NUMERIC_RANGE_RE.match(raw)
        if m_range:
            start_num = int(m_range.group(1))
            end_num = int(m_range.group(2))
            norm_range_str = f"{start_num}-{end_num}"
            if start_num <= end_num and (end_num - start_num) <= 50:
                expanded_targets = [str(i) for i in range(start_num, end_num + 1)]
            else:
                expanded_targets = [str(start_num), str(end_num)]

            return NormalizedReference(
                raw_reference=raw,
                normalized_string=norm_range_str,
                reference_type="numeric_range",
                lookup_targets=expanded_targets,
                is_range=True,
            )

        # 3. Check Numeric Multiple e.g. [3, 4]
        m_multi = NUMERIC_MULTIPLE_RE.match(raw)
        if m_multi:
            items_str = m_multi.group(1)
            targets = [t.strip() for t in items_str.split(",") if t.strip()]
            norm_str = ", ".join(targets)
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=norm_str,
                reference_type="numeric_citation",
                lookup_targets=targets,
                is_range=False,
            )

        # 4. Check Numeric Single e.g. [12]
        m_single = NUMERIC_SINGLE_RE.match(raw)
        if m_single:
            num = m_single.group(1)
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=num,
                reference_type="numeric_citation",
                lookup_targets=[num],
                is_range=False,
            )

        # 5. Check Author-Year e.g. (Smith et al., 2024)
        m_ay = AUTHOR_YEAR_RE.match(raw)
        if m_ay:
            authors_str = m_ay.group(1).strip()
            year_str = m_ay.group(2).strip()
            year = int(re.sub(r"[a-z]$", "", year_str))

            # Extract lead surname
            surname_match = re.match(r"^([A-Z][a-zA-Z\-]+)", authors_str)
            surname = surname_match.group(1) if surname_match else authors_str.split()[0]

            norm_str = f"{authors_str}, {year_str}"
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=norm_str,
                reference_type="author_year",
                lookup_targets=[f"{surname}_{year_str}"],
                author_surname=surname,
                publication_year=year,
                is_range=False,
            )

        # Fallback: Bare numeric without brackets e.g. "12"
        if raw.isdigit():
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=raw,
                reference_type="numeric_citation",
                lookup_targets=[raw],
                is_range=False,
            )

        # Fallback: Bare range e.g. "5-7" or "5–7"
        bare_range_match = re.match(r"^(\d+)\s*[\u2013\u2014\-]\s*(\d+)$", raw)
        if bare_range_match:
            s_num = int(bare_range_match.group(1))
            e_num = int(bare_range_match.group(2))
            norm_range_str = f"{s_num}-{e_num}"
            expanded = [str(i) for i in range(s_num, e_num + 1)] if s_num <= e_num else [str(s_num)]
            return NormalizedReference(
                raw_reference=raw,
                normalized_string=norm_range_str,
                reference_type="numeric_range",
                lookup_targets=expanded,
                is_range=True,
            )

        # Default fallback
        clean_fallback = re.sub(r"[^\w\s\-\.]", "", raw).strip()
        return NormalizedReference(
            raw_reference=raw,
            normalized_string=clean_fallback,
            reference_type="generic_reference",
            lookup_targets=[clean_fallback],
            is_range=False,
        )
