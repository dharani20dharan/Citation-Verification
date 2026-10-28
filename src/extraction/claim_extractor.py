"""Claim extractor identifying claim boundaries and associating substantive statements with references."""

import logging
import re
from typing import List, Optional, Tuple

from src.models.schemas import Domain
from src.extraction.citation_detector import CitationDetector
from src.extraction.models import CitationClaim, DetectedReference

logger = logging.getLogger(__name__)

# Abbreviations that should not trigger sentence boundaries
ABBREVIATIONS = {
    "v.",
    "vs.",
    "al.",
    "sec.",
    "art.",
    "no.",
    "e.g.",
    "i.e.",
    "dr.",
    "prof.",
    "mr.",
    "mrs.",
    "ms.",
    "inc.",
    "ltd.",
    "corp.",
    "co.",
    "ag.",
}

# Introductory prepositional phrases before statutory references
INTRO_LEGAL_PREFIX_RE = re.compile(
    r"^(?:Under|Pursuant to|According to|In accordance with|As per|By virtue of|Under the provisions of)\s+",
    re.IGNORECASE,
)

# Conjunction splitters for intra-sentence clause separation
CLAUSE_SPLIT_RE = re.compile(
    r"(?:;\s*|,\s*(?:and|but|while|whereas)\s+|\s+(?:and|but|while|whereas)\s+)",
    re.IGNORECASE,
)


# Heading title patterns and keywords
HEADING_TITLE_RE = re.compile(
    r"^(?:\d+[\.\)]?\s*)?(?:Section|Sec\.|Article|Art\.|Clause|Rule|Part|Chapter)?\s*[\w\(\)]*\s*"
    r"(?:Provisions?|Overview|Introduction|Background|Definitions?|Methodology|Results|Discussion|Conclusions?|Analysis|Scope|General|Exceptions?|Rules?|Clauses?|Title)\s*$",
    re.IGNORECASE,
)

HEADING_NOUNS = {
    "provisions",
    "provision",
    "overview",
    "introduction",
    "background",
    "definitions",
    "definition",
    "summary",
    "general",
    "analysis",
    "exceptions",
    "exception",
    "rules",
    "rule",
    "clauses",
    "clause",
    "notes",
    "title",
    "conclusion",
    "methodology",
}

# Verbs typically starting a predicate clause
PREDICATE_VERBS = {
    "improves",
    "provides",
    "achieves",
    "reduces",
    "demonstrates",
    "requires",
    "exhibits",
    "outperforms",
    "shows",
    "indicates",
    "prevents",
    "prohibits",
    "upholds",
    "leads",
    "yields",
    "enables",
    "enhances",
    "decreases",
    "increases",
    "eliminates",
    "ensures",
    "confirms",
}

SUBJECT_NP_RE = re.compile(
    r"^((?:(?:The|A|An|This|That|These|Those|Our|Their|Its|Such|Each|Every)\s+)?(?:[a-zA-Z0-9_\-]+\s+){0,3}[a-zA-Z0-9_\-]+)\s+(?=(?:is|are|was|were|has|have|can|will|reduces|improves|achieves|demonstrates|provides|exhibits|requires|outperforms|shows|indicates|prevents|prohibits|upholds|[a-z]+s|[a-z]+ed)\b)",
    re.IGNORECASE,
)


def is_structural_heading(text: str, cleaned_claim: str) -> bool:
    """Determine whether text is a structural heading rather than a substantive claim."""
    cleaned_lower = cleaned_claim.lower().strip(" .:")
    if cleaned_lower in HEADING_NOUNS:
        return True
    if HEADING_TITLE_RE.match(text.strip()):
        return True
    # If text starts with section numbering and has <= 2 words after removing reference
    words = cleaned_claim.split()
    if len(words) <= 2 and re.match(r"^\d+[\.\)]?\s+", text.strip()):
        return True
    return False


def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences while tracking exact character spans and protecting abbreviations.

    Args:
        text: Source text.

    Returns:
        List of tuples (sentence_text, start_offset, end_offset).
    """
    if not text:
        return []

    # Find potential sentence terminators: '.', '!', '?'
    spans: List[Tuple[str, int, int]] = []
    start = 0
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        if ch in {".", "!", "?"}:
            # Check if this terminator is part of a protected abbreviation
            word_start = i
            while word_start > 0 and not text[word_start - 1].isspace():
                word_start -= 1
            word = text[word_start : i + 1].lower()

            if word in ABBREVIATIONS:
                i += 1
                continue

            # Look ahead for whitespace and boundary
            j = i + 1
            while j < length and text[j].isspace():
                j += 1

            if j < length:
                next_ch = text[j]
                # If followed by capital letter, digit, or opening bracket/quote, treat as boundary
                if next_ch.isupper() or next_ch.isdigit() or next_ch in {'"', "'", "(", "["}:
                    sent_text = text[start : i + 1].strip()
                    if sent_text:
                        spans.append((sent_text, start, i + 1))
                    start = j
                    i = j
                    continue
            else:
                # End of text
                sent_text = text[start:length].strip()
                if sent_text:
                    spans.append((sent_text, start, length))
                start = length
                break
        i += 1

    if start < length:
        trailing = text[start:length].strip()
        if trailing:
            spans.append((trailing, start, length))

    return spans if spans else [(text.strip(), 0, len(text))]


class ClaimExtractor:
    """Extracts substantive claims and associates them with detected citations/references."""

    def __init__(self, detector: Optional[CitationDetector] = None):
        self.detector = detector or CitationDetector()

    def _is_essential_grammatical_argument(
        self,
        raw_text: str,
        ref_text: str,
        ref_start: Optional[int] = None,
    ) -> bool:
        """Determine if a reference is an essential grammatical element of the proposition.

        Annotation citations (e.g. numeric brackets '[12]', author-year '(Smith, 2024)',
        or parenthetical references) can be excised leaving a complete syntactic clause.
        In contrast, statutory or entity references functioning as direct objects
        (e.g. 'upheld Section 3(d)'), subjects, or prepositional arguments must be
        preserved to keep the verified proposition complete and accurate.
        """
        clean_ref = ref_text.strip()

        # 1. Bracketed or parenthetical references are annotations
        if (clean_ref.startswith("[") and clean_ref.endswith("]")) or (
            clean_ref.startswith("(") and clean_ref.endswith(")")
        ):
            return False

        # 2. Check if reference in raw_text is enclosed in parentheses: e.g. "(Section 3(d))"
        paren_pattern = re.compile(r"\(\s*" + re.escape(clean_ref) + r"\s*\)")
        if paren_pattern.search(raw_text):
            return False

        # 3. Locate reference in raw_text if offset not provided
        idx = ref_start if ref_start is not None and ref_start >= 0 else raw_text.find(ref_text)
        if idx == -1:
            return False

        prefix = raw_text[:idx].rstrip()

        # 4. Appositive citation after comma/semicolon at the end of a clause
        # e.g. "The mere discovery of a new form is not patentable, Section 3(d)."
        if prefix.endswith(",") or prefix.endswith(";") or prefix.endswith("–") or prefix.endswith("—"):
            return False

        # 5. If preceded by a verb, preposition, or clause boundary, it is a grammatical argument
        # (e.g. "upheld Section 3(d)", "interprets Section 3(d)", "under Section 3(d)")
        return True

    def _clean_claim_text(
        self,
        raw_text: str,
        ref_text: str,
        is_legal_start: bool = False,
        ref_start: Optional[int] = None,
    ) -> str:
        """Strip citation marker and clean up surrounding punctuation/prepositions.

        When the reference is an essential grammatical part of the proposition (e.g. the
        direct object in 'the Supreme Court upheld Section 3(d)'), the reference is preserved
        in the substantive claim text while remaining recorded in citation metadata.

        Args:
            raw_text: Raw sentence or clause containing the reference.
            ref_text: Exact reference text.
            is_legal_start: Whether the reference appeared at the beginning of legal sentence.
            ref_start: Optional character offset of ref_text within raw_text.

        Returns:
            Cleaned substantive claim text representing the proposition.
        """
        text = raw_text

        # If citation is at start with introductory preposition or statutory start (e.g. "Under Section 3(d), ..." or "Section 3(e) ...")
        if is_legal_start:
            pattern = re.compile(
                r"^(?:Under|Pursuant to|According to|In accordance with|As per|By virtue of)?\s*"
                + re.escape(ref_text)
                + r"[,:\s]*",
                re.IGNORECASE,
            )
            text = pattern.sub("", text)
        elif self._is_essential_grammatical_argument(text, ref_text, ref_start):
            # Preserve essential grammatical argument in claim text (e.g. direct object in "upheld Section 3(d)")
            text = raw_text
        else:
            # Replace annotation citation with whitespace
            text = text.replace(ref_text, " ")

        # Normalize internal whitespace
        cleaned = re.sub(r"\s+", " ", text).strip()

        # Remove leading/trailing stray commas, semicolons, dashes, colons, or periods
        cleaned = re.sub(r"^[,\s;:–—\-]+", "", cleaned)
        cleaned = re.sub(r"[,\s;:–—\.]+$", "", cleaned)

        return cleaned.strip()

    def _extract_subject_prefix(self, clause: str) -> Optional[str]:
        """Detect subject + auxiliary/copula verb prefix (e.g. 'The method is')."""
        match = re.match(
            r"^((?:The|A|An|This|That|These|Those|Our|Their|Its|Such)\s+[a-zA-Z0-9_\-]+\s+(?:is|are|was|were|has\s+been|have\s+been|can\s+be|will\s+be))\s+",
            clause.strip(),
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    def _extract_subject_noun_phrase(self, clause: str) -> Optional[str]:
        """Detect subject noun phrase at the start of a clause (e.g. 'The proposed architecture')."""
        match = SUBJECT_NP_RE.match(clause.strip())
        return match.group(1).strip() if match else None

    def extract_claims_from_text(
        self,
        text: str,
        domain: Domain,
        document_id: str,
        source_chunk_id: Optional[str] = None,
        page: Optional[int] = None,
        section: Optional[str] = None,
        section_id: Optional[str] = None,
        start_claim_index: int = 1,
    ) -> List[CitationClaim]:
        """Extract substantive CitationClaim objects from raw text with complete provenance.

        Args:
            text: Text to extract from (e.g. chunk text).
            domain: Domain enum.
            document_id: Source document ID.
            source_chunk_id: Originating chunk ID.
            page: Page number.
            section: Section title.
            section_id: Section identifier.
            start_claim_index: Starting sequential sequence number for claim IDs.

        Returns:
            List of validated CitationClaim objects.
        """
        if not text or not text.strip():
            return []

        # 1. Detect all citations in the text
        all_refs = self.detector.detect_references(text, domain=domain)
        if not all_refs:
            return []

        # 2. Partition text into sentences with exact character spans
        sentences = split_sentences_with_spans(text)
        claims: List[CitationClaim] = []
        claim_seq = start_claim_index

        for sent_text, sent_start, sent_end in sentences:
            # Find references situated within this sentence
            sent_refs = [
                r for r in all_refs
                if r.start >= sent_start and r.end <= sent_end
            ]
            if not sent_refs:
                continue

            # Check if sentence has multiple distinct references situated in separate clauses
            # e.g. "The method is faster [12] and more accurate [13]."
            if len(sent_refs) > 1:
                # Check for clause splitting
                split_matches = list(CLAUSE_SPLIT_RE.finditer(sent_text))
                # Check if a conjunction exists between first and second ref
                first_ref_rel = sent_refs[0].end - sent_start
                second_ref_rel = sent_refs[1].start - sent_start

                split_point = None
                split_match = None
                for sm in split_matches:
                    if first_ref_rel <= sm.start() and sm.end() <= second_ref_rel:
                        split_point = sm.start()
                        split_match = sm
                        break

                if split_point is not None and split_match is not None:
                    # Clause 1: start to split_match.start()
                    clause_1_raw = sent_text[: split_match.start()].strip()
                    ref_1 = sent_refs[0]
                    claim_1_text = self._clean_claim_text(clause_1_raw, ref_1.reference_text)

                    # Clause 2: after split_match to end
                    clause_2_raw = sent_text[split_match.end() :].strip()
                    ref_2 = sent_refs[1]
                    claim_2_text = self._clean_claim_text(clause_2_raw, ref_2.reference_text)

                    # Check if clause 2 lacks subject and can inherit from clause 1
                    subj_np = self._extract_subject_noun_phrase(clause_1_raw)
                    copula_prefix = self._extract_subject_prefix(claim_1_text)

                    # Inspect beginning of claim_2_text
                    c2_words = claim_2_text.split()
                    if c2_words:
                        first_w = c2_words[0].lower()
                        is_verb = first_w in PREDICATE_VERBS or (
                            len(first_w) > 3
                            and (first_w.endswith("s") or first_w.endswith("ed"))
                            and first_w not in {"this", "thus", "less", "yes", "as", "is"}
                        )
                        if is_verb and subj_np:
                            claim_2_text = f"{subj_np} {claim_2_text}"
                        elif first_w in {"more", "less", "higher", "lower", "better", "faster", "slower", "superior", "inferior", "accurate"}:
                            if copula_prefix:
                                claim_2_text = f"{copula_prefix} {claim_2_text}"
                            elif subj_np:
                                claim_2_text = f"{subj_np} is {claim_2_text}"

                    # Claim 1
                    if claim_1_text and not is_structural_heading(clause_1_raw, claim_1_text):
                        char_start = sent_start
                        char_end = sent_start + len(clause_1_raw)
                        ctx_before = text[max(0, char_start - 100) : char_start].strip() or None
                        ctx_after = text[char_end : min(len(text), char_end + 100)].strip() or None

                        claims.append(
                            CitationClaim(
                                claim_id=f"{document_id}_claim_{claim_seq:03d}",
                                document_id=document_id,
                                domain=domain,
                                claim_text=claim_1_text,
                                reference_text=ref_1.reference_text,
                                normalized_reference=ref_1.normalized_reference,
                                normalized_references=ref_1.normalized_references,
                                reference_type=ref_1.reference_type,
                                source_chunk_id=source_chunk_id,
                                page=page,
                                section=section,
                                section_id=section_id,
                                character_start=char_start,
                                character_end=char_end,
                                context_before=ctx_before,
                                context_after=ctx_after,
                            )
                        )
                        claim_seq += 1

                    # Claim 2
                    if claim_2_text and not is_structural_heading(clause_2_raw, claim_2_text):
                        char_start = sent_start + split_match.end()
                        char_end = sent_end
                        ctx_before = text[max(0, char_start - 100) : char_start].strip() or None
                        ctx_after = text[char_end : min(len(text), char_end + 100)].strip() or None

                        claims.append(
                            CitationClaim(
                                claim_id=f"{document_id}_claim_{claim_seq:03d}",
                                document_id=document_id,
                                domain=domain,
                                claim_text=claim_2_text,
                                reference_text=ref_2.reference_text,
                                normalized_reference=ref_2.normalized_reference,
                                normalized_references=ref_2.normalized_references,
                                reference_type=ref_2.reference_type,
                                source_chunk_id=source_chunk_id,
                                page=page,
                                section=section,
                                section_id=section_id,
                                character_start=char_start,
                                character_end=char_end,
                                context_before=ctx_before,
                                context_after=ctx_after,
                            )
                        )
                        claim_seq += 1
                    continue

            # Standard single reference or grouped references in sentence (e.g. [12, 13] or Section 3(d))
            for ref in sent_refs:
                is_legal_start = False
                rel_ref_start = ref.start - sent_start
                # Check if legal reference starts the sentence (allowing introductory prepositions)
                if domain == Domain.LEGAL and rel_ref_start < 40:
                    prefix_slice = sent_text[:rel_ref_start]
                    if not prefix_slice or INTRO_LEGAL_PREFIX_RE.match(prefix_slice):
                        is_legal_start = True

                cleaned_claim = self._clean_claim_text(
                    sent_text,
                    ref.reference_text,
                    is_legal_start=is_legal_start,
                    ref_start=rel_ref_start,
                )

                if not cleaned_claim or is_structural_heading(sent_text, cleaned_claim):
                    continue

                ctx_before = text[max(0, sent_start - 100) : sent_start].strip() or None
                ctx_after = text[sent_end : min(len(text), sent_end + 100)].strip() or None

                claims.append(
                    CitationClaim(
                        claim_id=f"{document_id}_claim_{claim_seq:03d}",
                        document_id=document_id,
                        domain=domain,
                        claim_text=cleaned_claim,
                        reference_text=ref.reference_text,
                        normalized_reference=ref.normalized_reference,
                        normalized_references=ref.normalized_references,
                        reference_type=ref.reference_type,
                        source_chunk_id=source_chunk_id,
                        page=page,
                        section=section,
                        section_id=section_id,
                        character_start=sent_start,
                        character_end=sent_end,
                        context_before=ctx_before,
                        context_after=ctx_after,
                    )
                )
                claim_seq += 1

        return claims
