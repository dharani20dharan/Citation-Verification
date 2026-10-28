"""Comprehensive test suite for Stage 10A: Bibliography and Reference Resolution."""

import copy
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import ContentType, Domain, EvidenceChunk
from src.references.models import (
    BibliographicEntry,
    ReferenceResolutionResult,
    ResolutionStatus,
    generate_entry_id,
    generate_resolution_id,
)
from src.references.reference_detector import ReferenceSectionDetector
from src.references.reference_normalizer import ReferenceNormalizer
from src.references.reference_parser import BibliographicParser
from src.references.reference_resolver import ReferenceResolver
from src.references.resolution_engine import ReferenceResolutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser() -> BibliographicParser:
    return BibliographicParser()


@pytest.fixture
def normalizer() -> ReferenceNormalizer:
    return ReferenceNormalizer()


@pytest.fixture
def detector() -> ReferenceSectionDetector:
    return ReferenceSectionDetector()


@pytest.fixture
def resolver() -> ReferenceResolver:
    return ReferenceResolver()


@pytest.fixture
def sample_bibliography(parser: BibliographicParser) -> list[BibliographicEntry]:
    """Synthetic reference fixture containing targets [3], [4], [5], [6], [7], [12], and Smith et al., 2024."""
    doc_id = "sci_eval_01"
    raw_items = [
        {
            "raw_text": "[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. Attention is All You Need. NeurIPS, 2017.",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_003",
        },
        {
            "raw_text": "[4] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_004",
        },
        {
            "raw_text": "[5] Thorne, J., Vlachos, A., Christodoulopoulos, C., Mittal, A. FEVER: a large-scale dataset for Fact Extraction and VERification. NAACL-HLT, 2018.",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_005",
        },
        {
            "raw_text": "[6] Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., Hajishirzi, H. Fact or Fiction: Verifying Scientific Claims. EMNLP, 2020.",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_006",
        },
        {
            "raw_text": "[7] Schuster, T., Shah, D., Yeo, Y. J., Filizzola, R., Santus, E., Barzilay, R. Toward Robust Fact-Checking: Analyzing and Mitigating Bias in FEVER. EMNLP, 2019.",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_007",
        },
        {
            "raw_text": "[12] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., et al. Language Models are Few-Shot Learners. NeurIPS, 2020. doi:10.48550/arXiv.2005.14165",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_012",
        },
        {
            "raw_text": "Smith, J., Johnson, K., and Williams, R. (2024). Precision and Recall in Automated Citation Verification. Journal of Artificial Intelligence Research, 79, 101-125. https://doi.org/10.1613/jair.1.12345",
            "page": 3,
            "section": "References",
            "chunk_id": "sci_eval_01_ref_smith_2024",
        },
    ]

    entries = []
    for item in raw_items:
        entry = parser.parse_entry(
            raw_text=item["raw_text"],
            document_id=doc_id,
            domain=Domain.SCIENTIFIC,
            page=item["page"],
            section=item["section"],
            chunk_id=item["chunk_id"],
        )
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# 1. Model & Schema Tests
# ---------------------------------------------------------------------------

class TestReferenceModels:
    """Test schema invariants, validation, and ID generation."""

    def test_bibliographic_entry_valid(self):
        entry = BibliographicEntry(
            entry_id="doc1_ref_12",
            document_id="doc1",
            domain=Domain.SCIENTIFIC,
            reference_label="[12]",
            normalized_label="12",
            raw_text="[12] Brown, T. et al. Language Models... 2020.",
            title="Language Models",
            authors=["Brown, T."],
            year=2020,
            page=4,
            section="References",
            provenance={"chunk_id": "chunk_012"},
        )
        assert entry.entry_id == "doc1_ref_12"
        assert entry.normalized_label == "12"
        assert entry.chunk_id == "chunk_012"
        assert entry.domain == Domain.SCIENTIFIC

    def test_bibliographic_entry_empty_ids_rejected(self):
        with pytest.raises(ValidationError):
            BibliographicEntry(
                entry_id="   ",
                document_id="doc1",
                reference_label="[12]",
                normalized_label="12",
                raw_text="...",
            )

    def test_bibliographic_entry_optional_fields_none(self):
        entry = BibliographicEntry(
            entry_id="doc1_ref_1",
            document_id="doc1",
            reference_label="[1]",
            normalized_label="1",
            raw_text="[1] Just some text.",
        )
        assert entry.title is None
        assert entry.year is None
        assert entry.doi is None
        assert entry.url is None
        assert entry.page is None
        assert entry.provenance == {}

    def test_resolution_result_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ReferenceResolutionResult(
                resolution_id="res_01",
                reference_text="[12]",
                normalized_reference="12",
                reference_type="numeric_citation",
                status=ResolutionStatus.RESOLVED,
                confidence=1.5,  # > 1.0 invalid
                reasoning="test",
            )

    def test_deterministic_id_generation(self):
        id1 = generate_entry_id("doc_alpha", "12")
        id2 = generate_entry_id("doc_alpha", "12")
        assert id1 == id2 == "doc_alpha_ref_12"

        res_id1 = generate_resolution_id("[12]", "doc1", "claim1")
        res_id2 = generate_resolution_id("[12]", "doc1", "claim1")
        assert res_id1 == res_id2
        assert res_id1.startswith("res_")


# ---------------------------------------------------------------------------
# 2. Reference Section Detection Tests
# ---------------------------------------------------------------------------

class TestReferenceSectionDetector:
    """Test detection and extraction of reference sections from document structure."""

    def test_is_reference_heading(self, detector: ReferenceSectionDetector):
        headings = [
            "References",
            "REFERENCES",
            "10. References",
            "Bibliography",
            "Literature Cited",
            "Works Cited",
            "Reference List",
        ]
        for h in headings:
            assert detector.is_reference_heading(h), f"Expected '{h}' to be recognized as reference heading"

        non_headings = [
            "Introduction",
            "Methodology",
            "Results and Discussion",
            "Related Work",
        ]
        for nh in non_headings:
            assert not detector.is_reference_heading(nh), f"Expected '{nh}' NOT to be recognized as reference heading"

    def test_is_terminating_heading(self, detector: ReferenceSectionDetector):
        assert detector.is_terminating_heading("Appendix A: Proofs")
        assert detector.is_terminating_heading("Supplementary Material")
        assert detector.is_terminating_heading("Author Contributions")
        assert detector.is_terminating_heading("Acknowledgements")
        assert not detector.is_terminating_heading("References")

    def test_detect_from_elements_skips_preceding_sections(self, detector: ReferenceSectionDetector):
        elements = [
            {"element_type": "heading", "text": "1. Introduction", "page": 1, "element_id": "e1"},
            {"element_type": "paragraph", "text": "This paper explores citation verification [12].", "page": 1, "element_id": "e2"},
            {"element_type": "heading", "text": "References", "page": 2, "element_id": "e3"},
            {"element_type": "paragraph", "text": "[1] First ref entry.", "page": 2, "element_id": "e4"},
            {"element_type": "paragraph", "text": "[2] Second ref entry.", "page": 2, "element_id": "e5"},
            {"element_type": "heading", "text": "Appendix", "page": 3, "element_id": "e6"},
            {"element_type": "paragraph", "text": "Proof of Lemma 1.", "page": 3, "element_id": "e7"},
        ]
        refs = detector.detect_from_elements(elements, document_id="doc_test")
        assert len(refs) == 2
        assert refs[0]["raw_text"] == "[1] First ref entry."
        assert refs[1]["raw_text"] == "[2] Second ref entry."
        assert refs[0]["page"] == 2
        assert refs[0]["section"] == "References"

    def test_detect_from_chunks(self, detector: ReferenceSectionDetector):
        chunks = [
            EvidenceChunk(
                chunk_id="c1",
                document_id="doc1",
                title="Test Document",
                source="test.pdf",
                section_id="sec_01",
                content_type=ContentType.PARAGRAPH,
                text="In section 2 we describe methods.",
                section="Methods",
                page=1,
                domain=Domain.SCIENTIFIC,
            ),
            EvidenceChunk(
                chunk_id="c2",
                document_id="doc1",
                title="Test Document",
                source="test.pdf",
                section_id="sec_02",
                content_type=ContentType.PARAGRAPH,
                text="[1] Alpha, A. First paper.\n[2] Beta, B. Second paper.",
                section="References",
                page=3,
                domain=Domain.SCIENTIFIC,
            ),
        ]
        refs = detector.detect_from_chunks(chunks, document_id="doc1")
        assert len(refs) == 2
        assert refs[0]["raw_text"] == "[1] Alpha, A. First paper."
        assert refs[0]["chunk_id"] == "c2"
        assert refs[1]["raw_text"] == "[2] Beta, B. Second paper."


# ---------------------------------------------------------------------------
# 3. Parsing and Normalization Tests
# ---------------------------------------------------------------------------

class TestReferenceParserAndNormalizer:
    """Test parsing raw strings into entries and normalizing reference citation queries."""

    def test_parse_bracketed_numeric(self, parser: BibliographicParser):
        raw = "[12] Brown, T. Language Models. NeurIPS, 2020. doi:10.48550/arXiv.2005.14165"
        entry = parser.parse_entry(raw, document_id="doc1")
        assert entry.reference_label == "[12]"
        assert entry.normalized_label == "12"
        assert entry.year == 2020
        assert entry.doi == "10.48550/arXiv.2005.14165"
        assert entry.raw_text == raw

    def test_parse_dotted_numeric(self, parser: BibliographicParser):
        raw = "1. Smith, J. Title of paper. Journal, 2024."
        entry = parser.parse_entry(raw, document_id="doc1")
        assert entry.reference_label == "1."
        assert entry.normalized_label == "1"
        assert entry.year == 2024
        assert entry.raw_text == raw

    def test_parse_author_year_style(self, parser: BibliographicParser):
        raw = "Smith, J., Johnson, K. (2024). Automated Verification. JAIR, 2024."
        entry = parser.parse_entry(raw, document_id="doc1")
        assert entry.normalized_label == "Smith_2024"
        assert entry.year == 2024
        assert entry.raw_text == raw

    def test_year_extraction_avoids_arxiv_id_collision(self, parser: BibliographicParser):
        raw = "[12] Brown, T. et al. Language Models. 2020. doi:10.48550/arXiv.2005.14165"
        entry = parser.parse_entry(raw, document_id="doc1")
        # 2005 is part of the arXiv ID, actual publication year is 2020
        assert entry.year == 2020

    def test_normalizer_single_numeric(self, normalizer: ReferenceNormalizer):
        norm = normalizer.normalize("[12]")
        assert norm.lookup_targets == ["12"]
        assert norm.normalized_string == "12"
        assert not norm.is_range
        assert not norm.is_legal

    def test_normalizer_multiple_numeric(self, normalizer: ReferenceNormalizer):
        norm = normalizer.normalize("[3, 4]")
        assert norm.lookup_targets == ["3", "4"]
        assert norm.normalized_string == "3, 4"
        assert not norm.is_range

        norm2 = normalizer.normalize("[3,4]")
        assert norm2.lookup_targets == ["3", "4"]

    def test_normalizer_range(self, normalizer: ReferenceNormalizer):
        for raw in ["[5-7]", "[5–7]", "[5—7]"]:
            norm = normalizer.normalize(raw)
            assert norm.is_range
            assert norm.normalized_string == "5-7"
            assert norm.lookup_targets == ["5", "6", "7"]

    def test_normalizer_author_year(self, normalizer: ReferenceNormalizer):
        norm = normalizer.normalize("(Smith et al., 2024)")
        assert norm.reference_type == "author_year"
        assert norm.author_surname == "Smith"
        assert norm.publication_year == 2024
        assert norm.lookup_targets == ["Smith_2024"]

    def test_normalizer_legal(self, normalizer: ReferenceNormalizer):
        norm = normalizer.normalize("Section 3(d)")
        assert norm.is_legal
        assert norm.reference_type == "legal_section"
        assert norm.lookup_targets == ["3(d)"]


# ---------------------------------------------------------------------------
# 4. Numeric Reference Resolution Tests
# ---------------------------------------------------------------------------

class TestNumericResolution:
    """Test resolution of numeric citations ([12], [99], duplicates)."""

    def test_numeric_single_resolves(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[12]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.RESOLVED
        assert len(res.matched_entries) == 1
        assert res.matched_entries[0].normalized_label == "12"
        assert res.matched_entries[0].entry_id == "sci_eval_01_ref_12"
        assert res.confidence == 0.95
        assert "sci_eval_01_ref_12" in res.matched_entry_ids

    def test_numeric_single_missing_returns_not_found(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[99]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.NOT_FOUND
        assert len(res.matched_entries) == 0
        assert res.matched_entry_ids == []
        assert res.confidence == 0.95
        assert "not found" in res.reasoning.lower()

    def test_numeric_single_duplicate_returns_ambiguous(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        # Create duplicate entry with same normalized_label
        dup = copy.deepcopy(sample_bibliography[0])
        dup.entry_id = "sci_eval_01_ref_3_dup"
        dup.normalized_label = "3"
        dup_bib = sample_bibliography + [dup]

        res = resolver.resolve("[3]", dup_bib, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.AMBIGUOUS
        assert len(res.matched_entries) == 2
        assert res.confidence == 0.70
        assert "ambiguous" in res.reasoning.lower()


# ---------------------------------------------------------------------------
# 5. Multiple Numeric Citations Tests
# ---------------------------------------------------------------------------

class TestMultipleNumericResolution:
    """Test resolution of grouped numeric citations ([3,4], [3,99])."""

    def test_multiple_all_present_resolves(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[3,4]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.RESOLVED
        assert len(res.matched_entries) == 2
        matched_labels = {e.normalized_label for e in res.matched_entries}
        assert matched_labels == {"3", "4"}
        assert res.confidence == 0.95

    def test_multiple_partially_present_returns_partially_resolved(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[3,99]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.PARTIALLY_RESOLVED
        assert len(res.matched_entries) == 1
        assert res.matched_entries[0].normalized_label == "3"
        assert res.provenance["resolved_targets"] == ["3"]
        assert res.provenance["missing_targets"] == ["99"]

    def test_multiple_none_present_returns_not_found(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[98,99]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.NOT_FOUND
        assert len(res.matched_entries) == 0


# ---------------------------------------------------------------------------
# 6. Numeric Range Resolution Tests
# ---------------------------------------------------------------------------

class TestRangeResolution:
    """Test resolution of numeric ranges ([5-7], missing range members)."""

    def test_range_all_present_resolves_individually(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[5–7]", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.RESOLVED
        assert len(res.matched_entries) == 3
        labels = [e.normalized_label for e in res.matched_entries]
        assert labels == ["5", "6", "7"]
        assert res.provenance["targets"] == ["5", "6", "7"]
        assert res.provenance["is_range"] is True

    def test_range_missing_member_returns_partially_resolved(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        # Exclude entry [6]
        bib_without_6 = [e for e in sample_bibliography if e.normalized_label != "6"]
        res = resolver.resolve("[5–7]", bib_without_6, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.PARTIALLY_RESOLVED
        assert len(res.matched_entries) == 2
        labels = [e.normalized_label for e in res.matched_entries]
        assert labels == ["5", "7"]
        assert "6" in res.provenance["missing_targets"]


# ---------------------------------------------------------------------------
# 7. Author-Year Resolution Tests
# ---------------------------------------------------------------------------

class TestAuthorYearResolution:
    """Test deterministic author-year metadata resolution."""

    def test_author_year_uniquely_resolves(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("(Smith et al., 2024)", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.RESOLVED
        assert len(res.matched_entries) == 1
        entry = res.matched_entries[0]
        assert entry.entry_id == "sci_eval_01_ref_smith_2024"
        assert entry.year == 2024
        assert res.confidence == 0.90

    def test_author_year_duplicate_returns_ambiguous(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry], parser: BibliographicParser
    ):
        # Add another distinct publication by Smith in 2024
        another_smith = parser.parse_entry(
            raw_text="Smith, J., Davis, P. (2024). Second Paper on Verification. ICML, 2024.",
            document_id="sci_eval_01",
        )
        another_smith.entry_id = "sci_eval_01_ref_smith_2024_second"
        extended_bib = sample_bibliography + [another_smith]

        res = resolver.resolve("(Smith et al., 2024)", extended_bib, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.AMBIGUOUS
        assert len(res.matched_entries) == 2
        assert res.confidence == 0.70

    def test_author_year_missing_returns_not_found(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("(Johnson et al., 2015)", sample_bibliography, document_id="sci_eval_01")
        assert res.status == ResolutionStatus.NOT_FOUND
        assert len(res.matched_entries) == 0
        assert res.confidence == 0.95


# ---------------------------------------------------------------------------
# 8. Legal Reference Handling Tests
# ---------------------------------------------------------------------------

class TestLegalReferenceHandling:
    """Test that legal structured references route to NOT_APPLICABLE and preserve retrieval routing."""

    def test_legal_section_returns_not_applicable(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("Section 3(d)", sample_bibliography, document_id="legal_sample_fixture")
        assert res.status == ResolutionStatus.NOT_APPLICABLE
        assert res.confidence == 1.0
        assert len(res.matched_entries) == 0
        assert res.reference_type == "legal_section"
        assert "stage_7_structural_retrieval" in res.provenance.get("legal_routing", "")

    def test_legal_article_returns_not_applicable(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("Article 14", sample_bibliography, document_id="legal_sample_fixture")
        assert res.status == ResolutionStatus.NOT_APPLICABLE
        assert res.confidence == 1.0


# ---------------------------------------------------------------------------
# 9. Provenance and Raw Text Preservation Tests
# ---------------------------------------------------------------------------

class TestProvenanceAndRawText:
    """Verify that document_id, page, section, chunk_id, and original raw_text are strictly preserved."""

    def test_provenance_preservation(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        res = resolver.resolve("[12]", sample_bibliography, document_id="sci_eval_01")
        entry = res.matched_entries[0]

        assert entry.document_id == "sci_eval_01"
        assert entry.page == 3
        assert entry.section == "References"
        assert entry.chunk_id == "sci_eval_01_ref_012"

        # Check resolution provenance trace
        assert res.provenance["matched_entry_id"] == "sci_eval_01_ref_12"
        assert res.provenance["entry_page"] == 3
        assert res.provenance["entry_section"] == "References"
        assert res.provenance["entry_chunk_id"] == "sci_eval_01_ref_012"

    def test_raw_text_preserved_unmodified(
        self, parser: BibliographicParser
    ):
        raw = " [12] Brown, T., Mann, B. Language Models... NeurIPS, 2020. doi:10.48550/arXiv.2005.14165 "
        entry = parser.parse_entry(raw, document_id="doc1")
        assert entry.raw_text == raw.strip()


# ---------------------------------------------------------------------------
# 10. Determinism Tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify repeated execution produces byte-identical IDs, labels, and results."""

    def test_deterministic_parsing(self, parser: BibliographicParser):
        raw = "[12] Brown, T. Language Models are Few-Shot Learners. NeurIPS, 2020."
        e1 = parser.parse_entry(raw, document_id="sci_eval_01")
        e2 = parser.parse_entry(raw, document_id="sci_eval_01")

        assert e1.entry_id == e2.entry_id
        assert e1.normalized_label == e2.normalized_label
        assert e1.model_dump() == e2.model_dump()

    def test_deterministic_resolution(
        self, resolver: ReferenceResolver, sample_bibliography: list[BibliographicEntry]
    ):
        r1 = resolver.resolve("[12]", sample_bibliography, document_id="sci_eval_01", claim_id="claim_01")
        r2 = resolver.resolve("[12]", sample_bibliography, document_id="sci_eval_01", claim_id="claim_01")

        assert r1.resolution_id == r2.resolution_id
        assert r1.status == r2.status
        assert r1.matched_entry_ids == r2.matched_entry_ids
        assert r1.confidence == r2.confidence
        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# 11. Resolution Engine and End-to-End Integration Tests
# ---------------------------------------------------------------------------

class TestReferenceResolutionEngine:
    """Integration test verifying end-to-end flow with ReferenceResolutionEngine."""

    def test_engine_resolution_from_claim(self, sample_bibliography: list[BibliographicEntry]):
        engine = ReferenceResolutionEngine(references_dir=Path("data/references"))
        engine.add_entries(sample_bibliography)

        claim = CitationClaim(
            claim_id="claim_test_12",
            document_id="sci_eval_01",
            claim_text="Language models demonstrate few-shot learning capabilities.",
            reference_text="[12]",
            normalized_reference="12",
            reference_type=ReferenceType.NUMERIC.value,
            page=2,
            chunk_id="chunk_test_01",
            domain=Domain.SCIENTIFIC,
        )

        res = engine.resolve(claim)
        assert res.status == ResolutionStatus.RESOLVED
        assert res.claim_id == "claim_test_12"
        assert res.document_id == "sci_eval_01"
        assert len(res.matched_entries) == 1
        assert res.matched_entries[0].entry_id == "sci_eval_01_ref_12"

    def test_engine_resolves_and_reports_missing_and_partial(
        self, sample_bibliography: list[BibliographicEntry]
    ):
        engine = ReferenceResolutionEngine(references_dir=Path("data/references"))
        engine.add_entries(sample_bibliography)

        # [12] -> resolved
        res_12 = engine.resolve_reference("[12]", domain=Domain.SCIENTIFIC, document_id="sci_eval_01")
        assert res_12.status == ResolutionStatus.RESOLVED

        # [99] -> not_found
        res_99 = engine.resolve_reference("[99]", domain=Domain.SCIENTIFIC, document_id="sci_eval_01")
        assert res_99.status == ResolutionStatus.NOT_FOUND

        # [3, 99] -> partially_resolved
        res_part = engine.resolve_reference("[3, 99]", domain=Domain.SCIENTIFIC, document_id="sci_eval_01")
        assert res_part.status == ResolutionStatus.PARTIALLY_RESOLVED
        assert len(res_part.matched_entries) == 1
        assert res_part.matched_entries[0].normalized_label == "3"
