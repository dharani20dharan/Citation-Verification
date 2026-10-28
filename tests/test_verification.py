"""Comprehensive test suite for Stage 9: Evidence Verification and Verdict Engine."""

import hashlib
from typing import List
import pytest
from pydantic import ValidationError

from src.embeddings.base_embedder import BaseEmbedder
from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import ContentType, Domain, EvidenceChunk
from src.retrieval.exact_match import ExactMatcher
from src.retrieval.models import MatchType, RetrievalRequest, RetrievalResult
from src.retrieval.ranker import DeterministicRanker
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaVectorStore
from src.verification.evidence_selector import EvidenceSelector
from src.verification.models import (
    EvidenceItem,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
    VerificationResult,
)
from src.verification.verification_engine import VerificationEngine
from src.verification.verifier import (
    DeterministicClaimVerifier,
    LLMClaimVerifier,
    LLMStructuredOutput,
    LLMVerificationPayload,
)


class FastMockEmbedder(BaseEmbedder):
    """Deterministic, fast mock embedder for verification testing."""

    def __init__(self, dimension: int = 1024, model_name: str = "mock-embedder"):
        self._dim = dimension
        self._name = model_name

    def embed_text(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [float((h[i % len(h)] * 31 + i * 17) % 256) + 1.0 for i in range(self._dim)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        return "cpu"


@pytest.fixture(scope="module")
def mock_legal_embedder():
    return FastMockEmbedder(dimension=1024, model_name="BAAI/bge-large-en-v1.5")


@pytest.fixture(scope="module")
def mock_sci_embedder():
    return FastMockEmbedder(dimension=768, model_name="allenai/specter2_base")


@pytest.fixture(scope="module")
def populated_vector_store(tmp_path_factory, mock_legal_embedder, mock_sci_embedder):
    """Populate an isolated Chroma vector store with rich legal and scientific fixtures."""
    chroma_dir = tmp_path_factory.mktemp("test_chroma_verification")
    store = ChromaVectorStore(persist_dir=chroma_dir)

    # 1. Legal corpus
    sec_3 = EvidenceChunk(
        chunk_id="patents_sec_3",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="The Patents Act 1970",
        source="legal/patents.pdf",
        page=14,
        section="Section 3",
        section_id="sec_3",
        parent_id=None,
        content_type=ContentType.SECTION,
        text="Section 3. What are not inventions.",
    )
    clause_3d = EvidenceChunk(
        chunk_id="patents_sec_3_d",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="The Patents Act 1970",
        source="legal/patents.pdf",
        page=14,
        section="Section 3(d)",
        section_id="sec_3_d",
        parent_id="patents_sec_3",
        content_type=ContentType.CLAUSE,
        text="The mere discovery of a new form of a known substance is not patentable.",
        clause_id="3(d)",
    )
    subclause_3di = EvidenceChunk(
        chunk_id="patents_sec_3_d_i",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="The Patents Act 1970",
        source="legal/patents.pdf",
        page=14,
        section="Section 3(d)(i)",
        section_id="sec_3_d_i",
        parent_id="patents_sec_3_d",
        content_type=ContentType.CLAUSE,
        text="salts, esters, ethers, polymorphs, metabolites are considered the same substance.",
        clause_id="3(d)(i)",
    )
    subclause_3dii = EvidenceChunk(
        chunk_id="patents_sec_3_d_ii",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="The Patents Act 1970",
        source="legal/patents.pdf",
        page=14,
        section="Section 3(d)(ii)",
        section_id="sec_3_d_ii",
        parent_id="patents_sec_3_d",
        content_type=ContentType.CLAUSE,
        text="combinations and other derivatives of known substance are held unpatentable.",
        clause_id="3(d)(ii)",
    )
    clause_3e = EvidenceChunk(
        chunk_id="patents_sec_3_e",
        document_id="patents_act",
        domain=Domain.LEGAL,
        title="The Patents Act 1970",
        source="legal/patents.pdf",
        page=15,
        section="Section 3(e)",
        section_id="sec_3_e",
        parent_id="patents_sec_3",
        content_type=ContentType.CLAUSE,
        text="A substance obtained by a mere admixture resulting only in aggregation of properties.",
        clause_id="3(e)",
    )
    legal_chunks = [sec_3, clause_3d, subclause_3di, subclause_3dii, clause_3e]
    legal_records = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "domain": "legal",
            "model_name": mock_legal_embedder.model_name,
            "dimension": mock_legal_embedder.dimension,
            "embedding": mock_legal_embedder.embed_text(c.text),
        }
        for c in legal_chunks
    ]
    store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=legal_records,
        chunk_map={c.chunk_id: c for c in legal_chunks},
    )

    # 2. Scientific corpus
    sec_intro = EvidenceChunk(
        chunk_id="sci_sec_01",
        document_id="sci_paper_01",
        domain=Domain.SCIENTIFIC,
        title="Automated Verification Benchmarks",
        source="scientific/benchmarks.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01",
        parent_id=None,
        content_type=ContentType.SECTION,
        text="1. Introduction to automated verification.",
    )
    ref_12_chunk = EvidenceChunk(
        chunk_id="sci_ref_12",
        document_id="sci_paper_01",
        domain=Domain.SCIENTIFIC,
        title="Automated Verification Benchmarks",
        source="scientific/benchmarks.pdf",
        page=5,
        section="References",
        section_id="sec_references",
        parent_id=None,
        content_type=ContentType.PARAGRAPH,
        text="[12] Vaswani et al. Attention is all you need. NeurIPS, 2017.",
        clause_id="12",
    )
    sci_chunks = [sec_intro, ref_12_chunk]
    sci_records = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "domain": "scientific",
            "model_name": mock_sci_embedder.model_name,
            "dimension": mock_sci_embedder.dimension,
            "embedding": mock_sci_embedder.embed_text(c.text),
        }
        for c in sci_chunks
    ]
    store.ingest_records(
        domain=Domain.SCIENTIFIC,
        embedding_records=sci_records,
        chunk_map={c.chunk_id: c for c in sci_chunks},
    )

    return store


@pytest.fixture
def mock_retriever(populated_vector_store, mock_legal_embedder):
    """Retriever using mock embedders for lightning-fast testing."""
    return Retriever(
        vector_store=populated_vector_store,
        embedder_override=mock_legal_embedder,
    )


@pytest.fixture
def sci_retriever(populated_vector_store, mock_sci_embedder):
    return Retriever(
        vector_store=populated_vector_store,
        embedder_override=mock_sci_embedder,
    )


@pytest.fixture
def verification_engine(mock_retriever):
    return VerificationEngine(
        retriever=mock_retriever,
        evidence_selector=EvidenceSelector(max_evidence=5),
        verifier=DeterministicClaimVerifier(),
    )


@pytest.fixture
def sci_verification_engine(sci_retriever):
    return VerificationEngine(
        retriever=sci_retriever,
        evidence_selector=EvidenceSelector(max_evidence=5),
        verifier=DeterministicClaimVerifier(),
    )


# ==============================================================================
# 1. REFERENCE HANDLING TESTS
# ==============================================================================

def test_exact_legal_reference_found(verification_engine):
    claim = CitationClaim(
        claim_id="claim_001",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.reference_status == ReferenceStatus.EXACT_MATCH
    assert any(e.clause_id == "3(d)" and e.match_type == "exact" for e in res.evidence)


def test_exact_legal_reference_not_found(verification_engine):
    claim = CitationClaim(
        claim_id="claim_002",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Provisional specifications must be filed under Section 9.",
        reference_text="Section 9",
        normalized_reference="9",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.reference_status == ReferenceStatus.NOT_FOUND
    assert res.verdict == Verdict.NOT_FOUND
    assert res.confidence == 0.95
    # Critical rule: Semantic similarity candidates must not make reference_status exact_match
    assert res.reference_status != ReferenceStatus.EXACT_MATCH


def test_section_3d_does_not_match_3e(verification_engine):
    claim = CitationClaim(
        claim_id="claim_003",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Substance obtained by mere admixture.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    # The exact matches returned must be strictly 3(d), never 3(e)
    exact_matches = [e for e in res.evidence if e.match_type == "exact"]
    for em in exact_matches:
        assert em.clause_id != "3(e)"
        assert em.chunk_id != "patents_sec_3_e"


def test_section_3d_does_not_become_exact_match_for_3di(verification_engine):
    claim = CitationClaim(
        claim_id="claim_004",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Salts and esters under Section 3(d).",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    exact_matches = [e for e in res.evidence if e.match_type == "exact"]
    for em in exact_matches:
        # 3(d) must not be substituted by 3(d)(i) as an exact match
        assert em.clause_id != "3(d)(i)"


def test_numeric_citation_exact_handling(sci_verification_engine):
    # Found numeric citation [12]
    claim_12 = CitationClaim(
        claim_id="claim_sci_12",
        document_id="doc_sci",
        domain=Domain.SCIENTIFIC,
        claim_text="Attention is all you need architecture.",
        reference_text="[12]",
        normalized_reference="12",
        reference_type="numeric_citation",
    )
    res_12 = sci_verification_engine.verify(claim_12)
    assert res_12.reference_status == ReferenceStatus.EXACT_MATCH

    # Non-existent numeric citation [99]
    claim_99 = CitationClaim(
        claim_id="claim_sci_99",
        document_id="doc_sci",
        domain=Domain.SCIENTIFIC,
        claim_text="Quantum transformer acceleration.",
        reference_text="[99]",
        normalized_reference="99",
        reference_type="numeric_citation",
    )
    res_99 = sci_verification_engine.verify(claim_99)
    assert res_99.reference_status == ReferenceStatus.NOT_FOUND
    assert res_99.verdict == Verdict.NOT_FOUND


# ==============================================================================
# 2. EVIDENCE SELECTION TESTS
# ==============================================================================

def test_evidence_selection_exact_outranks_semantic():
    selector = EvidenceSelector(max_evidence=5)
    exact_res = RetrievalResult(
        chunk_id="chunk_exact",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Exact text",
        score=1.0,
        match_type=MatchType.EXACT.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    semantic_res = RetrievalResult(
        chunk_id="chunk_semantic",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Semantic text",
        score=0.98,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    selected = selector.select_evidence([semantic_res, exact_res])
    assert len(selected) == 2
    assert selected[0].chunk_id == "chunk_exact"
    assert selected[0].match_type == "exact"
    assert selected[1].chunk_id == "chunk_semantic"


def test_evidence_selection_parent_context_retained():
    selector = EvidenceSelector(max_evidence=5)
    exact_res = RetrievalResult(
        chunk_id="child_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Child text",
        score=1.0,
        match_type=MatchType.EXACT.value,
        content_type=ContentType.PARAGRAPH.value,
        parent_id="parent_chunk",
    )
    parent_res = RetrievalResult(
        chunk_id="parent_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Parent section text",
        score=0.80,
        match_type=MatchType.PARENT_CONTEXT.value,
        content_type=ContentType.SECTION.value,
    )
    selected = selector.select_evidence([parent_res, exact_res])
    chunk_ids = [e.chunk_id for e in selected]
    assert "parent_chunk" in chunk_ids
    assert "child_chunk" in chunk_ids


def test_evidence_selection_duplicate_chunks_removed():
    selector = EvidenceSelector(max_evidence=5)
    dup1 = RetrievalResult(
        chunk_id="same_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Duplicate text",
        score=1.0,
        match_type=MatchType.EXACT.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    dup2 = RetrievalResult(
        chunk_id="same_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Duplicate text",
        score=0.70,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    selected = selector.select_evidence([dup1, dup2])
    assert len(selected) == 1
    assert selected[0].chunk_id == "same_chunk"
    assert selected[0].match_type == "exact"


def test_evidence_selection_deterministic_ordering():
    selector = EvidenceSelector(max_evidence=3)
    item_b = RetrievalResult(
        chunk_id="chunk_b",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Text B",
        score=0.85,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    item_a = RetrievalResult(
        chunk_id="chunk_a",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Text A",
        score=0.85,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    # Tie-breaking by chunk_id ASC
    selected = selector.select_evidence([item_b, item_a])
    assert selected[0].chunk_id == "chunk_a"
    assert selected[1].chunk_id == "chunk_b"


# ==============================================================================
# 3. VERDICT LOGIC TESTS (SUPPORTED, UNSUPPORTED, PARTIALLY_CORRECT, NOT_FOUND, WRONG_REFERENCE)
# ==============================================================================

def test_verdict_supported(verification_engine):
    # Verbatim corroboration in exact reference Section 3(d)
    claim = CitationClaim(
        claim_id="claim_supp",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.verdict == Verdict.SUPPORTED
    assert res.reference_status == ReferenceStatus.EXACT_MATCH
    assert res.confidence >= 0.85
    assert "corroborates" in res.reasoning.lower()


def test_verdict_unsupported(verification_engine):
    # Direct contradiction: claim asserts "is patentable", evidence says "is not patentable"
    claim = CitationClaim(
        claim_id="claim_unsupp",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.verdict == Verdict.UNSUPPORTED
    assert res.reference_status == ReferenceStatus.EXACT_MATCH
    assert res.confidence >= 0.85
    assert "contradicts" in res.reasoning.lower()


def test_verdict_partially_correct(verification_engine):
    # Claim contains true discovery condition, but also adds unmentioned criminal sanctions
    claim = CitationClaim(
        claim_id="claim_part",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form is not patentable and results in criminal imprisonment.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.verdict == Verdict.PARTIALLY_CORRECT
    assert res.confidence == 0.70
    assert "part" in res.reasoning.lower()


def test_verdict_not_found(verification_engine):
    # Non-existent Section 9
    claim = CitationClaim(
        claim_id="claim_nf",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Section 9 provisions govern provisional specifications.",
        reference_text="Section 9",
        normalized_reference="9",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.verdict == Verdict.NOT_FOUND
    assert res.reference_status == ReferenceStatus.NOT_FOUND
    assert res.confidence == 0.95


def test_verdict_wrong_reference(verification_engine):
    # Claim asserts facts about Section 3(e) (mere admixture) while citing Section 3(d)
    claim = CitationClaim(
        claim_id="claim_wr",
        document_id="doc_test",
        domain=Domain.LEGAL,
        claim_text="Section 3(e) excludes substances obtained by mere admixture from patentability.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res = verification_engine.verify(claim)
    assert res.verdict == Verdict.WRONG_REFERENCE
    assert res.confidence == 0.85
    assert "Section 3(e)" in res.reasoning or "mismatch" in res.reasoning.lower()


# ==============================================================================
# 4. PROVENANCE PRESERVATION TESTS
# ==============================================================================

def test_provenance_preservation_in_selected_evidence(verification_engine):
    claim = CitationClaim(
        claim_id="claim_prov",
        document_id="patents_act",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
        source_chunk_id="patents_sec_3_d",
        page=14,
        section="Section 3(d)",
        section_id="sec_3_d",
    )
    res = verification_engine.verify(claim)

    # Top-level result provenance
    assert res.provenance["claim_id"] == "claim_prov"
    assert res.provenance["source_chunk_id"] == "patents_sec_3_d"
    assert res.provenance["document_id"] == "patents_act"
    assert res.provenance["page"] == 14
    assert res.provenance["section"] == "Section 3(d)"
    assert res.provenance["section_id"] == "sec_3_d"

    # Evidence item provenance
    for ev in res.evidence:
        assert ev.chunk_id is not None
        assert ev.document_id is not None
        assert ev.match_type in ("exact", "semantic", "parent_context")
        assert ev.score >= 0.0
        assert "section" in ev.provenance


# ==============================================================================
# 5. DETERMINISM TESTS
# ==============================================================================

def test_determinism_twice_run(verification_engine):
    claim = CitationClaim(
        claim_id="claim_det",
        document_id="doc_det",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    res1 = verification_engine.verify(claim)
    res2 = verification_engine.verify(claim)

    assert res1.verification_id == res2.verification_id
    assert res1.verdict == res2.verdict
    assert res1.reference_status == res2.reference_status
    assert res1.confidence == res2.confidence
    assert res1.reasoning == res2.reasoning
    assert [e.chunk_id for e in res1.evidence] == [e.chunk_id for e in res2.evidence]
    assert [e.score for e in res1.evidence] == [e.score for e in res2.evidence]


# ==============================================================================
# 6. LLM VERIFIER INTERFACE & VALIDATION TESTS
# ==============================================================================

def test_llm_claim_verifier_payload_format():
    verifier = LLMClaimVerifier()
    claim = CitationClaim(
        claim_id="claim_llm",
        document_id="doc_llm",
        domain=Domain.LEGAL,
        claim_text="A known substance form is not patentable.",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    evidence = [
        EvidenceItem(
            chunk_id="chk_1",
            document_id="doc_llm",
            text="Evidence text sample",
            match_type="exact",
            score=1.0,
            provenance={"page": 1},
        )
    ]
    payload = verifier.format_payload(claim, ReferenceStatus.EXACT_MATCH, evidence)
    assert payload.claim_text == claim.claim_text
    assert payload.cited_reference == "3(d)"
    assert payload.reference_status == ReferenceStatus.EXACT_MATCH
    assert len(payload.evidence_items) == 1
    assert payload.evidence_items[0]["chunk_id"] == "chk_1"


def test_llm_claim_verifier_structured_output_validation():
    # Valid output
    valid_output = LLMStructuredOutput(
        verdict=Verdict.SUPPORTED,
        confidence=0.88,
        reasoning="Corroborated by Section 3(d).",
        evidence_chunk_ids_used=["chk_1"],
    )
    assert valid_output.verdict == Verdict.SUPPORTED

    # Invalid confidence bounds (> 1.0)
    with pytest.raises(ValidationError):
        LLMStructuredOutput(
            verdict=Verdict.SUPPORTED,
            confidence=1.5,
            reasoning="Invalid confidence test",
        )


def test_llm_claim_verifier_mock_callable():
    def mock_callable(payload: LLMVerificationPayload) -> LLMStructuredOutput:
        return LLMStructuredOutput(
            verdict=Verdict.SUPPORTED,
            confidence=0.92,
            reasoning=f"LLM confirmed support for {payload.cited_reference}",
            evidence_chunk_ids_used=["chk_1"],
        )

    verifier = LLMClaimVerifier(llm_callable=mock_callable, model_name="test-gpt")
    claim = CitationClaim(
        claim_id="c1",
        document_id="d1",
        domain=Domain.LEGAL,
        claim_text="Test claim",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, [])
    assert decision.verdict == Verdict.SUPPORTED
    assert decision.confidence == 0.92
    assert "LLM confirmed" in decision.reasoning
    assert "test-gpt" in decision.verifier_version


def test_llm_claim_verifier_dry_run_fallback():
    verifier = LLMClaimVerifier(llm_callable=None)
    claim = CitationClaim(
        claim_id="c2",
        document_id="d2",
        domain=Domain.LEGAL,
        claim_text="Section 9 is missing",
        reference_text="Section 9",
        normalized_reference="9",
        reference_type="legal_section",
    )
    # Dry run fallback should not raise an error or attempt network calls
    decision = verifier.verify(claim, ReferenceStatus.NOT_FOUND, [])
    assert decision.verdict == Verdict.NOT_FOUND
    assert decision.confidence == 0.95
    assert "[LLM Dry-Run / Interface Mode]" in decision.reasoning


# ==============================================================================
# 7. STAGE 9.1 REGRESSION TESTS (EVIDENCE QUALITY & CLAIM OBJECT PRESERVATION)
# ==============================================================================

def test_regression_legal_object_preservation():
    """Verify that when a legal reference is an essential grammatical object, it is preserved in claim_text."""
    from src.extraction.extractor import ClaimCitationExtractor
    extractor = ClaimCitationExtractor()
    text = "In Novartis AG v. Union of India, the Supreme Court upheld Section 3(d)."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.LEGAL,
        document_id="legal_case",
    )
    assert len(claims) == 1
    claim = claims[0]
    assert "Section 3(d)" in claim.claim_text
    assert claim.claim_text == "In Novartis AG v. Union of India, the Supreme Court upheld Section 3(d)"
    assert claim.reference_text == "Section 3(d)"
    assert claim.normalized_reference == "3(d)"


def test_regression_annotation_citation_removal():
    """Verify that bracketed annotation citations are cleanly excised without losing claim content."""
    from src.extraction.extractor import ClaimCitationExtractor
    extractor = ClaimCitationExtractor()
    text = "Recent studies demonstrate that hallucinated citations compromise scientific integrity [12]."
    claims = extractor.extract_from_text(
        text=text,
        domain=Domain.SCIENTIFIC,
        document_id="sci_paper",
    )
    assert len(claims) == 1
    claim = claims[0]
    assert claim.reference_text == "[12]"
    assert claim.normalized_reference == "12"
    assert "[12]" not in claim.claim_text
    assert claim.claim_text == "Recent studies demonstrate that hallucinated citations compromise scientific integrity"


def test_regression_structural_heading_cannot_produce_supported_verdict():
    """Verify that a structural heading alone cannot produce a 'supported' verdict."""
    verifier = DeterministicClaimVerifier()
    claim = CitationClaim(
        claim_id="claim_struct_test",
        document_id="doc_struct",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is not patentable",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )
    # Evidence set containing ONLY the structural heading
    structural_evidence = [
        EvidenceItem(
            chunk_id="legal_sec_3_d_heading",
            document_id="doc_struct",
            text="1. Section 3(d) Provisions",
            match_type="exact",
            score=1.0,
            is_structural=True,
            evidence_type="structural",
            provenance={"content_type": "heading", "section": "Section 3(d)"},
        )
    ]

    decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, structural_evidence)
    assert decision.verdict != Verdict.SUPPORTED
    assert decision.verdict == Verdict.UNSUPPORTED
    assert "structural" in decision.reasoning.lower()


def test_regression_substantive_evidence_supports_legal_claim():
    """Verify that substantive paragraph evidence is preferred and supports the legal claim."""
    verifier = DeterministicClaimVerifier()
    selector = EvidenceSelector(max_evidence=5)

    claim = CitationClaim(
        claim_id="claim_substantive_test",
        document_id="doc_legal",
        domain=Domain.LEGAL,
        claim_text="The mere discovery of a new form of a known substance is not patentable",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )

    candidates = [
        RetrievalResult(
            chunk_id="sec_3_d_heading",
            document_id="doc_legal",
            domain=Domain.LEGAL,
            text="1. Section 3(d) Provisions",
            content_type=ContentType.HEADING,
            match_type="exact",
            score=1.0,
            rank=1,
            section="Section 3(d)",
        ),
        RetrievalResult(
            chunk_id="sec_3_d_chunk_001",
            document_id="doc_legal",
            domain=Domain.LEGAL,
            text="The mere discovery of a new form of a known substance is not patentable.",
            content_type=ContentType.PARAGRAPH,
            match_type="exact",
            score=1.0,
            rank=2,
            section="Section 3(d)",
        ),
    ]

    selected = selector.select_evidence(candidates, max_evidence=5)
    # Verify substantive paragraph is prioritized over structural heading
    assert selected[0].chunk_id == "sec_3_d_chunk_001"
    assert not selected[0].is_structural
    assert selected[1].chunk_id == "sec_3_d_heading"
    assert selected[1].is_structural

    # Verify verdict against selected evidence
    decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, selected)
    assert decision.verdict == Verdict.SUPPORTED
    assert decision.confidence == 0.90


def test_regression_missing_reference_negative_test(verification_engine):
    """Verify that Section 9 negative test still produces reference_status=not_found and verdict=not_found."""
    claim = CitationClaim(
        claim_id="claim_sec9_test",
        document_id="legal_case",
        domain=Domain.LEGAL,
        claim_text="Provisions of Section 9 govern provisional specifications.",
        reference_text="Section 9",
        normalized_reference="9",
        reference_type="legal_section",
    )
    result = verification_engine.verify(claim)
    assert result.reference_status == ReferenceStatus.NOT_FOUND
    assert result.verdict == Verdict.NOT_FOUND
    assert result.confidence == 0.95
    assert "not found" in result.reasoning.lower()


def test_regression_determinism(verification_engine):
    """Verify that repeated verification produces strictly identical claim_text, reference_status, verdict, ordering, reasoning."""
    claim = CitationClaim(
        claim_id="claim_det_test",
        document_id="legal_sample_fixture",
        domain=Domain.LEGAL,
        claim_text="In Novartis AG v. Union of India, the Supreme Court upheld Section 3(d)",
        reference_text="Section 3(d)",
        normalized_reference="3(d)",
        reference_type="legal_section",
    )

    run_1 = verification_engine.verify(claim)
    run_2 = verification_engine.verify(claim)

    assert run_1.claim_text == run_2.claim_text
    assert run_1.reference_status == run_2.reference_status
    assert run_1.verdict == run_2.verdict
    assert run_1.confidence == run_2.confidence
    assert [e.chunk_id for e in run_1.evidence] == [e.chunk_id for e in run_2.evidence]
    assert [e.is_structural for e in run_1.evidence] == [e.is_structural for e in run_2.evidence]
    assert run_1.reasoning == run_2.reasoning
    assert run_1.verification_id == run_2.verification_id
