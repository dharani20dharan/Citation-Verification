"""Unit and integration tests for the Retrieval API and engine."""

from typing import List
import pytest
from pydantic import ValidationError

from src.embeddings.base_embedder import BaseEmbedder
from src.models.schemas import ContentType, Domain, EvidenceChunk
from src.retrieval.exact_match import ExactMatcher
from src.retrieval.models import MatchType, RetrievalRequest, RetrievalResult
from src.retrieval.ranker import DeterministicRanker
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.chroma_store import ChromaVectorStore


import hashlib


class MockEmbedder(BaseEmbedder):
    """Mock embedder providing deterministic vectors for testing without loading HF models."""

    def __init__(self, dimension: int = 1024, model_name: str = "mock-embedder"):
        self._dim = dimension
        self._name = model_name

    def embed_text(self, text: str) -> List[float]:
        # Generate deterministic, distinct pseudo-random vector based on text hash
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
    return MockEmbedder(dimension=1024, model_name="BAAI/bge-large-en-v1.5")


@pytest.fixture(scope="module")
def mock_scientific_embedder():
    return MockEmbedder(dimension=768, model_name="allenai/specter2_base")


@pytest.fixture(scope="module")
def populated_vector_store(tmp_path_factory, mock_legal_embedder, mock_scientific_embedder):
    """Populate an isolated ChromaVectorStore with nested legal and scientific hierarchies."""
    chroma_dir = tmp_path_factory.mktemp("test_chroma_retrieval")
    store = ChromaVectorStore(persist_dir=chroma_dir)

    # 1. Legal corpus hierarchy
    # Section 3 (Root) -> Clause 3(d) & Clause 3(e) -> Subclauses 3(d)(i) & 3(d)(ii)
    sec_3_root = EvidenceChunk(
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
        text="the mere discovery of a new form of a known substance...",
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
        text="salts, esters, ethers, polymorphs, metabolites...",
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
        text="combinations and other derivatives of known substance...",
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
        text="a substance obtained by a mere admixture resulting only in...",
        clause_id="3(e)",
    )

    legal_chunks = [sec_3_root, clause_3d, subclause_3di, subclause_3dii, clause_3e]
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

    # 2. Scientific corpus hierarchy
    sec_intro = EvidenceChunk(
        chunk_id="paper_sec_01",
        document_id="paper_01",
        domain=Domain.SCIENTIFIC,
        title="Automated Citation Verification",
        source="scientific/paper.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01",
        parent_id=None,
        content_type=ContentType.SECTION,
        text="1. Introduction to verifiable scientific citations.",
    )
    child_intro = EvidenceChunk(
        chunk_id="paper_sec_01_chunk_001",
        document_id="paper_01",
        domain=Domain.SCIENTIFIC,
        title="Automated Citation Verification",
        source="scientific/paper.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01",
        parent_id="paper_sec_01",
        content_type=ContentType.PARAGRAPH,
        text="We propose an offline evidence corpus indexing pipeline.",
    )

    sci_chunks = [sec_intro, child_intro]
    sci_records = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "domain": "scientific",
            "model_name": mock_scientific_embedder.model_name,
            "dimension": mock_scientific_embedder.dimension,
            "embedding": mock_scientific_embedder.embed_text(c.text),
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
def legal_retriever(populated_vector_store, mock_legal_embedder):
    return Retriever(
        vector_store=populated_vector_store,
        embedder_override=mock_legal_embedder,
    )


@pytest.fixture
def scientific_retriever(populated_vector_store, mock_scientific_embedder):
    return Retriever(
        vector_store=populated_vector_store,
        embedder_override=mock_scientific_embedder,
    )


# ==============================================================================
# 1-5. EXACT MATCHING TESTS
# ==============================================================================

def test_exact_3d_lookup(legal_retriever):
    req = RetrievalRequest(
        query="mere discovery of new form",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    assert any(r.clause_id == "3(d)" and r.match_type == MatchType.EXACT.value for r in results)


def test_exact_3e_lookup(legal_retriever):
    req = RetrievalRequest(
        query="mere admixture",
        domain=Domain.LEGAL,
        exact_identifier="3(e)",
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    exact_matches = [r for r in results if r.match_type == MatchType.EXACT.value]
    assert len(exact_matches) == 1
    assert exact_matches[0].clause_id == "3(e)"
    assert exact_matches[0].chunk_id == "patents_sec_3_e"


def test_exact_3d_does_not_match_3e(legal_retriever):
    req = RetrievalRequest(
        query="admixture",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    exact_matches = [r for r in results if r.match_type == MatchType.EXACT.value]
    # None of the exact matches should be 3(e)
    for em in exact_matches:
        assert em.clause_id != "3(e)"
        assert em.chunk_id != "patents_sec_3_e"


def test_exact_3d_does_not_match_3di(legal_retriever):
    req = RetrievalRequest(
        query="substances",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    exact_matches = [r for r in results if r.match_type == MatchType.EXACT.value]
    # No substring matching: '3(d)' must NOT exactly match '3(d)(i)'
    for em in exact_matches:
        assert em.clause_id != "3(d)(i)"


def test_exact_3di_differs_from_3dii(legal_retriever):
    req_i = RetrievalRequest(
        query="salts esters",
        domain=Domain.LEGAL,
        exact_identifier="3(d)(i)",
        expand_parent_context=False,
    )
    res_i = legal_retriever.retrieve(req_i)
    exact_i = [r for r in res_i if r.match_type == MatchType.EXACT.value]
    assert len(exact_i) == 1
    assert exact_i[0].clause_id == "3(d)(i)"
    assert exact_i[0].chunk_id == "patents_sec_3_d_i"

    req_ii = RetrievalRequest(
        query="derivatives",
        domain=Domain.LEGAL,
        exact_identifier="3(d)(ii)",
        expand_parent_context=False,
    )
    res_ii = legal_retriever.retrieve(req_ii)
    exact_ii = [r for r in res_ii if r.match_type == MatchType.EXACT.value]
    assert len(exact_ii) == 1
    assert exact_ii[0].clause_id == "3(d)(ii)"
    assert exact_ii[0].chunk_id == "patents_sec_3_d_ii"


# ==============================================================================
# 6-7. DOMAIN ISOLATION TESTS
# ==============================================================================

def test_domain_isolation_legal(legal_retriever):
    req = RetrievalRequest(
        query="offline evidence corpus",
        domain=Domain.LEGAL,
        expand_parent_context=True,
    )
    results = legal_retriever.retrieve(req)
    for r in results:
        assert r.domain == Domain.LEGAL
        assert not r.chunk_id.startswith("paper_")


def test_domain_isolation_scientific(scientific_retriever):
    req = RetrievalRequest(
        query="patent invention section 3",
        domain=Domain.SCIENTIFIC,
        expand_parent_context=True,
    )
    results = scientific_retriever.retrieve(req)
    for r in results:
        assert r.domain == Domain.SCIENTIFIC
        assert not r.chunk_id.startswith("patents_")


# ==============================================================================
# 8-10. SEMANTIC SEARCH TESTS
# ==============================================================================

def test_query_embedding_uses_correct_domain_embedder(legal_retriever, scientific_retriever):
    assert legal_retriever._embedder_override.dimension == 1024
    assert scientific_retriever._embedder_override.dimension == 768


def test_top_k_respected(legal_retriever):
    req = RetrievalRequest(
        query="inventions and efficacy",
        domain=Domain.LEGAL,
        top_k=2,
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    # Without parent context or exact ID, semantic search returns min(top_k, total)
    assert len(results) <= 2


def test_scores_distances_preserved(scientific_retriever):
    req = RetrievalRequest(
        query="citation verification",
        domain=Domain.SCIENTIFIC,
        top_k=2,
        expand_parent_context=False,
    )
    results = scientific_retriever.retrieve(req)
    for r in results:
        if r.match_type == MatchType.SEMANTIC.value:
            assert r.distance is not None
            assert r.similarity is not None
            assert pytest.approx(r.score, abs=1e-5) == r.similarity


# ==============================================================================
# 11-14. PARENT CONTEXT EXPANSION TESTS
# ==============================================================================

def test_child_result_retrieves_parent(scientific_retriever):
    req = RetrievalRequest(
        query="We propose an offline evidence corpus indexing pipeline.",
        domain=Domain.SCIENTIFIC,
        top_k=1,
        expand_parent_context=True,
        max_parent_depth=1,
    )
    results = scientific_retriever.retrieve(req)
    parent_matches = [r for r in results if r.match_type == MatchType.PARENT_CONTEXT.value]
    assert len(parent_matches) >= 1
    assert parent_matches[0].chunk_id == "paper_sec_01"


def test_nested_parent_expansion(legal_retriever):
    # Lookup 3(d)(i) child with max_depth=2
    # Hierarchy: 3(d)(i) -> 3(d) -> Section 3
    req = RetrievalRequest(
        query="salts and esters",
        domain=Domain.LEGAL,
        exact_identifier="3(d)(i)",
        expand_parent_context=True,
        max_parent_depth=2,
        semantic_search_enabled=False,
    )
    results = legal_retriever.retrieve(req)
    retrieved_ids = {r.chunk_id for r in results}

    assert "patents_sec_3_d_i" in retrieved_ids  # Exact match
    assert "patents_sec_3_d" in retrieved_ids    # Parent (depth 1)
    assert "patents_sec_3" in retrieved_ids      # Grandparent (depth 2)


def test_max_depth_respected(legal_retriever):
    # Lookup 3(d)(i) with max_depth=1 (only direct parent, no grandparent)
    req = RetrievalRequest(
        query="salts and esters",
        domain=Domain.LEGAL,
        exact_identifier="3(d)(i)",
        expand_parent_context=True,
        max_parent_depth=1,
        semantic_search_enabled=False,
    )
    results = legal_retriever.retrieve(req)
    retrieved_ids = {r.chunk_id for r in results}

    assert "patents_sec_3_d_i" in retrieved_ids
    assert "patents_sec_3_d" in retrieved_ids
    assert "patents_sec_3" not in retrieved_ids  # Grandparent excluded


def test_parent_expansion_disabled(scientific_retriever):
    req = RetrievalRequest(
        query="verifiable citations",
        domain=Domain.SCIENTIFIC,
        expand_parent_context=False,
    )
    results = scientific_retriever.retrieve(req)
    assert all(r.match_type != MatchType.PARENT_CONTEXT.value for r in results)


# ==============================================================================
# 15-17. RANKING POLICY TESTS
# ==============================================================================

def test_exact_ranks_before_semantic(legal_retriever):
    req = RetrievalRequest(
        query="pharmaceutical patents",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=False,
        semantic_search_enabled=True,
    )
    results = legal_retriever.retrieve(req)
    assert len(results) >= 2
    assert results[0].match_type == MatchType.EXACT.value
    assert results[0].rank == 1


def test_semantic_ranks_before_parent_context(legal_retriever):
    req = RetrievalRequest(
        query="pharmaceutical patents",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=True,
        semantic_search_enabled=True,
    )
    results = legal_retriever.retrieve(req)
    match_types = [r.match_type for r in results]

    # Find first index of each match type
    first_exact = match_types.index(MatchType.EXACT.value) if MatchType.EXACT.value in match_types else -1
    first_semantic = match_types.index(MatchType.SEMANTIC.value) if MatchType.SEMANTIC.value in match_types else -1
    first_parent = match_types.index(MatchType.PARENT_CONTEXT.value) if MatchType.PARENT_CONTEXT.value in match_types else -1

    if first_exact != -1 and first_semantic != -1:
        assert first_exact < first_semantic
    if first_semantic != -1 and first_parent != -1:
        assert first_semantic < first_parent


def test_deterministic_tie_breaking():
    ranker = DeterministicRanker()
    # Two semantic items with identical score
    res_b = RetrievalResult(
        chunk_id="chunk_b",
        document_id="doc_01",
        domain=Domain.LEGAL,
        text="Text B",
        score=0.85,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
        page=2,
    )
    res_a = RetrievalResult(
        chunk_id="chunk_a",
        document_id="doc_01",
        domain=Domain.LEGAL,
        text="Text A",
        score=0.85,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
        page=1,
    )
    ranked = ranker.rank([res_b, res_a])
    # Tie-breaking by page: res_a (page 1) before res_b (page 2)
    assert ranked[0].chunk_id == "chunk_a"
    assert ranked[1].chunk_id == "chunk_b"


# ==============================================================================
# 18-19. DEDUPLICATION TESTS
# ==============================================================================

def test_deduplication_single_appearance(legal_retriever):
    req = RetrievalRequest(
        query="mere discovery of a new form",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=True,
        semantic_search_enabled=True,
    )
    results = legal_retriever.retrieve(req)
    chunk_ids = [r.chunk_id for r in results]
    # Every chunk_id must be strictly unique
    assert len(chunk_ids) == len(set(chunk_ids))


def test_deduplication_strongest_match_type():
    ranker = DeterministicRanker()
    exact_res = RetrievalResult(
        chunk_id="shared_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Shared text",
        score=1.0,
        match_type=MatchType.EXACT.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    semantic_res = RetrievalResult(
        chunk_id="shared_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Shared text",
        score=0.75,
        match_type=MatchType.SEMANTIC.value,
        content_type=ContentType.PARAGRAPH.value,
    )
    parent_res = RetrievalResult(
        chunk_id="shared_chunk",
        document_id="doc_1",
        domain=Domain.LEGAL,
        text="Shared text",
        score=0.50,
        match_type=MatchType.PARENT_CONTEXT.value,
        content_type=ContentType.PARAGRAPH.value,
    )

    ranked = ranker.rank([parent_res, semantic_res, exact_res])
    assert len(ranked) == 1
    assert ranked[0].match_type == MatchType.EXACT.value


# ==============================================================================
# 20. PROVENANCE PRESERVATION TEST
# ==============================================================================

def test_provenance_preservation(legal_retriever):
    req = RetrievalRequest(
        query="discovery",
        domain=Domain.LEGAL,
        exact_identifier="3(d)",
        expand_parent_context=False,
    )
    results = legal_retriever.retrieve(req)
    r3d = next(r for r in results if r.chunk_id == "patents_sec_3_d")

    assert r3d.document_id == "patents_act"
    assert r3d.domain == Domain.LEGAL
    assert r3d.source == "legal/patents.pdf"
    assert r3d.page == 14
    assert r3d.section == "Section 3(d)"
    assert r3d.section_id == "sec_3_d"
    assert r3d.parent_id == "patents_sec_3"
    assert r3d.content_type == "clause"
    assert r3d.clause_id == "3(d)"
    assert len(r3d.text) > 0


# ==============================================================================
# 21-23. REQUEST VALIDATION TESTS
# ==============================================================================

def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="   ", domain=Domain.LEGAL)


def test_invalid_top_k_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="valid query", domain=Domain.LEGAL, top_k=0)


def test_invalid_domain_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="valid query", domain="unsupported_domain")
