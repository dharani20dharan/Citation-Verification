"""Unit and integration tests for ChromaVectorStore persistence and isolation."""

import math
from pathlib import Path
import pytest

from src.models.schemas import ContentType, Domain, EvidenceChunk
from src.vectorstore.chroma_store import ChromaVectorStore, DOMAIN_CONFIG
from src.vectorstore.metadata import serialize_chunk_metadata_for_chroma


@pytest.fixture
def temp_chroma_store(tmp_path):
    """Provide a ChromaVectorStore isolated to a temporary directory."""
    chroma_dir = tmp_path / "test_chroma"
    return ChromaVectorStore(persist_dir=chroma_dir)


@pytest.fixture
def sample_legal_chunk():
    return EvidenceChunk(
        chunk_id="leg_doc_01_sec_3_d_chunk_001",
        document_id="leg_doc_01",
        domain=Domain.LEGAL,
        title="Patents Act Analysis",
        source="legal/patents.pdf",
        page=14,
        section="Section 3(d)",
        section_id="sec_3_d",
        parent_id="leg_doc_01_sec_3_d",
        content_type=ContentType.PARAGRAPH,
        text="The mere discovery of a new form of a known substance...",
        clause_id="3(d)",
    )


@pytest.fixture
def sample_scientific_chunk():
    return EvidenceChunk(
        chunk_id="sci_doc_01_sec_01_chunk_001",
        document_id="sci_doc_01",
        domain=Domain.SCIENTIFIC,
        title="Citation Integrity",
        source="scientific/integrity.pdf",
        page=1,
        section="1. Introduction",
        section_id="sec_01",
        parent_id="sci_doc_01_sec_01",
        content_type=ContentType.PARAGRAPH,
        text="We present our citation verification methodology...",
    )


# 1. Collection Creation
def test_collection_creation(temp_chroma_store):
    legal_col = temp_chroma_store.get_or_create_collection(Domain.LEGAL)
    sci_col = temp_chroma_store.get_or_create_collection(Domain.SCIENTIFIC)

    assert legal_col.name == "legal_corpus"
    assert sci_col.name == "scientific_corpus"
    assert legal_col.metadata["dimension"] == 1024
    assert sci_col.metadata["dimension"] == 768


# 2. Record Count
def test_record_count(temp_chroma_store, sample_legal_chunk, sample_scientific_chunk):
    leg_rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    sci_rec = {
        "chunk_id": sample_scientific_chunk.chunk_id,
        "document_id": sample_scientific_chunk.document_id,
        "domain": "scientific",
        "model_name": "allenai/specter2_base",
        "dimension": 768,
        "embedding": [0.2] * 768,
    }

    rep_leg = temp_chroma_store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=[leg_rec],
        chunk_map={sample_legal_chunk.chunk_id: sample_legal_chunk},
    )
    rep_sci = temp_chroma_store.ingest_records(
        domain=Domain.SCIENTIFIC,
        embedding_records=[sci_rec],
        chunk_map={sample_scientific_chunk.chunk_id: sample_scientific_chunk},
    )

    assert rep_leg.inserted_updated == 1
    assert rep_leg.collection_count == 1
    assert rep_sci.inserted_updated == 1
    assert rep_sci.collection_count == 1


# 3. Metadata Preservation
def test_metadata_preservation(temp_chroma_store, sample_legal_chunk):
    rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.05] * 1024,
    }
    temp_chroma_store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=[rec],
        chunk_map={sample_legal_chunk.chunk_id: sample_legal_chunk},
    )

    col = temp_chroma_store.get_or_create_collection(Domain.LEGAL)
    peek = col.get(ids=[sample_legal_chunk.chunk_id])
    assert len(peek["metadatas"]) == 1
    meta = peek["metadatas"][0]

    assert meta["chunk_id"] == sample_legal_chunk.chunk_id
    assert meta["document_id"] == sample_legal_chunk.document_id
    assert meta["source"] == "legal/patents.pdf"
    assert meta["page"] == 14
    assert meta["section"] == "Section 3(d)"
    assert meta["section_id"] == "sec_3_d"
    assert meta["parent_id"] == "leg_doc_01_sec_3_d"
    assert meta["content_type"] == "paragraph"
    assert meta["clause_id"] == "3(d)"


# 4. Source Text Preservation
def test_source_text_preservation(temp_chroma_store, sample_legal_chunk):
    rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.01] * 1024,
    }
    temp_chroma_store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=[rec],
        chunk_map={sample_legal_chunk.chunk_id: sample_legal_chunk},
    )

    col = temp_chroma_store.get_or_create_collection(Domain.LEGAL)
    peek = col.get(ids=[sample_legal_chunk.chunk_id])
    assert len(peek["documents"]) == 1
    assert peek["documents"][0] == sample_legal_chunk.text


# 5. Dimension Enforcement
def test_dimension_enforcement(temp_chroma_store, sample_legal_chunk):
    # Invalid dimension 512 for legal_corpus (expects 1024)
    bad_rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 512,
        "embedding": [0.1] * 512,
    }
    rep = temp_chroma_store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=[bad_rec],
        chunk_map={sample_legal_chunk.chunk_id: sample_legal_chunk},
    )
    assert rep.passed is False
    assert rep.rejected == 1
    assert rep.inserted_updated == 0
    assert rep.collection_count == 0
    assert any("Dimension mismatch" in err for err in rep.errors)


# 6. Domain Isolation
def test_domain_isolation(temp_chroma_store, sample_legal_chunk):
    # Attempting to ingest legal chunk into scientific_corpus
    rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    rep = temp_chroma_store.ingest_records(
        domain=Domain.SCIENTIFIC,
        embedding_records=[rec],
        chunk_map={sample_legal_chunk.chunk_id: sample_legal_chunk},
    )
    assert rep.passed is False
    assert rep.rejected == 1
    assert any("Domain mismatch" in err for err in rep.errors)
    assert temp_chroma_store.get_or_create_collection(Domain.SCIENTIFIC).count() == 0


# 7. Missing Chunk Mapping
def test_missing_chunk_mapping(temp_chroma_store):
    rec = {
        "chunk_id": "non_existent_chunk_id",
        "document_id": "doc_01",
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    rep = temp_chroma_store.ingest_records(
        domain=Domain.LEGAL,
        embedding_records=[rec],
        chunk_map={},  # Empty chunk map
    )
    assert rep.passed is False
    assert rep.rejected == 1
    assert any("No corresponding EvidenceChunk" in err for err in rep.errors)


# 8. Idempotent Ingestion
def test_idempotent_ingestion(temp_chroma_store, sample_legal_chunk):
    rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    chunk_map = {sample_legal_chunk.chunk_id: sample_legal_chunk}

    rep1 = temp_chroma_store.ingest_records(Domain.LEGAL, [rec], chunk_map)
    assert rep1.collection_count == 1

    rep2 = temp_chroma_store.ingest_records(Domain.LEGAL, [rec], chunk_map)
    assert rep2.collection_count == 1
    assert rep2.inserted_updated == 1


# 9. Update Behavior
def test_update_behavior(temp_chroma_store, sample_legal_chunk):
    rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    chunk_map = {sample_legal_chunk.chunk_id: sample_legal_chunk}
    temp_chroma_store.ingest_records(Domain.LEGAL, [rec], chunk_map)

    # Modify chunk text and re-upsert
    updated_chunk = sample_legal_chunk.model_copy(update={"text": "Updated statutory wording"})
    temp_chroma_store.ingest_records(Domain.LEGAL, [rec], {sample_legal_chunk.chunk_id: updated_chunk})

    col = temp_chroma_store.get_or_create_collection(Domain.LEGAL)
    assert col.count() == 1
    peek = col.get(ids=[sample_legal_chunk.chunk_id])
    assert peek["documents"][0] == "Updated statutory wording"


# 10. Persistence Across Reload
def test_persistence_across_reload(tmp_path, sample_scientific_chunk):
    chroma_dir = tmp_path / "reload_test_chroma"
    store1 = ChromaVectorStore(persist_dir=chroma_dir)

    sci_rec = {
        "chunk_id": sample_scientific_chunk.chunk_id,
        "document_id": sample_scientific_chunk.document_id,
        "domain": "scientific",
        "model_name": "allenai/specter2_base",
        "dimension": 768,
        "embedding": [0.3] * 768,
    }
    store1.ingest_records(
        domain=Domain.SCIENTIFIC,
        embedding_records=[sci_rec],
        chunk_map={sample_scientific_chunk.chunk_id: sample_scientific_chunk},
    )

    # Release store1 and instantiate a brand-new store2 pointing to same path
    del store1
    store2 = ChromaVectorStore(persist_dir=chroma_dir)
    res = store2.verify_persistence(Domain.SCIENTIFIC)

    assert res["reloaded"] is True
    assert res["collection_name"] == "scientific_corpus"
    assert res["count"] == 1
    assert sample_scientific_chunk.chunk_id in res["sample_ids"]


# 11. Malformed Embedding Rejection
def test_malformed_embedding_rejection(temp_chroma_store, sample_legal_chunk):
    # Vector containing NaN
    nan_vec = [0.1] * 1024
    nan_vec[5] = float("nan")

    bad_rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": nan_vec,
    }
    rep = temp_chroma_store.ingest_records(
        Domain.LEGAL,
        [bad_rec],
        {sample_legal_chunk.chunk_id: sample_legal_chunk},
    )
    assert rep.passed is False
    assert rep.rejected == 1
    assert any("Non-finite" in err for err in rep.errors)


# 12. Model Consistency
def test_model_consistency(temp_chroma_store, sample_legal_chunk):
    bad_model_rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "unauthorized/random_model",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    rep = temp_chroma_store.ingest_records(
        Domain.LEGAL,
        [bad_model_rec],
        {sample_legal_chunk.chunk_id: sample_legal_chunk},
    )
    assert rep.passed is False
    assert any("Model inconsistency" in err for err in rep.errors)


# 13. Both-Domain Isolation
def test_both_domain_isolation(temp_chroma_store, sample_legal_chunk, sample_scientific_chunk):
    leg_rec = {
        "chunk_id": sample_legal_chunk.chunk_id,
        "document_id": sample_legal_chunk.document_id,
        "domain": "legal",
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "embedding": [0.1] * 1024,
    }
    sci_rec = {
        "chunk_id": sample_scientific_chunk.chunk_id,
        "document_id": sample_scientific_chunk.document_id,
        "domain": "scientific",
        "model_name": "allenai/specter2_base",
        "dimension": 768,
        "embedding": [0.2] * 768,
    }

    temp_chroma_store.ingest_records(Domain.LEGAL, [leg_rec], {sample_legal_chunk.chunk_id: sample_legal_chunk})
    temp_chroma_store.ingest_records(Domain.SCIENTIFIC, [sci_rec], {sample_scientific_chunk.chunk_id: sample_scientific_chunk})

    leg_col = temp_chroma_store.get_or_create_collection(Domain.LEGAL)
    sci_col = temp_chroma_store.get_or_create_collection(Domain.SCIENTIFIC)

    assert leg_col.count() == 1
    assert sci_col.count() == 1

    leg_items = leg_col.get()
    sci_items = sci_col.get()

    assert leg_items["ids"] == [sample_legal_chunk.chunk_id]
    assert leg_items["metadatas"][0]["domain"] == "legal"

    assert sci_items["ids"] == [sample_scientific_chunk.chunk_id]
    assert sci_items["metadatas"][0]["domain"] == "scientific"
