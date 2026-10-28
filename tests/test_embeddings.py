"""Unit and integration tests for domain-specific text embeddings."""

import math
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from src.embeddings.base_embedder import BaseEmbedder, resolve_device
from src.embeddings.factory import get_embedder
from src.embeddings.legal_embedder import LegalEmbedder
from src.embeddings.scientific_embedder import ScientificEmbedder
from src.models.schemas import ContentType, Domain, EvidenceChunk


class MockSPECTER2Model:
    """Mock PyTorch model mimicking SPECTER2 output structure."""

    def __init__(self, hidden_size: int = 768):
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.load_count = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        input_ids = inputs.get("input_ids")
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10

        # Create reproducible non-normalized hidden states
        base_tensor = torch.arange(
            batch_size * seq_len * self.hidden_size,
            dtype=torch.float32,
        ).reshape(batch_size, seq_len, self.hidden_size) + 1.0

        result = MagicMock()
        result.last_hidden_state = base_tensor
        return result


class MockTokenizer:
    """Mock Hugging Face tokenizer."""

    def __call__(self, texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones((batch_size, 16), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 16), dtype=torch.long),
        }


class MockSentenceTransformer:
    """Mock SentenceTransformer model."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.load_count = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **kwargs):
        n = len(texts)
        # Deterministic vectors based on string hash
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 1000 + 1
            arr = np.full((self.dimension,), float(seed), dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            vecs.append(arr)
        return np.array(vecs)


@pytest.fixture
def mock_scientific_embedder():
    mock_model = MockSPECTER2Model(hidden_size=768)
    mock_tokenizer = MockTokenizer()
    return ScientificEmbedder(
        model_name="allenai/specter2_base",
        batch_size=2,
        device="cpu",
        normalize=True,
        _model=mock_model,
        _tokenizer=mock_tokenizer,
    )


@pytest.fixture
def mock_legal_embedder():
    mock_st = MockSentenceTransformer(dimension=1024)
    return LegalEmbedder(
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=2,
        device="cpu",
        normalize=True,
        _model=mock_st,
    )


# 1. Embedder Interface
def test_embedder_interface(mock_scientific_embedder, mock_legal_embedder):
    assert isinstance(mock_scientific_embedder, BaseEmbedder)
    assert isinstance(mock_legal_embedder, BaseEmbedder)

    for embedder in [mock_scientific_embedder, mock_legal_embedder]:
        assert hasattr(embedder, "embed_text")
        assert hasattr(embedder, "embed_texts")
        assert hasattr(embedder, "dimension")
        assert hasattr(embedder, "model_name")
        assert hasattr(embedder, "device")


# 2. Scientific Model Configuration
def test_scientific_model_configuration():
    embedder = ScientificEmbedder(
        model_name="custom/specter-model",
        batch_size=16,
        device="cpu",
        normalize=False,
    )
    assert embedder.model_name == "custom/specter-model"
    assert embedder.batch_size == 16
    assert embedder.device == "cpu"
    assert embedder.normalize is False


# 3. Legal Model Configuration
def test_legal_model_configuration():
    embedder = LegalEmbedder(
        model_name="thenlper/gte-large",
        batch_size=8,
        device="cpu",
        normalize=True,
    )
    assert embedder.model_name == "thenlper/gte-large"
    assert embedder.batch_size == 8
    assert embedder.device == "cpu"
    assert embedder.normalize is True


# 4. Empty Input Handling
def test_empty_input(mock_scientific_embedder, mock_legal_embedder):
    assert mock_scientific_embedder.embed_texts([]) == []
    assert mock_legal_embedder.embed_texts([]) == []


# 5. Batch Input
def test_batch_input(mock_scientific_embedder, mock_legal_embedder):
    texts = [f"Text chunk number {i}" for i in range(5)]
    sci_vecs = mock_scientific_embedder.embed_texts(texts)
    legal_vecs = mock_legal_embedder.embed_texts(texts)

    assert len(sci_vecs) == 5
    assert len(legal_vecs) == 5


# 6. Output Count
def test_output_count(mock_scientific_embedder):
    texts = ["Alpha", "Beta", "Gamma"]
    vecs = mock_scientific_embedder.embed_texts(texts)
    assert len(vecs) == len(texts)


# 7. Vector Dimensions
def test_vector_dimensions(mock_scientific_embedder, mock_legal_embedder):
    sci_vec = mock_scientific_embedder.embed_text("Scientific text")
    assert len(sci_vec) == 768
    assert mock_scientific_embedder.dimension == 768

    legal_vec = mock_legal_embedder.embed_text("Section 3(d) provision")
    assert len(legal_vec) == 1024
    assert mock_legal_embedder.dimension == 1024


# 8. Deterministic Ordering
def test_deterministic_ordering(mock_legal_embedder):
    texts = ["First statement", "Second statement", "Third statement"]
    batch_vecs = mock_legal_embedder.embed_texts(texts)

    single_0 = mock_legal_embedder.embed_text("First statement")
    single_1 = mock_legal_embedder.embed_text("Second statement")
    single_2 = mock_legal_embedder.embed_text("Third statement")

    assert batch_vecs[0] == single_0
    assert batch_vecs[1] == single_1
    assert batch_vecs[2] == single_2


# 9. Finite Vector Values
def test_finite_vector_values(mock_scientific_embedder, mock_legal_embedder):
    for emb in [mock_scientific_embedder, mock_legal_embedder]:
        vec = emb.embed_text("Check numeric stability")
        for val in vec:
            assert not math.isnan(val)
            assert not math.isinf(val)


# 10. Normalization Behavior
def test_normalization_behavior():
    # Scientific with normalization
    mock_model = MockSPECTER2Model(hidden_size=768)
    norm_emb = ScientificEmbedder(
        normalize=True,
        _model=mock_model,
        _tokenizer=MockTokenizer(),
    )
    vec_norm = norm_emb.embed_text("Normalized text")
    norm_val = np.linalg.norm(vec_norm)
    assert pytest.approx(norm_val, abs=1e-4) == 1.0

    # Scientific without normalization
    unnorm_emb = ScientificEmbedder(
        normalize=False,
        _model=mock_model,
        _tokenizer=MockTokenizer(),
    )
    vec_unnorm = unnorm_emb.embed_text("Unnormalized text")
    unnorm_val = np.linalg.norm(vec_unnorm)
    assert unnorm_val > 1.0


# 11. Chunk to Vector Mapping
def test_chunk_to_vector_mapping(mock_legal_embedder):
    chunks = [
        EvidenceChunk(
            chunk_id=f"chunk_{i:02d}",
            document_id="doc_1",
            domain=Domain.LEGAL,
            title="Title",
            source="legal/doc.pdf",
            page=1,
            section="Section 3",
            section_id="sec_3",
            parent_id=None,
            content_type=ContentType.PARAGRAPH,
            text=f"Legal text evidence {i}",
        )
        for i in range(3)
    ]
    texts = [c.text for c in chunks]
    vecs = mock_legal_embedder.embed_texts(texts)

    mapping = {c.chunk_id: vec for c, vec in zip(chunks, vecs)}
    assert len(mapping) == 3
    assert "chunk_00" in mapping
    assert len(mapping["chunk_00"]) == 1024


# 12. Invalid Chunk Rejection in build_embeddings logic
def test_invalid_chunk_rejection(tmp_path):
    from scripts.build_embeddings import process_chunk_file

    invalid_jsonl = tmp_path / "corrupted_chunk.jsonl"
    invalid_jsonl.write_text("CORRUPTED_NOT_JSON\n", encoding="utf-8")
    out_file = tmp_path / "out.jsonl"

    result = process_chunk_file(invalid_jsonl, Domain.LEGAL, out_file)
    assert result == 0
    assert not out_file.exists()


# 13. CPU Fallback
def test_cpu_fallback():
    dev_cpu = resolve_device("cpu")
    assert dev_cpu == "cpu"

    # When requesting cuda on a non-cuda machine, falls back to cpu
    with patch("torch.cuda.is_available", return_value=False):
        dev_auto = resolve_device("auto")
        assert dev_auto == "cpu"
        dev_cuda = resolve_device("cuda")
        assert dev_cuda == "cpu"


# 14. Model Caching / No Repeated Loading
def test_model_caching():
    embedder = ScientificEmbedder(device="cpu")
    # Verify model is None initially (lazy loading)
    assert embedder._model is None
    assert embedder._tokenizer is None

    # Inject mock
    embedder._model = MockSPECTER2Model()
    embedder._tokenizer = MockTokenizer()

    # Calling embed_text does not replace or reload _model
    first_model = embedder._model
    embedder.embed_text("test")
    assert embedder._model is first_model


# Factory function test
def test_embedder_factory():
    sci = get_embedder(Domain.SCIENTIFIC, _model=MockSPECTER2Model(), _tokenizer=MockTokenizer())
    assert isinstance(sci, ScientificEmbedder)

    leg = get_embedder(Domain.LEGAL, _model=MockSentenceTransformer())
    assert isinstance(leg, LegalEmbedder)
