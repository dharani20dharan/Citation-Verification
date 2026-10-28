"""Domain-specific embedder for legal documents using BGE-large or GTE-large."""

import logging
from typing import Any, List, Optional
import numpy as np
import torch

from src.config.settings import Settings, get_settings
from src.embeddings.base_embedder import BaseEmbedder, resolve_device

logger = logging.getLogger(__name__)


class LegalEmbedder(BaseEmbedder):
    """Legal text embedder utilizing BAAI's BGE-large (or GTE-large).

    Provides deep representations tailored for statutory provisions, case law,
    clauses, and judicial citations.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        settings: Optional[Settings] = None,
        _model: Optional[Any] = None,
    ):
        """Initialize the legal embedder with lazy model loading.

        Args:
            model_name: Hugging Face model identifier (defaults to settings.LEGAL_EMBEDDING_MODEL).
            batch_size: Batch size for inference (defaults to settings.EMBEDDING_BATCH_SIZE).
            device: Compute device ('auto', 'cuda', 'cpu').
            normalize: Whether to L2-normalize vectors (defaults to settings.EMBEDDING_NORMALIZE).
            settings: Settings singleton.
            _model: Optional pre-loaded model instance (useful for mocking/testing).
        """
        self.settings = settings or get_settings()
        self._model_name = model_name or self.settings.LEGAL_EMBEDDING_MODEL
        self.batch_size = batch_size or self.settings.EMBEDDING_BATCH_SIZE
        self._device_str = resolve_device(device or self.settings.EMBEDDING_DEVICE)
        self.normalize = normalize if normalize is not None else self.settings.EMBEDDING_NORMALIZE

        # Lazy loading state
        self._model = _model
        self._dimension: Optional[int] = None

        if self._model is not None:
            if hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dimension = self._model.get_sentence_embedding_dimension()
            elif hasattr(self._model, "config") and hasattr(self._model.config, "hidden_size"):
                self._dimension = self._model.config.hidden_size

    def _ensure_loaded(self) -> None:
        """Lazily load SentenceTransformer or Hugging Face model onto target device."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading legal embedding model '%s' onto device '%s'",
            self._model_name,
            self._device_str,
        )
        self._model = SentenceTransformer(self._model_name, device=self._device_str)

        if hasattr(self._model, "get_sentence_embedding_dimension"):
            self._dimension = self._model.get_sentence_embedding_dimension()
        else:
            self._dimension = 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        return self._dimension or 1024

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        vectors = self.embed_texts([text])
        if not vectors:
            return []
        return vectors[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of legal texts in batches.

        Preserves exact input order and returns finite numeric vectors.
        """
        if not texts:
            return []

        self._ensure_loaded()

        # Check if model has sentence-transformers encode method
        if hasattr(self._model, "encode"):
            with torch.inference_mode():
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return [list(vec) for vec in embeddings]

        # Fallback for mock/generic PyTorch models
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            with torch.inference_mode():
                raw = self._model(batch_texts)
                if isinstance(raw, torch.Tensor):
                    if self.normalize:
                        raw = torch.nn.functional.normalize(raw, p=2, dim=1)
                    all_embeddings.extend(raw.cpu().tolist())
                elif isinstance(raw, np.ndarray):
                    all_embeddings.extend(raw.tolist())
                else:
                    all_embeddings.extend([list(v) for v in raw])

        return all_embeddings
