"""Domain-specific embedder for scientific literature using SPECTER2."""

import logging
from typing import Any, List, Optional
import torch
import torch.nn.functional as F

from src.config.settings import Settings, get_settings
from src.embeddings.base_embedder import BaseEmbedder, resolve_device

logger = logging.getLogger(__name__)


class ScientificEmbedder(BaseEmbedder):
    """Scientific text embedder utilizing AllenAI's SPECTER2 architecture.

    SPECTER2 extracts dense representations tailored for scientific citation
    and document similarity using CLS token pooling as specified in AllenAI documentation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        settings: Optional[Settings] = None,
        _model: Optional[Any] = None,
        _tokenizer: Optional[Any] = None,
    ):
        """Initialize the scientific embedder with lazy model loading.

        Args:
            model_name: Hugging Face model identifier (defaults to settings.SCIENTIFIC_EMBEDDING_MODEL).
            batch_size: Batch size for inference (defaults to settings.EMBEDDING_BATCH_SIZE).
            device: Compute device ('auto', 'cuda', 'cpu').
            normalize: Whether to L2-normalize vectors (defaults to settings.EMBEDDING_NORMALIZE).
            settings: Settings singleton.
            _model: Optional pre-loaded model instance (useful for mocking/testing).
            _tokenizer: Optional pre-loaded tokenizer instance (useful for mocking/testing).
        """
        self.settings = settings or get_settings()
        self._model_name = model_name or self.settings.SCIENTIFIC_EMBEDDING_MODEL
        self.batch_size = batch_size or self.settings.EMBEDDING_BATCH_SIZE
        self._device_str = resolve_device(device or self.settings.EMBEDDING_DEVICE)
        self.normalize = normalize if normalize is not None else self.settings.EMBEDDING_NORMALIZE

        # Lazy loading state
        self._model = _model
        self._tokenizer = _tokenizer
        self._dimension: Optional[int] = None

        if self._model is not None and hasattr(self._model, "config") and hasattr(self._model.config, "hidden_size"):
            self._dimension = self._model.config.hidden_size

    def _ensure_loaded(self) -> None:
        """Lazily load the model and tokenizer onto the target device if not already loaded."""
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(
            "Loading scientific embedding model '%s' onto device '%s'",
            self._model_name,
            self._device_str,
        )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device_str)
            self._model.eval()

        if hasattr(self._model, "config") and hasattr(self._model.config, "hidden_size"):
            self._dimension = self._model.config.hidden_size
        else:
            self._dimension = 768

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
        return self._dimension or 768

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        vectors = self.embed_texts([text])
        if not vectors:
            return []
        return vectors[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches using SPECTER2 CLS token pooling.

        Preserves exact input order and produces finite numeric vectors.
        """
        if not texts:
            return []

        self._ensure_loaded()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.settings.SCIENTIFIC_MAX_CHUNK_TOKENS,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)
                # SPECTER2 documented pooling: extract the [CLS] token embedding (index 0)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                if self.normalize:
                    cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

                batch_vectors = cls_embeddings.cpu().tolist()
                all_embeddings.extend(batch_vectors)

        return all_embeddings
