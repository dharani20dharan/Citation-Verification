"""Base interface and abstractions for domain-specific text embedders."""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve compute device based on availability and request.

    Args:
        requested_device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning("CUDA requested but not available; falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"

    logger.info("Embedding device selected: %s", device)
    return device


class BaseEmbedder(ABC):
    """Abstract base class for domain-specific text embedders."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a dense vector."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings in batches into dense vectors.

        Preserves exact input ordering and handles empty lists cleanly.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension size."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier or checkpoint path."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Inference device (cuda or cpu)."""
        pass
