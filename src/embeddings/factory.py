"""Factory for domain-specific text embedders."""

import logging
from typing import Optional, Union

from src.config.settings import Settings, get_settings
from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.legal_embedder import LegalEmbedder
from src.embeddings.scientific_embedder import ScientificEmbedder
from src.models.schemas import Domain

logger = logging.getLogger(__name__)


def get_embedder(
    domain: Union[Domain, str],
    settings: Optional[Settings] = None,
    **kwargs,
) -> BaseEmbedder:
    """Instantiate the appropriate embedder for the given domain.

    Args:
        domain: Domain enum or string ('legal' or 'scientific').
        settings: Settings singleton.
        **kwargs: Optional overrides (model_name, batch_size, device, normalize).

    Returns:
        BaseEmbedder subclass instance (LegalEmbedder or ScientificEmbedder).
    """
    domain_enum = Domain(domain) if isinstance(domain, str) else domain
    active_settings = settings or get_settings()

    if domain_enum == Domain.SCIENTIFIC:
        logger.info("Initializing ScientificEmbedder (model: %s)", kwargs.get("model_name", active_settings.SCIENTIFIC_EMBEDDING_MODEL))
        return ScientificEmbedder(settings=active_settings, **kwargs)
    elif domain_enum == Domain.LEGAL:
        logger.info("Initializing LegalEmbedder (model: %s)", kwargs.get("model_name", active_settings.LEGAL_EMBEDDING_MODEL))
        return LegalEmbedder(settings=active_settings, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding domain: {domain_enum}")
