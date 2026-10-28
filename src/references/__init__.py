"""Stage 10A: Bibliography and Reference Resolution package."""

from src.references.models import (
    BibliographicEntry,
    ReferenceResolutionResult,
    ResolutionStatus,
    generate_entry_id,
    generate_resolution_id,
)
from src.references.reference_detector import ReferenceSectionDetector
from src.references.reference_normalizer import NormalizedReference, ReferenceNormalizer
from src.references.reference_parser import BibliographicParser
from src.references.reference_resolver import ReferenceResolver
from src.references.resolution_engine import ReferenceResolutionEngine

__all__ = [
    "BibliographicEntry",
    "ReferenceResolutionResult",
    "ResolutionStatus",
    "generate_entry_id",
    "generate_resolution_id",
    "ReferenceSectionDetector",
    "NormalizedReference",
    "ReferenceNormalizer",
    "BibliographicParser",
    "ReferenceResolver",
    "ReferenceResolutionEngine",
]
