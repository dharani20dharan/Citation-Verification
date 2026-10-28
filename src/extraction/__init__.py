"""Citation and claim extraction module."""

from src.extraction.citation_detector import CitationDetector
from src.extraction.claim_extractor import ClaimExtractor
from src.extraction.extractor import ClaimCitationExtractor
from src.extraction.models import CitationClaim, DetectedReference, ReferenceType
from src.extraction.reference_normalizer import ReferenceNormalizer

__all__ = [
    "CitationClaim",
    "CitationDetector",
    "ClaimCitationExtractor",
    "ClaimExtractor",
    "DetectedReference",
    "ReferenceNormalizer",
    "ReferenceType",
]
