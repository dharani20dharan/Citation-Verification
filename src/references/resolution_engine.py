"""High-level Reference Resolution Engine coordinating bibliography indexing and reference resolution."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.extraction.models import CitationClaim
from src.models.schemas import Domain
from src.references.models import BibliographicEntry, ReferenceResolutionResult
from src.references.reference_normalizer import ReferenceNormalizer
from src.references.reference_resolver import ReferenceResolver

logger = logging.getLogger(__name__)


class ReferenceResolutionEngine:
    """Orchestrates citation-to-bibliography resolution across corpus documents.

    Flow:
        CitationClaim or reference string
              ↓
        Determine reference type (numeric, range, author-year, legal)
              ↓
        Lookup in document-scoped or global bibliography store
              ↓
        Structured ReferenceResolutionResult
    """

    def __init__(
        self,
        resolver: Optional[ReferenceResolver] = None,
        references_dir: Optional[Path] = None,
    ):
        self.resolver = resolver or ReferenceResolver()
        self.normalizer = ReferenceNormalizer()
        self.references_dir = (
            references_dir
            if references_dir is not None
            else Path("data/references")
        )
        # In-memory index of entries by document_id and global list
        self._entries_by_doc: Dict[str, List[BibliographicEntry]] = {}
        self._all_entries: List[BibliographicEntry] = []

        # Automatically load existing references if directory exists
        self._load_stored_references()

    def _load_stored_references(self) -> None:
        """Scan data/references/ for JSONL bibliography files and populate index."""
        if not self.references_dir.exists():
            return

        for domain_dir in self.references_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for jsonl_path in domain_dir.glob("*.jsonl"):
                try:
                    self.load_from_jsonl(jsonl_path)
                except Exception as e:
                    logger.warning("Failed loading reference file '%s': %s", jsonl_path, e)

    def load_from_jsonl(self, path: Path) -> int:
        """Load BibliographicEntry objects from a JSONL file into the engine index.

        Args:
            path: Path to JSONL file.

        Returns:
            Number of entries successfully loaded.
        """
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = BibliographicEntry.model_validate(data)
                self.add_entry(entry)
                loaded += 1

        logger.info("Loaded %d bibliographic entries from '%s'", loaded, path.name)
        return loaded

    def add_entry(self, entry: BibliographicEntry) -> None:
        """Register a BibliographicEntry in the engine's in-memory index."""
        doc_id = entry.document_id
        if doc_id not in self._entries_by_doc:
            self._entries_by_doc[doc_id] = []
        existing_ids = {e.entry_id for e in self._entries_by_doc[doc_id]}
        if entry.entry_id not in existing_ids:
            self._entries_by_doc[doc_id].append(entry)
            self._all_entries.append(entry)

    def add_entries(self, entries: List[BibliographicEntry]) -> None:
        """Register multiple BibliographicEntry objects."""
        for entry in entries:
            self.add_entry(entry)

    def resolve(self, claim: CitationClaim) -> ReferenceResolutionResult:
        """Resolve a CitationClaim's cited reference against the document's bibliography.

        Args:
            claim: CitationClaim containing reference_text and document provenance.

        Returns:
            ReferenceResolutionResult with status and matched BibliographicEntry objects.
        """
        doc_entries = self._entries_by_doc.get(claim.document_id, [])
        candidate_entries = doc_entries if doc_entries else self._all_entries

        return self.resolver.resolve(
            reference_text=claim.reference_text,
            bibliography=candidate_entries,
            document_id=claim.document_id,
            claim_id=claim.claim_id,
        )

    def resolve_reference(
        self,
        reference_text: str,
        domain: Domain = Domain.SCIENTIFIC,
        document_id: Optional[str] = None,
    ) -> ReferenceResolutionResult:
        """Resolve a raw reference text query for a domain and optional document.

        Args:
            reference_text: Raw reference string (e.g. '[12]', '[3, 4]', '(Smith et al., 2024)').
            domain: Corpus domain.
            document_id: Optional document identifier.

        Returns:
            Structured ReferenceResolutionResult.
        """
        if document_id and document_id in self._entries_by_doc:
            candidates = self._entries_by_doc[document_id]
        else:
            candidates = [e for e in self._all_entries if e.domain == domain]

        return self.resolver.resolve(
            reference_text=reference_text,
            bibliography=candidates,
            document_id=document_id,
        )

    def get_entries(self, document_id: Optional[str] = None) -> List[BibliographicEntry]:
        """Retrieve stored bibliography entries."""
        if document_id:
            return self._entries_by_doc.get(document_id, [])
        return self._all_entries
