"""Exact-match identifier lookup against ChromaDB metadata."""

import logging
from typing import Any, Dict, List, Optional, Set
from chromadb.api.models.Collection import Collection

from src.models.schemas import ContentType, Domain
from src.retrieval.models import MatchType, RetrievalResult

logger = logging.getLogger(__name__)

# Searchable metadata keys for exact identification
METADATA_IDENTIFIER_KEYS = [
    "clause_id",
    "section",
    "section_id",
    "subsection_id",
    "paragraph_id",
    "table_id",
    "figure_id",
]


class ExactMatcher:
    """Performs exact identifier lookups against stored ChromaDB metadata."""

    def match(
        self,
        collection: Collection,
        identifier: str,
        domain: Domain,
    ) -> List[RetrievalResult]:
        """Lookup evidence chunks exactly matching the structural identifier.

        Critical rule: Does NOT use substring matching (e.g. '3(d)' never matches '3(d)(i)').
        Preserves complete legal distinction between 'Section 3(d)', 'Section 3(e)',
        '3(d)(i)', and '3(d)(ii)'.

        Args:
            collection: ChromaDB persistent collection.
            identifier: User-specified identifier (e.g. 'Section 3(d)', '3(d)', '3(d)(i)').
            domain: Domain enum.

        Returns:
            List of matching RetrievalResult candidates with score=1.0.
        """
        clean_id = identifier.strip()
        if not clean_id:
            return []

        # Candidate exact terms to check
        target_terms: Set[str] = {clean_id}
        if clean_id.lower().startswith("section "):
            # e.g. "Section 3(d)" -> also check "3(d)"
            trimmed = clean_id[8:].strip()
            if trimmed:
                target_terms.add(trimmed)
        elif "(" in clean_id and not clean_id.lower().startswith("section "):
            # e.g. "3(d)" -> also check "Section 3(d)"
            target_terms.add(f"Section {clean_id}")

        matched_records: Dict[str, Dict[str, Any]] = {}

        # 1. Check direct chunk_id lookup
        for term in target_terms:
            try:
                res = collection.get(ids=[term], include=["documents", "metadatas"])
                if res and res.get("ids"):
                    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                        matched_records[cid] = {"id": cid, "document": doc, "metadata": meta}
            except Exception as exc:
                logger.debug("Chroma ID lookup failed for '%s': %s", term, exc)

        # 2. Check metadata exact matches
        for field in METADATA_IDENTIFIER_KEYS:
            for term in target_terms:
                try:
                    res = collection.get(
                        where={field: term},
                        include=["documents", "metadatas"],
                    )
                    if res and res.get("ids"):
                        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                            # Double check exact string equality: stored == term
                            val = str(meta.get(field, "")).strip()
                            if val == term:
                                matched_records[cid] = {
                                    "id": cid,
                                    "document": doc,
                                    "metadata": meta,
                                }
                except Exception as exc:
                    logger.debug("Chroma metadata query failed on %s=%s: %s", field, term, exc)

        results: List[RetrievalResult] = []
        for cid, item in matched_records.items():
            meta = item["metadata"]
            res = RetrievalResult(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                domain=domain,
                text=item["document"],
                score=1.0,
                match_type=MatchType.EXACT.value,
                distance=0.0,
                similarity=1.0,
                title=meta.get("title"),
                source=meta.get("source"),
                page=meta.get("page"),
                section=meta.get("section"),
                section_id=meta.get("section_id"),
                parent_id=meta.get("parent_id"),
                content_type=meta.get("content_type", ContentType.PARAGRAPH.value),
                clause_id=meta.get("clause_id"),
                subsection_id=meta.get("subsection_id"),
                paragraph_id=meta.get("paragraph_id"),
                case_name=meta.get("case_name"),
                statute_name=meta.get("statute_name"),
                table_id=meta.get("table_id"),
                figure_id=meta.get("figure_id"),
                rank=1,
            )
            results.append(res)

        logger.info(
            "Exact matcher found %d candidates for identifier '%s' in %s",
            len(results),
            identifier,
            collection.name,
        )
        return results
