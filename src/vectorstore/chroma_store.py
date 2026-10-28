"""Persistent ChromaDB vector store for legal and scientific evidence corpora.

Enforces domain isolation, strict dimension verification, model consistency,
idempotent batched upserts, and persistent on-disk storage.
No retrieval methods are implemented in this stage.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.models.schemas import Domain, EvidenceChunk
from src.vectorstore.metadata import serialize_chunk_metadata_for_chroma

logger = logging.getLogger(__name__)

# Canonical domain configuration
DOMAIN_CONFIG = {
    Domain.LEGAL: {
        "collection_name": "legal_corpus",
        "expected_dimension": 1024,
        "default_model": "BAAI/bge-large-en-v1.5",
    },
    Domain.SCIENTIFIC: {
        "collection_name": "scientific_corpus",
        "expected_dimension": 768,
        "default_model": "allenai/specter2_base",
    },
}


class IngestionReport(BaseModel):
    """Structured report produced following a ChromaDB ingestion run."""

    domain: str
    collection_name: str
    model_name: str
    dimension: int
    input_embeddings: int = 0
    valid_records: int = 0
    inserted_updated: int = 0
    rejected: int = 0
    collection_count: int = 0
    persist_dir: str = ""
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    passed: bool = True

    def add_error(self, message: str) -> None:
        """Record an error and mark report as failed."""
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(message)


class ChromaVectorStore:
    """Manages persistent ChromaDB collections for legal and scientific corpora."""

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize persistent Chroma client.

        Args:
            persist_dir: Directory path for Chroma persistence.
            settings: Project Settings singleton.
        """
        self.settings = settings or get_settings()
        self.persist_dir = (
            Path(persist_dir).resolve()
            if persist_dir is not None
            else self.settings.chroma_dir
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing persistent ChromaDB client at: %s", self.persist_dir)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    @property
    def client(self) -> chromadb.PersistentClient:
        """Active ChromaDB persistent client."""
        return self._client

    def get_collection_name(self, domain: Union[Domain, str]) -> str:
        """Get canonical collection name for a domain."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        return DOMAIN_CONFIG[dom_enum]["collection_name"]

    def get_expected_dimension(self, domain: Union[Domain, str]) -> int:
        """Get expected vector dimension for a domain."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        return DOMAIN_CONFIG[dom_enum]["expected_dimension"]

    def get_expected_model(self, domain: Union[Domain, str]) -> str:
        """Get expected embedding model name for a domain."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        if dom_enum == Domain.LEGAL:
            return self.settings.LEGAL_EMBEDDING_MODEL or DOMAIN_CONFIG[dom_enum]["default_model"]
        return self.settings.SCIENTIFIC_EMBEDDING_MODEL or DOMAIN_CONFIG[dom_enum]["default_model"]

    def get_or_create_collection(self, domain: Union[Domain, str]) -> Collection:
        """Retrieve or create the persistent collection for a domain."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        cfg = DOMAIN_CONFIG[dom_enum]
        col_name = cfg["collection_name"]
        expected_dim = cfg["expected_dimension"]
        expected_model = self.get_expected_model(dom_enum)

        collection = self._client.get_or_create_collection(
            name=col_name,
            metadata={
                "domain": dom_enum.value,
                "dimension": expected_dim,
                "model_name": expected_model,
                "hnsw:space": "cosine",
            },
        )
        return collection

    def reset_collection(self, domain: Union[Domain, str]) -> None:
        """Explicitly delete and recreate a domain collection."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        col_name = self.get_collection_name(dom_enum)
        try:
            self._client.delete_collection(name=col_name)
            logger.info("Deleted existing collection: %s", col_name)
        except Exception:
            logger.info("Collection %s did not exist, skipping deletion", col_name)
        self.get_or_create_collection(dom_enum)

    def validate_embedding_record(
        self,
        record: Dict[str, Any],
        target_domain: Domain,
    ) -> Tuple[bool, Optional[str]]:
        """Validate an embedding record against target domain and constraints."""
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            return False, "Missing chunk_id in embedding record"

        # Domain isolation
        rec_domain = record.get("domain")
        if rec_domain != target_domain.value:
            return (
                False,
                f"[{chunk_id}] Domain mismatch: record domain '{rec_domain}' != target '{target_domain.value}'",
            )

        # Dimension validation
        expected_dim = self.get_expected_dimension(target_domain)
        vec = record.get("embedding")
        if not isinstance(vec, list) or len(vec) == 0:
            return False, f"[{chunk_id}] Embedding vector is missing or empty"

        if len(vec) != expected_dim:
            return (
                False,
                f"[{chunk_id}] Dimension mismatch: expected {expected_dim}, got {len(vec)}",
            )

        # Model consistency
        expected_model = self.get_expected_model(target_domain)
        rec_model = record.get("model_name")
        if rec_model and rec_model != expected_model:
            return (
                False,
                f"[{chunk_id}] Model inconsistency: expected '{expected_model}', got '{rec_model}'",
            )

        # Numeric validity
        for v in vec:
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                return False, f"[{chunk_id}] Non-finite value detected in vector"

        return True, None

    def ingest_records(
        self,
        domain: Union[Domain, str],
        embedding_records: List[Dict[str, Any]],
        chunk_map: Dict[str, EvidenceChunk],
        batch_size: int = 100,
    ) -> IngestionReport:
        """Ingest pre-loaded embedding records joined with EvidenceChunks into ChromaDB.

        Args:
            domain: Domain of the collection.
            embedding_records: List of embedding dicts from JSONL.
            chunk_map: Dict mapping chunk_id to validated EvidenceChunk.
            batch_size: Batch size for Chroma upserts.

        Returns:
            IngestionReport summarizing results.
        """
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        col_name = self.get_collection_name(dom_enum)
        expected_dim = self.get_expected_dimension(dom_enum)
        expected_model = self.get_expected_model(dom_enum)

        report = IngestionReport(
            domain=dom_enum.value,
            collection_name=col_name,
            model_name=expected_model,
            dimension=expected_dim,
            input_embeddings=len(embedding_records),
            persist_dir=str(self.persist_dir),
        )

        collection = self.get_or_create_collection(dom_enum)

        valid_ids: List[str] = []
        valid_embeddings: List[List[float]] = []
        valid_documents: List[str] = []
        valid_metadatas: List[Dict[str, Any]] = []

        seen_in_batch: set = set()

        for rec in embedding_records:
            is_valid, err_msg = self.validate_embedding_record(rec, dom_enum)
            if not is_valid:
                report.rejected += 1
                report.add_error(err_msg or "Invalid embedding record")
                continue

            chunk_id = rec["chunk_id"]

            # Prevent intra-batch duplicates
            if chunk_id in seen_in_batch:
                report.add_warning(f"Duplicate chunk_id '{chunk_id}' encountered in input; updating latest.")
            seen_in_batch.add(chunk_id)

            # Check chunk existence for join
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                report.rejected += 1
                report.add_error(
                    f"[{chunk_id}] No corresponding EvidenceChunk found in chunk corpus"
                )
                continue

            # Verify chunk domain agrees with target domain
            if chunk.domain != dom_enum:
                report.rejected += 1
                report.add_error(
                    f"[{chunk_id}] EvidenceChunk domain '{chunk.domain.value}' != target '{dom_enum.value}'"
                )
                continue

            chroma_meta = serialize_chunk_metadata_for_chroma(chunk)

            valid_ids.append(chunk_id)
            valid_embeddings.append(rec["embedding"])
            valid_documents.append(chunk.text)
            valid_metadatas.append(chroma_meta)

        report.valid_records = len(valid_ids)

        # Perform batched upserts
        for i in range(0, len(valid_ids), batch_size):
            b_ids = valid_ids[i : i + batch_size]
            b_embs = valid_embeddings[i : i + batch_size]
            b_docs = valid_documents[i : i + batch_size]
            b_metas = valid_metadatas[i : i + batch_size]

            collection.upsert(
                ids=b_ids,
                embeddings=b_embs,
                documents=b_docs,
                metadatas=b_metas,
            )
            report.inserted_updated += len(b_ids)

        report.collection_count = collection.count()
        report.passed = report.rejected == 0 and len(report.errors) == 0
        return report

    def ingest_from_files(
        self,
        domain: Union[Domain, str],
        embeddings_path: Union[str, Path],
        chunks_path: Union[str, Path],
        batch_size: int = 100,
    ) -> IngestionReport:
        """Ingest embeddings from a JSONL file joined with a chunks JSONL file.

        Args:
            domain: Target domain ('legal' or 'scientific').
            embeddings_path: Path to embedding JSONL file.
            chunks_path: Path to EvidenceChunk JSONL file.
            batch_size: Batch size for upserts.

        Returns:
            IngestionReport.
        """
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        emb_p = Path(embeddings_path).resolve()
        chk_p = Path(chunks_path).resolve()

        if not emb_p.exists():
            rep = IngestionReport(
                domain=dom_enum.value,
                collection_name=self.get_collection_name(dom_enum),
                model_name=self.get_expected_model(dom_enum),
                dimension=self.get_expected_dimension(dom_enum),
                persist_dir=str(self.persist_dir),
            )
            rep.add_error(f"Embeddings file not found: {emb_p}")
            return rep

        if not chk_p.exists():
            rep = IngestionReport(
                domain=dom_enum.value,
                collection_name=self.get_collection_name(dom_enum),
                model_name=self.get_expected_model(dom_enum),
                dimension=self.get_expected_dimension(dom_enum),
                persist_dir=str(self.persist_dir),
            )
            rep.add_error(f"Chunks file not found: {chk_p}")
            return rep

        # Load chunks map
        chunk_map: Dict[str, EvidenceChunk] = {}
        with open(chk_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = EvidenceChunk.model_validate_json(line)
                    chunk_map[chunk.chunk_id] = chunk

        # Load embedding records
        embedding_records: List[Dict[str, Any]] = []
        with open(emb_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    embedding_records.append(json.loads(line))

        return self.ingest_records(
            domain=dom_enum,
            embedding_records=embedding_records,
            chunk_map=chunk_map,
            batch_size=batch_size,
        )

    def get_collection_stats(self, domain: Union[Domain, str]) -> Dict[str, Any]:
        """Retrieve current statistics for a domain collection."""
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        col = self.get_or_create_collection(dom_enum)
        return {
            "domain": dom_enum.value,
            "collection_name": col.name,
            "count": col.count(),
            "dimension": self.get_expected_dimension(dom_enum),
            "model_name": self.get_expected_model(dom_enum),
            "metadata": col.metadata,
            "persist_dir": str(self.persist_dir),
        }

    def verify_persistence(self, domain: Union[Domain, str]) -> Dict[str, Any]:
        """Verify on-disk persistence by initializing a completely fresh Chroma client.

        Args:
            domain: Domain collection to verify.

        Returns:
            Dictionary with reload verification metrics.
        """
        dom_enum = Domain(domain) if isinstance(domain, str) else domain
        col_name = self.get_collection_name(dom_enum)

        # Fresh client from scratch pointing to the same directory
        fresh_client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = fresh_client.get_collection(name=col_name)

        count = collection.count()
        sample = collection.peek(limit=1) if count > 0 else {}

        return {
            "reloaded": True,
            "collection_name": col_name,
            "count": count,
            "has_records": count > 0,
            "sample_ids": sample.get("ids", []),
            "sample_metadata": sample.get("metadatas", []),
        }
