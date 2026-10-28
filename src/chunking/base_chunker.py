"""Base chunking module providing shared abstractions and text splitting utilities."""

from abc import ABC, abstractmethod
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Set

from src.models.schemas import EvidenceChunk, DocumentParsingResult

logger = logging.getLogger(__name__)


def clean_section_slug(text: str) -> str:
    """Sanitize section title into a clean, deterministic identifier slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] if slug else "section"


def normalize_source_identifier(source_path: str, domain: Optional[str] = None) -> str:
    """Normalize a raw filesystem path into a stable, relative source identifier.

    Avoids leaking absolute developer filesystem paths into the corpus.
    Example: 'C:\\Users\\...\\data\\raw\\legal\\act.pdf' -> 'legal/act.pdf'.
    """
    p = Path(source_path)
    filename = p.name
    parent_name = p.parent.name.lower()
    if parent_name in {"legal", "scientific"}:
        return f"{parent_name}/{filename}"
    if domain:
        return f"{domain}/{filename}"
    return filename


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common boundaries and abbreviations."""
    # Split on sentence terminals followed by whitespace and a capital letter or quote
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    return sentences if sentences else [text.strip()]


def pack_sentences(
    sentences: List[str],
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> List[str]:
    """Pack sentences into chunks without breaking mid-sentence, using character overlap."""
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If adding this sentence exceeds max_chars and we already have content
        if current_len + sent_len + (1 if current_sentences else 0) > max_chars and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Retain overlap sentences
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_sent in reversed(current_sentences):
                if overlap_len + len(prev_sent) <= overlap_chars:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_len += len(prev_sent) + 1
                else:
                    break

            current_sentences = overlap_sentences + [sent]
            current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
        else:
            current_sentences.append(sent)
            current_len += sent_len + (1 if len(current_sentences) > 1 else 0)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


class BaseChunker(ABC):
    """Abstract base class for domain-specific chunking engines."""

    @abstractmethod
    def chunk_document(self, parsed_doc: DocumentParsingResult) -> List[EvidenceChunk]:
        """Convert a parsed document into a list of validated EvidenceChunks."""
        pass

    def validate_chunks(self, chunks: List[EvidenceChunk]) -> None:
        """Validate an entire collection of generated chunks for structural integrity.

        Ensures:
        - No duplicate chunk_ids
        - Text is non-empty
        - Pages are positive integers
        - Valid parent_ids if specified
        """
        seen_ids: Set[str] = set()
        chunk_id_set: Set[str] = {c.chunk_id for c in chunks}

        for chunk in chunks:
            # Check duplicate IDs
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id found in corpus: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)

            # Check text
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Empty text in chunk {chunk.chunk_id}")

            # Check page
            if chunk.page < 1:
                raise ValueError(f"Invalid page {chunk.page} in chunk {chunk.chunk_id}")

            # Check parent ID if pointing to a chunk in the same document
            if chunk.parent_id and chunk.parent_id not in chunk_id_set:
                logger.warning(
                    "Chunk %s has parent_id %s which is not in this document's chunk set.",
                    chunk.chunk_id,
                    chunk.parent_id,
                )

    @staticmethod
    def save_chunks_jsonl(chunks: List[EvidenceChunk], output_path: Path) -> Path:
        """Save a list of EvidenceChunks to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        logger.info("Saved %d chunks to %s", len(chunks), output_path)
        return output_path

    @staticmethod
    def load_chunks_jsonl(input_path: Path) -> List[EvidenceChunk]:
        """Load EvidenceChunks from a JSONL file."""
        chunks: List[EvidenceChunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    chunks.append(EvidenceChunk.model_validate(data))
                except Exception as e:
                    logger.error("Failed to parse JSONL line %d in %s: %s", line_no, input_path, str(e))
                    raise ValueError(f"Corrupt JSONL record at line {line_no} in {input_path}") from e
        return chunks
