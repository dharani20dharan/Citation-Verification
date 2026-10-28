"""CLI script to generate domain-specific dense embeddings from validated EvidenceChunks."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config.settings import get_settings
from src.embeddings.factory import get_embedder
from src.metadata.validator import CorpusValidator
from src.models.schemas import Domain, EvidenceChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_embeddings")


def process_chunk_file(
    chunk_file: Path,
    domain: Domain,
    output_file: Path,
    limit: int = 0,
) -> int:
    """Validate, embed, and persist vectors for a single chunk file."""
    settings = get_settings()
    validator = CorpusValidator(settings=settings)

    # 1. Validate file before embedding
    logger.info("Validating chunk file: %s", chunk_file)
    val_report = validator.validate_jsonl_file(chunk_file)
    if not val_report.passed:
        logger.error(
            "Validation failed for %s. Refusing to generate embeddings.",
            chunk_file.name,
        )
        for err in val_report.errors[:10]:
            logger.error("  %s", err)
        return 0

    # 2. Load EvidenceChunks
    chunks: List[EvidenceChunk] = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(EvidenceChunk.model_validate_json(line))

    if limit > 0:
        logger.info("Limiting to first %d chunks for smoke test", limit)
        chunks = chunks[:limit]

    if not chunks:
        logger.warning("No chunks to embed in %s", chunk_file)
        return 0

    # 3. Instantiate domain-specific embedder
    embedder = get_embedder(domain=domain, settings=settings)
    logger.info(
        "Using embedder: %s (model: %s, device: %s, dimension: %d)",
        embedder.__class__.__name__,
        embedder.model_name,
        embedder.device,
        embedder.dimension,
    )

    # 4. Extract texts in exact sequence (ONLY chunk.text is embedded)
    texts = [c.text for c in chunks]

    # 5. Generate embeddings in batches
    logger.info("Generating embeddings for %d chunks...", len(texts))
    vectors = embedder.embed_texts(texts)

    # 6. Verify shape and values
    if len(vectors) != len(chunks):
        raise ValueError(
            f"Shape mismatch: {len(chunks)} chunks produced {len(vectors)} vectors"
        )

    for i, vec in enumerate(vectors):
        if len(vec) != embedder.dimension:
            raise ValueError(
                f"Dimension mismatch at index {i}: expected {embedder.dimension}, got {len(vec)}"
            )
        if any(math.isnan(v) or math.isinf(v) for v in vec):
            raise ValueError(f"Non-finite value detected in vector {i} for chunk {chunks[i].chunk_id}")

    # 7. Persist to intermediate JSONL format
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk, vec in zip(chunks, vectors):
            record = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "domain": chunk.domain.value,
                "model_name": embedder.model_name,
                "dimension": embedder.dimension,
                "embedding": vec,
            }
            f.write(json.dumps(record) + "\n")

    logger.info("Saved %d embeddings to %s", len(vectors), output_file)
    return len(vectors)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense embeddings for validated EvidenceChunks."
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["scientific", "legal", "all"],
        default="all",
        help="Domain to embed (scientific, legal, or all)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Optional specific input JSONL file",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help="Optional custom output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit number of chunks to process (for smoke testing)",
    )

    args = parser.parse_args()
    settings = get_settings()
    settings.ensure_directories()

    total_embedded = 0

    if args.input:
        input_path = Path(args.input).resolve()
        dom = Domain(args.domain) if args.domain != "all" else Domain.LEGAL
        out_path = Path(args.output_file).resolve() if args.output_file else (
            (settings.embeddings_legal_dir if dom == Domain.LEGAL else settings.embeddings_scientific_dir)
            / f"{input_path.stem}_embeddings.jsonl"
        )
        total_embedded += process_chunk_file(input_path, dom, out_path, limit=args.limit)
    else:
        domains_to_process = []
        if args.domain in {"legal", "all"}:
            domains_to_process.append((Domain.LEGAL, settings.chunks_legal_dir, settings.embeddings_legal_dir))
        if args.domain in {"scientific", "all"}:
            domains_to_process.append((Domain.SCIENTIFIC, settings.chunks_scientific_dir, settings.embeddings_scientific_dir))

        for dom, in_dir, out_dir in domains_to_process:
            files = sorted(list(in_dir.glob("*.jsonl")))
            for f in files:
                out_path = out_dir / f"{f.stem}_embeddings.jsonl"
                total_embedded += process_chunk_file(f, dom, out_path, limit=args.limit)

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Domain              : {args.domain}")
    print(f"Total Vectors Saved : {total_embedded}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
