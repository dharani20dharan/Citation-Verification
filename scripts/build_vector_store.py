"""CLI script to ingest dense embeddings and EvidenceChunks into persistent ChromaDB collections."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config.settings import get_settings
from src.models.schemas import Domain
from src.vectorstore.chroma_store import ChromaVectorStore, IngestionReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_vector_store")


def print_ingestion_report(report: IngestionReport) -> None:
    """Print the human-readable ChromaDB ingestion report."""
    print("\n" + "=" * 60)
    print("CHROMA INGESTION")
    print("=" * 60)
    print(f"Domain              : {report.domain}")
    print(f"Collection          : {report.collection_name}")
    print(f"Embedding Model     : {report.model_name}")
    print(f"Dimension           : {report.dimension}")
    print()
    print(f"Input Embeddings    : {report.input_embeddings}")
    print(f"Valid Records       : {report.valid_records}")
    print(f"Inserted/Updated    : {report.inserted_updated}")
    print(f"Rejected            : {report.rejected}")
    print()
    print(f"Collection Count    : {report.collection_count}")
    print(f"Persistence Path    : {report.persist_dir}")
    print()
    status_str = "PASS" if report.passed else "FAIL"
    print(f"STATUS              : {status_str}")
    print("=" * 60)

    if report.errors:
        print("\n--- Ingestion Errors ---")
        for err in report.errors[:20]:
            print(f"  [ERROR] {err}")
        if len(report.errors) > 20:
            print(f"  ... and {len(report.errors) - 20} more errors.")

    if report.warnings:
        print("\n--- Ingestion Warnings ---")
        for warn in report.warnings[:10]:
            print(f"  [WARN]  {warn}")
    print()


def ingest_domain(
    store: ChromaVectorStore,
    domain: Domain,
    reset: bool = False,
    batch_size: int = 100,
) -> bool:
    """Ingest embeddings and chunks for a specific domain."""
    settings = get_settings()

    if reset:
        logger.warning("Explicit --reset requested: Recreating collection for %s", domain.value)
        store.reset_collection(domain)

    # Locate embeddings and chunks
    emb_dir = settings.embeddings_legal_dir if domain == Domain.LEGAL else settings.embeddings_scientific_dir
    chk_dir = settings.chunks_legal_dir if domain == Domain.LEGAL else settings.chunks_scientific_dir

    emb_files = sorted(list(emb_dir.glob("*.jsonl")))
    if not emb_files:
        logger.warning("No embedding files found in %s", emb_dir)
        return False

    all_passed = True
    for emb_f in emb_files:
        # Match chunk file by prefix or doc_id
        # e.g. legal_sample_fixture_embeddings.jsonl -> legal_sample_fixture.jsonl
        doc_stem = emb_f.stem.replace("_embeddings", "")
        chk_f = chk_dir / f"{doc_stem}.jsonl"

        if not chk_f.exists():
            logger.error("Corresponding chunk file not found: %s for %s", chk_f, emb_f.name)
            all_passed = False
            continue

        logger.info("Ingesting %s joined with %s into %s", emb_f.name, chk_f.name, store.get_collection_name(domain))
        report = store.ingest_from_files(
            domain=domain,
            embeddings_path=emb_f,
            chunks_path=chk_f,
            batch_size=batch_size,
        )
        print_ingestion_report(report)
        if not report.passed:
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Ingest validated embeddings and EvidenceChunks into persistent ChromaDB collections."
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["legal", "scientific", "all"],
        default="all",
        help="Domain collection to populate (legal, scientific, or all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Explicitly delete and recreate the target collection before ingestion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Chroma upsert operations (default: 100)",
    )

    args = parser.parse_args()
    settings = get_settings()
    store = ChromaVectorStore(settings=settings)

    domains = []
    if args.domain in {"legal", "all"}:
        domains.append(Domain.LEGAL)
    if args.domain in {"scientific", "all"}:
        domains.append(Domain.SCIENTIFIC)

    overall_success = True
    for dom in domains:
        success = ingest_domain(
            store=store,
            domain=dom,
            reset=args.reset,
            batch_size=args.batch_size,
        )
        if not success:
            overall_success = False

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
