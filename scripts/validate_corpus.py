"""CLI script to validate corpus files and metadata before embeddings/vector storage."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config.settings import get_settings
from src.metadata.validator import CorpusValidator, ValidationReport
from src.models.schemas import Domain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_corpus")


def print_report(report: ValidationReport) -> None:
    """Print the human-readable corpus validation report."""
    print("\n" + "=" * 60)
    print("CORPUS VALIDATION")
    print("=" * 60)
    print(f"Domain              : {report.domain}")
    print(f"Documents           : {report.documents_count}")
    print(f"Total Chunks        : {report.total_chunks}")
    print(f"Valid Chunks        : {report.valid_chunks}")
    print(f"Embeddable Chunks   : {report.embeddable_chunks}")
    print(f"Non-Embeddable      : {report.non_embeddable_chunks}")
    print(f"Errors              : {len(report.errors)}")
    print(f"Warnings            : {len(report.warnings)}")
    print()
    print(f"Parent References   : {'VALID' if report.parent_references_valid else 'INVALID'}")
    print(f"Chunk IDs           : {'UNIQUE' if report.chunk_ids_unique else 'DUPLICATE'}")
    print(f"Source Paths        : {'PORTABLE' if report.source_paths_portable else 'NON-PORTABLE'}")

    if report.domain in {"legal", "all"}:
        print(f"Legal Identifiers   : {'VALID' if report.legal_identifiers_valid else 'INVALID'}")
    if report.domain in {"scientific", "all"}:
        print(f"Scientific Metadata : {'VALID' if report.scientific_metadata_valid else 'INVALID'}")

    print()
    status_str = "PASS" if report.passed else "FAIL"
    print(f"STATUS              : {status_str}")
    print("=" * 60)

    if report.errors:
        print("\n--- Validation Errors ---")
        for err in report.errors[:20]:
            print(f"  [ERROR] {err}")
        if len(report.errors) > 20:
            print(f"  ... and {len(report.errors) - 20} more errors.")

    if report.warnings:
        print("\n--- Validation Warnings ---")
        for warn in report.warnings[:10]:
            print(f"  [WARN]  {warn}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate corpus chunks and metadata across legal and scientific domains."
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["legal", "scientific", "all"],
        default="all",
        help="Domain to validate (legal, scientific, or all)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Optional specific .jsonl file to validate",
    )

    args = parser.parse_args()
    settings = get_settings()
    validator = CorpusValidator(settings=settings)

    overall_passed = True

    if args.file:
        file_path = Path(args.file).resolve()
        logger.info("Validating single file: %s", file_path)
        report = validator.validate_jsonl_file(file_path)
        report.domain = args.domain
        print_report(report)
        if not report.passed:
            overall_passed = False
    else:
        domains_to_validate = []
        if args.domain in {"legal", "all"}:
            domains_to_validate.append((Domain.LEGAL, settings.chunks_legal_dir))
        if args.domain in {"scientific", "all"}:
            domains_to_validate.append((Domain.SCIENTIFIC, settings.chunks_scientific_dir))

        for dom, chunk_dir in domains_to_validate:
            logger.info("Validating domain %s at %s", dom.value, chunk_dir)
            report = validator.validate_directory(chunk_dir, domain=dom)
            print_report(report)
            if not report.passed:
                overall_passed = False

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
