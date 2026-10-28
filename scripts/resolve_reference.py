"""CLI script to resolve citation references to bibliographic entries."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import Domain
from src.references.models import ReferenceResolutionResult
from src.references.resolution_engine import ReferenceResolutionEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def print_resolution_result(result: ReferenceResolutionResult) -> None:
    """Format and print resolution result cleanly according to Stage 10A specifications."""
    print("=" * 80)
    print("REFERENCE RESOLUTION RESULT")
    print("=" * 80)
    print(f"REFERENCE\n{result.reference_text}\n")
    print(f"TYPE\n{result.reference_type}\n")
    print(f"STATUS\n{result.status.value}\n")
    print(f"CONFIDENCE\n{result.confidence:.2f} (deterministic tier)\n")

    print("MATCHED BIBLIOGRAPHIC ENTRIES")
    print("-" * 30)
    if not result.matched_entries:
        print("(No bibliographic entries matched)")
    else:
        for idx, entry in enumerate(result.matched_entries, 1):
            print(f"[{idx}] entry_id: {entry.entry_id}")
            print(f"    label   : {entry.reference_label} (normalized: {entry.normalized_label})")
            if entry.authors:
                print(f"    authors : {', '.join(entry.authors)}")
            if entry.year is not None:
                print(f"    year    : {entry.year}")
            if entry.title:
                print(f"    title   : {entry.title}")
            if entry.journal:
                print(f"    venue   : {entry.journal}")
            if entry.doi:
                print(f"    doi     : {entry.doi}")
            if entry.url:
                print(f"    url     : {entry.url}")
            print(f"    raw_text: {entry.raw_text}")
            print()

    print("PROVENANCE")
    print("-" * 10)
    if not result.matched_entries:
        print(f"searched: {result.provenance}")
    else:
        for idx, entry in enumerate(result.matched_entries, 1):
            print(f"[{idx}] document_id: {entry.document_id}")
            print(f"    page       : {entry.page}")
            print(f"    section    : {entry.section}")
            print(f"    chunk_id   : {entry.chunk_id}")
            print()

    print("REASONING")
    print("-" * 9)
    print(result.reasoning)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a citation reference to bibliography entry.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["scientific", "legal"],
        help="Target corpus domain ('scientific' or 'legal')",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Citation reference string to resolve (e.g. '[12]', '[3, 4]', '[5-7]', '(Smith et al., 2024)', 'Section 3(d)')",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to scope bibliographic resolution",
    )

    args = parser.parse_args()
    domain = Domain(args.domain)

    engine = ReferenceResolutionEngine()
    result = engine.resolve_reference(
        reference_text=args.reference,
        domain=domain,
        document_id=args.document_id,
    )

    print_resolution_result(result)


if __name__ == "__main__":
    main()
