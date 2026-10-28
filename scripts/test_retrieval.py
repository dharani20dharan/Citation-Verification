"""Diagnostic CLI tool to test exact matching, semantic search, and context expansion."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.schemas import Domain
from src.retrieval.models import RetrievalRequest
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_retrieval")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic CLI for testing exact matching, semantic search, and context expansion."
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["legal", "scientific"],
        required=True,
        help="Target corpus domain",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Query text string to search",
    )
    parser.add_argument(
        "--identifier",
        "-i",
        type=str,
        default=None,
        help="Optional exact structural identifier (e.g. '3(d)', 'Section 3(e)', '3(d)(i)')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Maximum semantic candidates (default: 5)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic search (exact match + context only)",
    )
    parser.add_argument(
        "--no-parent",
        action="store_true",
        help="Disable parent-context expansion",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Maximum parent hierarchy depth (default: 2)",
    )

    args = parser.parse_args()
    domain_enum = Domain(args.domain)

    request = RetrievalRequest(
        query=args.query,
        domain=domain_enum,
        top_k=args.top_k,
        exact_identifier=args.identifier,
        expand_parent_context=not args.no_parent,
        semantic_search_enabled=not args.no_semantic,
        max_parent_depth=args.depth,
    )

    retriever = Retriever()
    results = retriever.retrieve(request)

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    print(f"Domain       : {args.domain}")
    print(f"Query        : {args.query}")
    print(f"Identifier   : {args.identifier or 'None'}")
    print(f"Total Found  : {len(results)}")
    print()
    print(f"{'Rank':<6}{'Match Type':<18}{'Chunk ID'}")
    print("-" * 60)
    for r in results:
        print(f"{r.rank:<6}{r.match_type:<18}{r.chunk_id}")

    print("\nProvenance Details:")
    print("-" * 60)
    for r in results:
        preview = r.text.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        score_str = f"{r.score:.4f}"
        print(f"  [Rank {r.rank}] ({r.match_type}) Score: {score_str}")
        print(f"    Chunk ID    : {r.chunk_id}")
        print(f"    Document ID : {r.document_id}")
        print(f"    Section     : {r.section} (ID: {r.section_id})")
        print(f"    Parent ID   : {r.parent_id or 'None'}")
        if r.clause_id:
            print(f"    Clause ID   : {r.clause_id}")
        if r.page is not None:
            print(f"    Page        : {r.page}")
        print(f"    Text        : {preview}")
        print()
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
