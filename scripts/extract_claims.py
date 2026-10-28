"""CLI script for extracting citation claims from chunked corpora."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.schemas import Domain, EvidenceChunk
from src.extraction.extractor import ClaimCitationExtractor
from src.extraction.models import CitationClaim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_claims")


def process_domain(
    domain: Domain,
    chunks_dir: Path,
    output_dir: Path,
    extractor: ClaimCitationExtractor,
) -> List[CitationClaim]:
    """Process all chunk JSONL files for a given domain and save extracted claims."""
    domain_chunks_dir = chunks_dir / domain.value
    domain_output_dir = output_dir / domain.value
    domain_output_dir.mkdir(parents=True, exist_ok=True)

    if not domain_chunks_dir.exists():
        logger.warning("No chunks directory found at %s", domain_chunks_dir)
        return []

    chunk_files = sorted(domain_chunks_dir.glob("*.jsonl"))
    if not chunk_files:
        logger.warning("No JSONL chunk files found in %s", domain_chunks_dir)
        return []

    all_domain_claims: List[CitationClaim] = []

    for file_path in chunk_files:
        logger.info("Processing chunks from %s", file_path.name)
        chunks: List[EvidenceChunk] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = EvidenceChunk.model_validate(data)
                    chunks.append(chunk)
                except Exception as exc:
                    logger.error("Failed to parse chunk at line %d in %s: %s", line_no, file_path.name, exc)

        claims = extractor.extract_from_chunks(chunks)
        all_domain_claims.extend(claims)

        # Write output JSONL
        out_file = domain_output_dir / f"{file_path.stem}_claims.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for claim in claims:
                f.write(json.dumps(claim.model_dump(mode="json")) + "\n")

        logger.info(
            "Saved %d extracted claims to %s",
            len(claims),
            out_file,
        )

    return all_domain_claims


def main():
    parser = argparse.ArgumentParser(
        description="Extract verifiable claims and associated citations from chunked documents.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["legal", "scientific", "all"],
        help="Target corpus domain (legal, scientific, or all)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/chunks",
        help="Directory containing domain chunk JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted",
        help="Directory to save extracted claims JSONL files",
    )

    args = parser.parse_args()
    chunks_path = Path(args.chunks_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    extractor = ClaimCitationExtractor()
    domains_to_process = (
        [Domain.LEGAL, Domain.SCIENTIFIC]
        if args.domain == "all"
        else [Domain(args.domain)]
    )

    total_claims = 0
    print("\n" + "=" * 65)
    print("CITATION AND CLAIM EXTRACTION PIPELINE")
    print("=" * 65)

    for domain in domains_to_process:
        claims = process_domain(
            domain=domain,
            chunks_dir=chunks_path,
            output_dir=output_path,
            extractor=extractor,
        )
        total_claims += len(claims)

        print(f"\nDomain: {domain.value.upper()} | Extracted Claims: {len(claims)}")
        print("-" * 65)
        for i, claim in enumerate(claims[:5], start=1):
            print(f"[{i}] Claim ID : {claim.claim_id}")
            print(f"    Reference: {claim.reference_text} -> (norm: '{claim.normalized_reference}') [{claim.reference_type}]")
            print(f"    Claim    : \"{claim.claim_text}\"")
            print(f"    Chunk ID : {claim.source_chunk_id} | Page: {claim.page}")
            print()

    print("=" * 65)
    print(f"EXTRACTION COMPLETE: {total_claims} claims extracted.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
