"""CLI script to build and store bibliographic entries from parsed/chunked documents."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import Domain, EvidenceChunk
from src.references.models import BibliographicEntry
from src.references.reference_detector import ReferenceSectionDetector
from src.references.reference_parser import BibliographicParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_references")


# Synthetic reference entries for testing resolution mechanics (Section 17)
SYNTHETIC_SCIENTIFIC_REFERENCES = [
    {
        "raw_text": "[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. Attention is All You Need. Advances in Neural Information Processing Systems, 2017.",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_003",
    },
    {
        "raw_text": "[4] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_004",
    },
    {
        "raw_text": "[5] Thorne, J., Vlachos, A., Christodoulopoulos, C., Mittal, A. FEVER: a large-scale dataset for Fact Extraction and VERification. NAACL-HLT, 2018.",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_005",
    },
    {
        "raw_text": "[6] Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., Hajishirzi, H. Fact or Fiction: Verifying Scientific Claims. EMNLP, 2020.",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_006",
    },
    {
        "raw_text": "[7] Schuster, T., Shah, D., Yeo, Y. J., Filizzola, R., Santus, E., Barzilay, R. Toward Robust Fact-Checking: Analyzing and Mitigating Bias in FEVER. EMNLP, 2019.",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_007",
    },
    {
        "raw_text": "[12] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., et al. Language Models are Few-Shot Learners. NeurIPS, 2020. doi:10.48550/arXiv.2005.14165",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_012",
    },
    {
        "raw_text": "Smith, J., Johnson, K., and Williams, R. (2024). Precision and Recall in Automated Citation Verification. Journal of Artificial Intelligence Research, 79, 101-125. https://doi.org/10.1613/jair.1.12345",
        "page": 3,
        "section": "References",
        "chunk_id": "sci_eval_01_ref_smith_2024",
    },
]


def build_references_for_domain(domain: Domain) -> int:
    """Build and save bibliographic reference entries for a domain."""
    references_dir = PROJECT_ROOT / "data" / "references" / domain.value
    references_dir.mkdir(parents=True, exist_ok=True)

    detector = ReferenceSectionDetector()
    parser = BibliographicParser()
    total_entries = 0

    if domain == Domain.LEGAL:
        # Legal domain has statutory provisions, not bibliographic references
        logger.info("Legal domain does not use bibliographic references. Directory prepared at %s", references_dir)
        return 0

    # 1. Check chunks directory
    chunks_dir = PROJECT_ROOT / "data" / "chunks" / domain.value
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*.jsonl"):
            doc_id = chunk_file.stem
            chunks: List[EvidenceChunk] = []
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(EvidenceChunk.model_validate_json(line))

            raw_refs = detector.detect_from_chunks(chunks, document_id=doc_id, domain=domain)
            parsed_entries: List[BibliographicEntry] = []

            for r in raw_refs:
                entry = parser.parse_entry(
                    raw_text=r["raw_text"],
                    document_id=doc_id,
                    domain=domain,
                    page=r.get("page"),
                    section=r.get("section"),
                    chunk_id=r.get("chunk_id"),
                )
                parsed_entries.append(entry)

            if parsed_entries:
                out_path = references_dir / f"{doc_id}_references.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for e in parsed_entries:
                        f.write(e.model_dump_json() + "\n")
                logger.info("Saved %d extracted references to %s", len(parsed_entries), out_path.name)
                total_entries += len(parsed_entries)

    # 2. Build synthetic reference fixture for evaluation document sci_eval_01 (Section 17)
    eval_doc_id = "sci_eval_01"
    eval_entries: List[BibliographicEntry] = []
    for item in SYNTHETIC_SCIENTIFIC_REFERENCES:
        entry = parser.parse_entry(
            raw_text=item["raw_text"],
            document_id=eval_doc_id,
            domain=domain,
            page=item["page"],
            section=item["section"],
            chunk_id=item["chunk_id"],
        )
        eval_entries.append(entry)

    eval_out_path = references_dir / f"{eval_doc_id}_references.jsonl"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        for e in eval_entries:
            f.write(e.model_dump_json() + "\n")
    logger.info("Saved %d synthetic evaluation references to %s", len(eval_entries), eval_out_path.name)
    total_entries += len(eval_entries)

    return total_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and build bibliography entries from corpus documents.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["scientific", "legal"],
        help="Domain to build references for ('scientific' or 'legal')",
    )
    args = parser.parse_args()

    domain = Domain(args.domain)
    count = build_references_for_domain(domain)
    print(f"\nSuccessfully built {count} bibliographic entries for domain '{domain.value}'.")


if __name__ == "__main__":
    main()
