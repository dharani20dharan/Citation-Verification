"""CLI script to chunk parsed documents into validated EvidenceChunk JSONL files."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.chunking.legal_chunker import LegalChunker
from src.chunking.scientific_chunker import ScientificChunker
from src.config.settings import get_settings
from src.models.schemas import Domain, DocumentParsingResult
from src.parsing.docling_parser import DoclingDocumentParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into validated EvidenceChunks (JSONL)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input parsed JSON file or raw document (.pdf/.docx)",
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["scientific", "legal"],
        required=True,
        help="Document domain (scientific or legal)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help="Optional custom output JSONL file path. Defaults to data/chunks/<domain>/<doc_id>.jsonl",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    domain_enum = Domain(args.domain)
    settings = get_settings()

    if not input_path.exists():
        logger.error("Input file not found at: %s", input_path)
        sys.exit(1)

    # 1. Obtain DocumentParsingResult: either from JSON or by parsing raw document
    if input_path.suffix.lower() == ".json":
        logger.info("Loading parsed document from JSON: %s", input_path)
        parsed_doc = DoclingDocumentParser.load_parsed_result(input_path)
    elif input_path.suffix.lower() in {".pdf", ".docx"}:
        logger.info("Parsing raw document: %s", input_path)
        doc_parser = DoclingDocumentParser(settings=settings)
        parsed_doc = doc_parser.parse(file_path=input_path, domain=domain_enum)
        # Also persist parsed JSON
        doc_parser.save_parsed_result(parsed_doc)
    else:
        logger.error("Unsupported file extension: %s. Expected .json, .pdf, or .docx", input_path.suffix)
        sys.exit(1)

    # 2. Invoke domain chunker
    if domain_enum == Domain.SCIENTIFIC:
        chunker = ScientificChunker(settings=settings)
    elif domain_enum == Domain.LEGAL:
        chunker = LegalChunker(settings=settings)
    else:
        logger.error("Unsupported domain: %s", domain_enum.value)
        sys.exit(1)

    chunks = chunker.chunk_document(parsed_doc)

    # 3. Determine output path
    if args.output_file:
        output_path = Path(args.output_file).resolve()
    else:
        chunks_dir = settings.chunks_scientific_dir if domain_enum == Domain.SCIENTIFIC else settings.chunks_legal_dir
        output_path = chunks_dir / f"{parsed_doc.document_id}.jsonl"

    chunker.save_chunks_jsonl(chunks, output_path)

    # Statistics computation
    sections = {c.section for c in chunks}
    parent_ids = {c.parent_id for c in chunks if c.parent_id is not None}
    table_chunks = [c for c in chunks if c.content_type.value == "table"]
    figure_chunks = [c for c in chunks if c.content_type.value in {"figure", "caption"}]
    clause_chunks = [c for c in chunks if c.content_type.value == "clause" or c.clause_id is not None]
    pages = {c.page for c in chunks}

    print("\n" + "=" * 60)
    print("CHUNKING SUCCESSFUL")
    print("=" * 60)
    print(f"Document ID         : {parsed_doc.document_id}")
    print(f"Title               : {parsed_doc.title}")
    print(f"Domain              : {parsed_doc.domain.value}")
    print(f"Total Chunks        : {len(chunks)}")
    print(f"Structural Sections : {len(sections)}")
    print(f"Parent Sections     : {len(parent_ids)}")
    print(f"Clause Chunks       : {len(clause_chunks)}")
    print(f"Table Chunks        : {len(table_chunks)}")
    print(f"Figure Chunks       : {len(figure_chunks)}")
    print(f"Page Distribution   : {sorted(list(pages))}")
    print(f"Saved JSONL Output  : {output_path}")

    # Representative Chunks Display
    print("\n--- Representative Chunks ---")
    rep_chunks = []
    # Try to find section parents, clauses, and children
    for c in chunks:
        if c.clause_id or "3" in c.section or c.content_type.value in {"section", "heading"}:
            rep_chunks.append(c)
            if len(rep_chunks) >= 4:
                break
    if not rep_chunks and chunks:
        rep_chunks = chunks[:3]

    for rc in rep_chunks:
        print(f"  [{rc.chunk_id}]")
        print(f"    Content Type : {rc.content_type.value}")
        print(f"    Section      : {rc.section} (ID: {rc.section_id})")
        print(f"    Parent ID    : {rc.parent_id}")
        if rc.clause_id:
            print(f"    Clause ID    : {rc.clause_id}")
        print(f"    Page         : {rc.page}")
        preview = rc.text.replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"    Text Preview : {preview}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
