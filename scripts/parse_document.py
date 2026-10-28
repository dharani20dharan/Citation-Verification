"""CLI script to parse a legal or scientific PDF/DOCX using DoclingDocumentParser."""

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
from src.parsing.docling_parser import (
    DoclingDocumentParser,
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedFormatError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parse_document")


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDF or DOCX documents with Docling into structured JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input PDF or DOCX document",
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
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Optional custom output directory. Defaults to data/parsed/<domain>/",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    domain_enum = Domain(args.domain)
    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info("Parsing document: %s (domain: %s)", input_path, domain_enum.value)

    doc_parser = DoclingDocumentParser()
    try:
        result = doc_parser.parse(file_path=input_path, domain=domain_enum)
        saved_file = doc_parser.save_parsed_result(result, output_dir=output_dir)

        print("\n" + "=" * 60)
        print("PARSING SUCCESSFUL")
        print("=" * 60)
        print(f"Document ID     : {result.document_id}")
        print(f"Title           : {result.title}")
        print(f"Domain          : {result.domain.value}")
        print(f"Pages           : {result.pages}")
        print(f"Total Elements  : {len(result.elements)}")
        print(f"Total Tables    : {len(result.tables)}")
        print(f"Saved Output    : {saved_file}")
        print("=" * 60 + "\n")

    except (DocumentNotFoundError, UnsupportedFormatError) as e:
        logger.error("Validation error: %s", str(e))
        sys.exit(1)
    except DocumentParsingError as e:
        logger.error("Parsing failed: %s", str(e))
        sys.exit(2)
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
