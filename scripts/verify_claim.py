"""CLI script to verify representative or custom claims against corpus evidence."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain
from src.retrieval.retriever import Retriever
from src.verification.models import VerificationResult
from src.verification.verification_engine import VerificationEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_claim_by_reference(domain: Domain, reference: str) -> Optional[CitationClaim]:
    """Look up an extracted CitationClaim matching the requested reference in data/extracted/."""
    extracted_dir = PROJECT_ROOT / "data" / "extracted" / domain.value
    if not extracted_dir.exists():
        return None

    ref_clean = reference.strip().lower()

    for path in extracted_dir.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                claim = CitationClaim.model_validate(data)

                # Match against reference_text, normalized_reference, or brackets
                r_text = claim.reference_text.strip().lower()
                r_norm = claim.normalized_reference.strip().lower()
                bracketed = f"[{r_norm}]"

                if ref_clean in (r_text, r_norm, bracketed):
                    return claim
                if r_text in ref_clean:
                    return claim

    return None


def construct_synthetic_claim(domain: Domain, reference: str, claim_text: Optional[str] = None) -> CitationClaim:
    """Construct a representative CitationClaim for non-fixture or negative reference testing."""
    clean_ref = reference.strip()
    norm_ref = clean_ref.replace("Section", "").replace("[", "").replace("]", "").strip()

    if domain == Domain.LEGAL:
        ref_type = ReferenceType.LEGAL_SECTION.value
        default_claim = f"Provisions of {clean_ref} govern invention criteria and statutory exemptions."
    else:
        ref_type = ReferenceType.NUMERIC.value
        default_claim = f"Recent empirical evaluations benchmark retrieval accuracy as described in {clean_ref}."

    text = claim_text.strip() if claim_text and claim_text.strip() else default_claim

    return CitationClaim(
        claim_id=f"cli_{domain.value}_{norm_ref.replace(' ', '_').lower()}_claim",
        document_id=f"cli_{domain.value}_doc",
        domain=domain,
        claim_text=text,
        reference_text=clean_ref,
        normalized_reference=norm_ref,
        normalized_references=[norm_ref],
        reference_type=ref_type,
        source_chunk_id=None,
    )


def print_verification_result(res: VerificationResult) -> None:
    """Format verification result cleanly according to Stage 10B specifications."""
    print("=" * 80)
    print("CLAIM VERIFICATION RESULT")
    print("=" * 80)
    print(f"CLAIM\n{res.claim_text}\n")
    print(f"REFERENCE\n{res.reference_text} (Normalized: {res.normalized_reference}, Type: {res.reference_type})\n")
    
    print("REFERENCE STATUS")
    print("-" * 16)
    print(f"{res.reference_status.value}\n")

    bib_res = res.provenance.get("bibliographic_resolution")
    if bib_res:
        print("BIBLIOGRAPHIC RESOLUTION")
        print("-" * 24)
        print(f"status  : {bib_res.get('status')}")
        if bib_res.get("matched_entry_ids"):
            print(f"entries : {', '.join(bib_res['matched_entry_ids'])}")
        print()

    source_avail = res.provenance.get("source_availability")
    if source_avail:
        print("SOURCE AVAILABILITY")
        print("-" * 19)
        print(f"{source_avail}\n")

    print("SEMANTIC VERIFIER")
    print("-" * 17)
    verifier_name = res.verifier_version
    print(f"{verifier_name}\n")

    print("VERDICT")
    print("-" * 7)
    print(f"{res.verdict.value}\n")

    print("CONFIDENCE")
    print("-" * 10)
    print(f"{res.confidence:.2f}\n")

    print("EVIDENCE")
    print("-" * 8)
    if not res.evidence:
        print("(No evidence chunks selected)")
    else:
        for idx, ev in enumerate(res.evidence, 1):
            type_tag = " [structural heading]" if ev.is_structural else " [substantive evidence]"
            print(f"[{idx}] chunk_id: {ev.chunk_id}{type_tag}")
            print(f"    match_type: {ev.match_type}")
            print(f"    score: {ev.score:.4f}")
            if ev.page is not None:
                print(f"    page: {ev.page}, section: {ev.section}")
            print(f"    text: {ev.text}")
            print()

    print("REASONING")
    print("-" * 9)
    print(res.reasoning)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a claim and its citation against corpus evidence.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["legal", "scientific"],
        help="Target corpus domain ('legal' or 'scientific')",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Citation reference to verify (e.g. 'Section 3(d)', 'Section 9', '[12]', '[3, 4]')",
    )
    parser.add_argument(
        "--claim-text",
        default=None,
        help="Optional custom claim text to verify instead of extracted fixture claim",
    )
    parser.add_argument(
        "--verifier",
        default="deterministic",
        choices=["deterministic", "mock", "llm"],
        help="Claim verifier implementation ('deterministic', 'mock', or 'llm', default: 'deterministic')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max retrieval candidates from Stage 7 (default: 5)",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=5,
        help="Max evidence items selected for evaluation (default: 5)",
    )

    args = parser.parse_args()
    domain = Domain(args.domain)

    # 1. Resolve claim
    claim: Optional[CitationClaim] = None
    if not args.claim_text:
        claim = find_claim_by_reference(domain, args.reference)

    if claim is None:
        claim = construct_synthetic_claim(domain, args.reference, args.claim_text)
    elif args.claim_text:
        # Override claim text if explicitly specified
        claim.claim_text = args.claim_text.strip()

    # 2. Select verifier implementation
    from src.verification.llm_verifier import MockSemanticVerifier, OpenAISemanticVerifier
    from src.verification.verifier import DeterministicClaimVerifier
    from src.references.resolution_engine import ReferenceResolutionEngine

    if args.verifier == "mock":
        selected_verifier = MockSemanticVerifier()
    elif args.verifier == "llm":
        selected_verifier = OpenAISemanticVerifier()
    else:
        selected_verifier = DeterministicClaimVerifier()

    ref_engine = ReferenceResolutionEngine()

    # 3. Run verification engine
    engine = VerificationEngine(
        verifier=selected_verifier,
        reference_resolver=ref_engine,
        top_k=args.top_k,
        max_evidence=args.max_evidence,
    )
    result = engine.verify(claim)

    # 4. Print output
    print_verification_result(result)


if __name__ == "__main__":
    main()

