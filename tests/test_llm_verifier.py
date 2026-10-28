"""Comprehensive test suite for Stage 10B: LLM Semantic Claim-Evidence Verification."""

import copy
import json
import pytest
from pydantic import ValidationError

from src.extraction.models import CitationClaim, ReferenceType
from src.models.schemas import Domain
from src.references.models import BibliographicEntry
from src.verification.llm_verifier import (
    MockSemanticVerifier,
    OpenAISemanticVerifier,
)
from src.verification.models import (
    EvidenceItem,
    LLMVerificationDecision,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
    VerificationFailure,
    VerificationResult,
)
from src.verification.prompt_builder import VerificationPromptBuilder
from src.verification.response_parser import ResponseParser, ResponseParserError
from src.verification.verification_engine import VerificationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prompt_builder() -> VerificationPromptBuilder:
    return VerificationPromptBuilder()


@pytest.fixture
def response_parser() -> ResponseParser:
    return ResponseParser()


@pytest.fixture
def sample_claim() -> CitationClaim:
    return CitationClaim(
        claim_id="claim_sci_001",
        document_id="doc_eval_01",
        domain=Domain.SCIENTIFIC,
        claim_text="Language models demonstrate few-shot learning capabilities without fine-tuning.",
        reference_text="[12]",
        normalized_reference="12",
        reference_type=ReferenceType.NUMERIC.value,
        page=2,
        chunk_id="chunk_eval_01",
    )


@pytest.fixture
def sample_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk_012",
            evidence_id="ev_012",
            document_id="doc_eval_01",
            text="We show that scaling language models greatly improves task-agnostic, few-shot performance.",
            match_type="exact",
            score=0.95,
            is_structural=False,
            evidence_type="substantive",
            provenance={"page": 3, "section": "Results"},
        ),
        EvidenceItem(
            chunk_id="chunk_heading_ref",
            evidence_id="ev_head",
            document_id="doc_eval_01",
            text="References",
            match_type="exact",
            score=0.90,
            is_structural=True,
            evidence_type="structural",
            provenance={"page": 3, "section": "References"},
        ),
    ]


@pytest.fixture
def sample_resolved_bib() -> list[BibliographicEntry]:
    return [
        BibliographicEntry(
            entry_id="doc_eval_01_ref_12",
            document_id="doc_eval_01",
            domain=Domain.SCIENTIFIC,
            reference_label="[12]",
            normalized_label="12",
            raw_text="[12] Brown, T. Language Models are Few-Shot Learners. NeurIPS, 2020.",
            title="Language Models are Few-Shot Learners",
            authors=["Brown, T.", "Mann, B."],
            year=2020,
            journal="NeurIPS",
            doi="10.48550/arXiv.2005.14165",
            page=3,
            section="References",
        )
    ]


# ---------------------------------------------------------------------------
# 1. Output Validation Tests
# ---------------------------------------------------------------------------

class TestOutputValidation:
    """Test strict schema enforcement on LLM structured output."""

    def test_valid_supported_output(self, response_parser: ResponseParser):
        raw = {
            "verdict": "supported",
            "confidence": 0.92,
            "evidence_ids": ["ev_012"],
            "reasoning": "The evidence directly establishes that scaling improves few-shot performance.",
        }
        decision, diag = response_parser.parse(raw, supplied_evidence_ids=["ev_012", "ev_head"])
        assert decision.verdict == Verdict.SUPPORTED
        assert decision.confidence == 0.92
        assert decision.evidence_ids == ["ev_012"]
        assert "scaling improves" in decision.reasoning

    def test_valid_unsupported_output(self, response_parser: ResponseParser):
        raw = {
            "verdict": "unsupported",
            "confidence": 0.88,
            "evidence_ids": ["ev_012"],
            "reasoning": "The evidence asserts the opposite effect.",
        }
        decision, diag = response_parser.parse(raw, supplied_evidence_ids=["ev_012"])
        assert decision.verdict == Verdict.UNSUPPORTED

    def test_valid_partially_correct_output(self, response_parser: ResponseParser):
        raw = {
            "verdict": "partially_correct",
            "confidence": 0.75,
            "evidence_ids": ["ev_012"],
            "reasoning": "Only the first part of the compound claim is supported.",
        }
        decision, diag = response_parser.parse(raw, supplied_evidence_ids=["ev_012"])
        assert decision.verdict == Verdict.PARTIALLY_CORRECT

    def test_invalid_verdict_rejected(self, response_parser: ResponseParser):
        # LLM must NOT produce not_found or wrong_reference (these are deterministic)
        for invalid_v in ["not_found", "wrong_reference", "hallucination", "partially_supported"]:
            raw = {
                "verdict": invalid_v,
                "confidence": 0.80,
                "evidence_ids": ["ev_012"],
                "reasoning": "Test invalid verdict",
            }
            with pytest.raises(ResponseParserError) as exc_info:
                response_parser.parse(raw, supplied_evidence_ids=["ev_012"])
            assert exc_info.value.failure.error_type == "schema_validation_error"

    def test_confidence_outside_bounds_rejected(self, response_parser: ResponseParser):
        for invalid_conf in [-0.1, 1.2, 5.0]:
            raw = {
                "verdict": "supported",
                "confidence": invalid_conf,
                "evidence_ids": ["ev_012"],
                "reasoning": "Test invalid confidence",
            }
            with pytest.raises(ResponseParserError) as exc_info:
                response_parser.parse(raw, supplied_evidence_ids=["ev_012"])
            assert exc_info.value.failure.error_type == "schema_validation_error"

    def test_malformed_json_rejected(self, response_parser: ResponseParser):
        malformed = "This is not json: verdict is supported"
        with pytest.raises(ResponseParserError) as exc_info:
            response_parser.parse(malformed, supplied_evidence_ids=["ev_012"])
        assert exc_info.value.failure.error_type == "invalid_response"


# ---------------------------------------------------------------------------
# 2. Evidence Grounding & ID Validation Tests
# ---------------------------------------------------------------------------

class TestEvidenceGrounding:
    """Test that model can only cite supplied evidence IDs."""

    def test_unknown_evidence_id_rejected(self, response_parser: ResponseParser):
        raw = {
            "verdict": "supported",
            "confidence": 0.90,
            "evidence_ids": ["ev_012", "ev_UNSUPPLIED_GHOST_ID"],
            "reasoning": "Evidence supports proposition.",
        }
        with pytest.raises(ResponseParserError) as exc_info:
            response_parser.parse(raw, supplied_evidence_ids=["ev_012", "ev_head"])
        assert exc_info.value.failure.error_type == "unknown_evidence_id"
        assert "ev_UNSUPPLIED_GHOST_ID" in exc_info.value.failure.message

    def test_unsupplied_chunk_id_cannot_be_cited(self, response_parser: ResponseParser):
        raw = {
            "verdict": "supported",
            "confidence": 0.85,
            "evidence_ids": ["chunk_not_in_evidence"],
            "reasoning": "...",
        }
        with pytest.raises(ResponseParserError) as exc_info:
            response_parser.parse(raw, supplied_evidence_ids=["ev_012"])
        assert exc_info.value.failure.error_type == "unknown_evidence_id"


# ---------------------------------------------------------------------------
# 3. Semantic Verification Scenarios (MockSemanticVerifier)
# ---------------------------------------------------------------------------

class TestSemanticScenarios:
    """Test semantic verification behaviors using MockSemanticVerifier."""

    def test_supported_semantic_verdict(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        mock_resp = {
            "verdict": "supported",
            "confidence": 0.94,
            "evidence_ids": ["ev_012"],
            "reasoning": "The substantive evidence explicitly verifies task-agnostic few-shot performance.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.EXACT_MATCH,
            evidence=sample_evidence,
        )
        assert decision.verdict == Verdict.SUPPORTED
        assert decision.confidence == 0.94
        assert "task-agnostic few-shot" in decision.reasoning
        assert decision.metadata["verifier_type"] == "mock"

    def test_unsupported_semantic_verdict(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        mock_resp = {
            "verdict": "unsupported",
            "confidence": 0.90,
            "evidence_ids": ["ev_012"],
            "reasoning": "The evidence establishes that few-shot learning was unsuccessful.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.EXACT_MATCH,
            evidence=sample_evidence,
        )
        assert decision.verdict == Verdict.UNSUPPORTED
        assert decision.confidence == 0.90

    def test_partially_correct_semantic_verdict(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        mock_resp = {
            "verdict": "partially_correct",
            "confidence": 0.80,
            "evidence_ids": ["ev_012"],
            "reasoning": "Evidence supports few-shot learning, but does not confirm it occurs without fine-tuning.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.EXACT_MATCH,
            evidence=sample_evidence,
        )
        assert decision.verdict == Verdict.PARTIALLY_CORRECT


# ---------------------------------------------------------------------------
# 4. Quantitative, Negation, and Compound Claims
# ---------------------------------------------------------------------------

class TestLinguisticAndQuantitativeClaims:
    """Test quantitative claims, polarity reversal, and compound propositions."""

    def test_quantitative_claim_unsupported_percentage_is_partially_correct(
        self, sample_claim: CitationClaim
    ):
        # Claim states 40% improvement, evidence only states improvement without number
        claim = copy.deepcopy(sample_claim)
        claim.claim_text = "The proposed algorithm improves retrieval accuracy by 40%."

        evidence = [
            EvidenceItem(
                chunk_id="c_quant_1",
                evidence_id="ev_quant_1",
                document_id="doc_eval_01",
                text="The proposed algorithm improves retrieval accuracy across all benchmarks.",
                match_type="exact",
                score=0.95,
            )
        ]

        mock_resp = {
            "verdict": "partially_correct",
            "confidence": 0.82,
            "evidence_ids": ["ev_quant_1"],
            "reasoning": "The evidence supports general accuracy improvement, but does not state the 40% figure.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)
        assert decision.verdict == Verdict.PARTIALLY_CORRECT

    def test_quantitative_claim_explicit_percentage_is_supported(
        self, sample_claim: CitationClaim
    ):
        claim = copy.deepcopy(sample_claim)
        claim.claim_text = "The proposed algorithm improves retrieval accuracy by 40%."

        evidence = [
            EvidenceItem(
                chunk_id="c_quant_2",
                evidence_id="ev_quant_2",
                document_id="doc_eval_01",
                text="The proposed algorithm improves retrieval accuracy by 40% on standard datasets.",
                match_type="exact",
                score=0.98,
            )
        ]

        mock_resp = {
            "verdict": "supported",
            "confidence": 0.95,
            "evidence_ids": ["ev_quant_2"],
            "reasoning": "The evidence explicitly confirms the 40% accuracy improvement.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)
        assert decision.verdict == Verdict.SUPPORTED

    def test_negation_polarity_reversal_is_unsupported(
        self, sample_claim: CitationClaim
    ):
        claim = copy.deepcopy(sample_claim)
        claim.claim_text = "The method improves retrieval throughput."

        evidence = [
            EvidenceItem(
                chunk_id="c_pol_1",
                evidence_id="ev_pol_1",
                document_id="doc_eval_01",
                text="Our findings show that the method does not improve retrieval throughput, and in fact degrades latency.",
                match_type="exact",
                score=0.92,
            )
        ]

        mock_resp = {
            "verdict": "unsupported",
            "confidence": 0.92,
            "evidence_ids": ["ev_pol_1"],
            "reasoning": "The evidence directly negates the proposition, stating that it does not improve throughput.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)
        assert decision.verdict == Verdict.UNSUPPORTED

    def test_compound_claim_partially_supported(
        self, sample_claim: CitationClaim
    ):
        claim = copy.deepcopy(sample_claim)
        claim.claim_text = "Method A improves precision and reduces memory footprint by 50%."

        evidence = [
            EvidenceItem(
                chunk_id="c_comp_1",
                evidence_id="ev_comp_1",
                document_id="doc_eval_01",
                text="Method A achieves superior precision on all test splits.",
                match_type="exact",
                score=0.91,
            )
        ]

        mock_resp = {
            "verdict": "partially_correct",
            "confidence": 0.85,
            "evidence_ids": ["ev_comp_1"],
            "reasoning": "Evidence supports precision improvement, but memory footprint reduction is not mentioned.",
        }
        verifier = MockSemanticVerifier(response=mock_resp)
        decision = verifier.verify(claim, ReferenceStatus.EXACT_MATCH, evidence)
        assert decision.verdict == Verdict.PARTIALLY_CORRECT


# ---------------------------------------------------------------------------
# 5. Reference Status Guard Tests (CRITICAL)
# ---------------------------------------------------------------------------

class TestReferenceStatusGuard:
    """Verify that reference_status=NOT_FOUND or empty evidence never triggers semantic LLM verification."""

    def test_not_found_reference_status_bypasses_llm(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        # Even if a mock response is configured to say 'supported', the guard must block it
        verifier = MockSemanticVerifier(
            response={
                "verdict": "supported",
                "confidence": 0.99,
                "evidence_ids": ["ev_012"],
                "reasoning": "I should never run!",
            }
        )
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.NOT_FOUND,
            evidence=sample_evidence,
        )
        assert decision.verdict == Verdict.NOT_FOUND
        assert decision.confidence == 0.95
        assert decision.metadata["guard_applied"] == "not_found"

    def test_empty_evidence_bypasses_llm(self, sample_claim: CitationClaim):
        verifier = MockSemanticVerifier(
            response={
                "verdict": "supported",
                "confidence": 0.99,
                "evidence_ids": [],
                "reasoning": "I should never run!",
            }
        )
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.EXACT_MATCH,
            evidence=[],
        )
        assert decision.verdict == Verdict.NOT_FOUND
        assert decision.metadata["guard_applied"] == "empty_evidence"

    def test_verification_engine_enforces_reference_guard(
        self, sample_claim: CitationClaim
    ):
        # Test that VerificationEngine itself never calls verifier if reference is not found
        class SpyVerifier(MockSemanticVerifier):
            called = False

            def verify(self, *args, **kwargs):
                self.called = True
                return super().verify(*args, **kwargs)

        spy = SpyVerifier()

        class MissingRetriever:
            def retrieve(self, request):
                return []

        engine = VerificationEngine(
            retriever=MissingRetriever(),
            verifier=spy,
        )
        claim_missing = copy.deepcopy(sample_claim)
        claim_missing.reference_text = "[99]"
        claim_missing.normalized_reference = "99"

        result = engine.verify(claim_missing)
        assert result.verdict == Verdict.NOT_FOUND
        assert result.reference_status == ReferenceStatus.NOT_FOUND
        assert not spy.called


# ---------------------------------------------------------------------------
# 6. Prompt Design Tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    """Verify that generated prompts contain all required sections and behavioral constraints."""

    def test_prompt_contains_all_required_sections(
        self,
        prompt_builder: VerificationPromptBuilder,
        sample_claim: CitationClaim,
        sample_evidence: list[EvidenceItem],
        sample_resolved_bib: list[BibliographicEntry],
    ):
        prompt = prompt_builder.build_prompt(
            claim=sample_claim,
            evidence=sample_evidence,
            reference_status=ReferenceStatus.EXACT_MATCH,
            resolved_references=sample_resolved_bib,
        )

        required_sections = [
            "### ROLE",
            "### TASK",
            "### CLAIM",
            "### CITED REFERENCE",
            "### REFERENCE STATUS",
            "### BIBLIOGRAPHIC CONTEXT",
            "### EVIDENCE",
            "### DECISION RULES",
            "### OUTPUT FORMAT",
        ]
        for sec in required_sections:
            assert sec in prompt, f"Missing section: {sec}"

    def test_prompt_includes_evidence_ids_and_types(
        self,
        prompt_builder: VerificationPromptBuilder,
        sample_claim: CitationClaim,
        sample_evidence: list[EvidenceItem],
    ):
        prompt = prompt_builder.build_prompt(
            claim=sample_claim,
            evidence=sample_evidence,
            reference_status=ReferenceStatus.EXACT_MATCH,
        )
        assert "ev_012" in prompt
        assert "ev_head" in prompt
        assert "Evidence Type: structural" in prompt
        assert "Evidence Type: substantive" in prompt

    def test_prompt_instructs_strict_isolation_and_grounding(
        self,
        prompt_builder: VerificationPromptBuilder,
        sample_claim: CitationClaim,
        sample_evidence: list[EvidenceItem],
    ):
        prompt = prompt_builder.build_prompt(
            claim=sample_claim,
            evidence=sample_evidence,
            reference_status=ReferenceStatus.EXACT_MATCH,
        )
        assert "Use ONLY the supplied evidence" in prompt
        assert "Do not rely on: \n- prior knowledge" in prompt or "prior knowledge" in prompt
        assert "Do not determine whether the citation exists" in prompt
        assert "Structural headings establish document organization" in prompt

    def test_bibliographic_metadata_marked_not_substantive_truth(
        self,
        prompt_builder: VerificationPromptBuilder,
        sample_claim: CitationClaim,
        sample_evidence: list[EvidenceItem],
        sample_resolved_bib: list[BibliographicEntry],
    ):
        prompt = prompt_builder.build_prompt(
            claim=sample_claim,
            evidence=sample_evidence,
            reference_status=ReferenceStatus.EXACT_MATCH,
            resolved_references=sample_resolved_bib,
        )
        assert "Language Models are Few-Shot Learners" in prompt
        assert "Bibliographic metadata identifies the cited source. It is not itself evidence that the claim is true" in prompt


# ---------------------------------------------------------------------------
# 7. Failure Handling & Provider Configuration Tests
# ---------------------------------------------------------------------------

class TestFailureHandling:
    """Test controlled failure handling without crashes or credential exposure."""

    def test_openai_verifier_missing_api_key_returns_controlled_failure(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        verifier = OpenAISemanticVerifier(api_key="")
        decision = verifier.verify(
            claim=sample_claim,
            reference_status=ReferenceStatus.EXACT_MATCH,
            evidence=sample_evidence,
        )
        assert decision.verdict == Verdict.PARTIALLY_CORRECT
        assert "OPENAI_API_KEY is not configured" in decision.reasoning
        assert decision.metadata["failure"]["error_type"] == "configuration_error"

    def test_mock_verifier_unknown_evidence_id_produces_failure(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        verifier = MockSemanticVerifier(
            response={
                "verdict": "supported",
                "confidence": 0.90,
                "evidence_ids": ["non_existent_id"],
                "reasoning": "...",
            }
        )
        decision = verifier.verify(sample_claim, ReferenceStatus.EXACT_MATCH, sample_evidence)
        assert decision.verdict == Verdict.UNSUPPORTED
        assert "[Verification Failure]" in decision.reasoning
        assert decision.metadata["failure"]["error_type"] == "unknown_evidence_id"


# ---------------------------------------------------------------------------
# 8. Determinism Tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify that repeated verification with identical inputs produces identical results."""

    def test_verification_decision_determinism(
        self, sample_claim: CitationClaim, sample_evidence: list[EvidenceItem]
    ):
        mock_resp = {
            "verdict": "supported",
            "confidence": 0.93,
            "evidence_ids": ["ev_012"],
            "reasoning": "The evidence directly and completely confirms the proposition.",
        }
        v1 = MockSemanticVerifier(response=mock_resp)
        v2 = MockSemanticVerifier(response=mock_resp)

        d1 = v1.verify(sample_claim, ReferenceStatus.EXACT_MATCH, sample_evidence)
        d2 = v2.verify(sample_claim, ReferenceStatus.EXACT_MATCH, sample_evidence)

        assert d1.verdict == d2.verdict
        assert d1.confidence == d2.confidence
        assert d1.reasoning == d2.reasoning
        assert d1.model_dump() == d2.model_dump()
