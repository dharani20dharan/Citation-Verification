"""Semantic Claim-Evidence Verifier interface and implementations for Stage 10B."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from src.config.settings import get_settings
from src.extraction.models import CitationClaim
from src.references.models import BibliographicEntry
from src.verification.models import (
    EvidenceItem,
    LLMVerificationDecision,
    ReferenceStatus,
    Verdict,
    VerificationDecision,
    VerificationFailure,
)
from src.verification.prompt_builder import VerificationPromptBuilder
from src.verification.response_parser import ResponseParser, ResponseParserError
from src.verification.verifier import BaseClaimVerifier

logger = logging.getLogger(__name__)


class SemanticVerifier(Protocol):
    """Provider-independent protocol for semantic claim-evidence verification."""

    def verify(
        self,
        claim: CitationClaim,
        evidence: List[EvidenceItem],
        reference_status: ReferenceStatus,
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> VerificationDecision:
        """Evaluate whether trusted retrieved evidence supports the claim."""
        ...


class BaseSemanticVerifier(BaseClaimVerifier):
    """Abstract base class unifying SemanticVerifier and BaseClaimVerifier."""

    VERSION = "1.0.0-stage10b"

    def __init__(
        self,
        prompt_builder: Optional[VerificationPromptBuilder] = None,
        response_parser: Optional[ResponseParser] = None,
        model_name: str = "generic-semantic-verifier",
    ):
        self.prompt_builder = prompt_builder or VerificationPromptBuilder()
        self.response_parser = response_parser or ResponseParser()
        self.model_name = model_name

    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> VerificationDecision:
        """Verify claim against evidence and reference status."""
        raise NotImplementedError


class MockSemanticVerifier(BaseSemanticVerifier):
    """Deterministic mock semantic verifier for offline evaluation and unit testing."""

    def __init__(
        self,
        response: Optional[Union[Dict[str, Any], str]] = None,
        response_factory: Optional[Callable[[CitationClaim, List[EvidenceItem]], Dict[str, Any]]] = None,
        model_name: str = "mock-semantic-verifier",
    ):
        super().__init__(model_name=model_name)
        self.preset_response = response
        self.response_factory = response_factory

    def set_response(self, response: Union[Dict[str, Any], str]) -> None:
        """Update preset mock response for test cases."""
        self.preset_response = response

    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> VerificationDecision:
        """Execute mock semantic verification with full prompt generation and response parsing."""
        # 1. Guard check: reference status guard
        if reference_status == ReferenceStatus.NOT_FOUND:
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=(
                    f"The cited reference '{claim.reference_text}' was not found in the corpus. "
                    f"Semantic claim verification was not invoked."
                ),
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"guard_applied": "not_found", "verifier_type": "mock"},
            )

        if not evidence:
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=f"No evidence chunks were provided for claim '{claim.claim_id}'.",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"guard_applied": "empty_evidence", "verifier_type": "mock"},
            )

        # 2. Build prompt (validates prompt construction pipeline)
        prompt_str = self.prompt_builder.build_prompt(
            claim=claim,
            evidence=evidence,
            reference_status=reference_status,
            resolved_references=resolved_references,
        )

        supplied_ids = [e.evidence_id or e.chunk_id for e in evidence]

        # 3. Determine mock output
        raw_output: Union[Dict[str, Any], str]
        if self.preset_response is not None:
            raw_output = self.preset_response
        elif self.response_factory is not None:
            raw_output = self.response_factory(claim, evidence)
        else:
            # Default fallback mock behavior:
            # The default mock must NOT return supported merely because substantive evidence exists.
            eligible_substantive = [
                e for e in evidence if getattr(e, "is_eligible", True) and not e.is_structural
            ]
            first_id = eligible_substantive[0].evidence_id or eligible_substantive[0].chunk_id if eligible_substantive else None
            raw_output = {
                "verdict": "insufficient_evidence",
                "confidence": 0.70,
                "evidence_ids": [first_id] if first_id else [],
                "reasoning": (
                    "Default mock verifier does not infer semantic support without an explicitly configured "
                    "mock response or response factory. Insufficient evidence to establish support or contradiction."
                ),
            }

        # 4. Strictly validate through ResponseParser
        try:
            decision, diagnostics = self.response_parser.parse(
                raw_output=raw_output,
                supplied_evidence=evidence,
                claim_id=claim.claim_id,
            )
        except ResponseParserError as err:
            logger.warning("Mock response parsing error: %s", err)
            # Produce structured failure decision
            return VerificationDecision(
                verdict=Verdict.UNSUPPORTED,
                confidence=0.50,
                reasoning=f"[Verification Failure] {err.failure.message}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={
                    "verifier_type": "mock",
                    "failure": err.failure.model_dump(),
                    "prompt_length": len(prompt_str),
                },
            )

        return VerificationDecision(
            verdict=decision.verdict,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            verifier_version=f"{self.VERSION}:{self.model_name}",
            metadata={
                "verifier_type": "mock",
                "evidence_ids_used": decision.evidence_ids,
                "prompt_length": len(prompt_str),
                **diagnostics,
            },
        )


class OpenAISemanticVerifier(BaseSemanticVerifier):
    """Provider adapter calling OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        prompt_builder: Optional[VerificationPromptBuilder] = None,
        response_parser: Optional[ResponseParser] = None,
    ):
        settings = get_settings()
        chosen_model = model_name or getattr(settings, "LLM_MODEL", "gpt-4o-mini")
        super().__init__(
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            model_name=chosen_model,
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.timeout = timeout

    def verify(
        self,
        claim: CitationClaim,
        reference_status: ReferenceStatus,
        evidence: List[EvidenceItem],
        resolved_references: Optional[List[BibliographicEntry]] = None,
    ) -> VerificationDecision:
        """Call external LLM provider to evaluate claim against evidence."""
        # 1. Guard check
        if reference_status == ReferenceStatus.NOT_FOUND:
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=(
                    f"The cited reference '{claim.reference_text}' was not found in the corpus. "
                    f"LLM semantic verification was not invoked."
                ),
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"guard_applied": "not_found", "verifier_type": "llm"},
            )

        if not evidence:
            return VerificationDecision(
                verdict=Verdict.NOT_FOUND,
                confidence=0.95,
                reasoning=f"No evidence chunks were provided for claim '{claim.claim_id}'.",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"guard_applied": "empty_evidence", "verifier_type": "llm"},
            )

        if not self.api_key:
            # Safe controlled failure when no API key configured
            failure = VerificationFailure(
                error_type="configuration_error",
                message="OPENAI_API_KEY is not configured in environment or settings.",
                claim_id=claim.claim_id,
            )
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning=f"[Provider Configuration Error] {failure.message}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"failure": failure.model_dump(), "verifier_type": "llm"},
            )

        prompt_str = self.prompt_builder.build_prompt(
            claim=claim,
            evidence=evidence,
            reference_status=reference_status,
            resolved_references=resolved_references,
        )
        supplied_ids = [e.evidence_id or e.chunk_id for e in evidence]

        # Call API using urllib or requests/httpx if present, without leaking credentials
        try:
            import json
            import urllib.request
            import urllib.error

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": prompt_str},
                ],
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                raw_content = resp_data["choices"][0]["message"]["content"]

            decision, diagnostics = self.response_parser.parse(
                raw_output=raw_content,
                supplied_evidence=evidence,
                claim_id=claim.claim_id,
            )

            return VerificationDecision(
                verdict=decision.verdict,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={
                    "verifier_type": "llm",
                    "model_name": self.model_name,
                    "evidence_ids_used": decision.evidence_ids,
                    **diagnostics,
                },
            )

        except urllib.error.HTTPError as http_err:
            msg = f"HTTP Error calling LLM provider: {http_err.code}"
            logger.error(msg)
            failure = VerificationFailure(
                error_type="provider_error",
                message=msg,
                retryable=http_err.code in (429, 500, 503),
                claim_id=claim.claim_id,
            )
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning=f"[Provider Error] {msg}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"failure": failure.model_dump(), "verifier_type": "llm"},
            )
        except urllib.error.URLError as url_err:
            msg = f"Network or timeout error calling LLM provider: {url_err.reason}"
            logger.error(msg)
            failure = VerificationFailure(
                error_type="timeout" if "timed out" in str(url_err.reason).lower() else "provider_error",
                message=msg,
                retryable=True,
                claim_id=claim.claim_id,
            )
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning=f"[Network Error] {msg}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"failure": failure.model_dump(), "verifier_type": "llm"},
            )
        except ResponseParserError as parse_err:
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning=f"[Parsing Failure] {parse_err.failure.message}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"failure": parse_err.failure.model_dump(), "verifier_type": "llm"},
            )
        except Exception as exc:
            msg = f"Unexpected failure during LLM verification: {exc}"
            logger.error(msg)
            failure = VerificationFailure(
                error_type="provider_error",
                message=msg,
                claim_id=claim.claim_id,
            )
            return VerificationDecision(
                verdict=Verdict.PARTIALLY_CORRECT,
                confidence=0.50,
                reasoning=f"[Verification Error] {msg}",
                verifier_version=f"{self.VERSION}:{self.model_name}",
                metadata={"failure": failure.model_dump(), "verifier_type": "llm"},
            )
