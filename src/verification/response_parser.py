"""Response parser strictly validating structured LLM verification outputs."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import ValidationError

from src.verification.models import (
    EvidenceItem,
    LLMVerificationDecision,
    Verdict,
    VerificationFailure,
)

logger = logging.getLogger(__name__)


class ResponseParserError(Exception):
    """Exception raised when model output fails parsing or grounding validation."""

    def __init__(self, failure: VerificationFailure):
        self.failure = failure
        super().__init__(failure.message)


class ResponseParser:
    """Parses, validates, and grounds structured responses from LLM semantic verifiers."""

    def parse(
        self,
        raw_output: Union[str, Dict[str, Any]],
        supplied_evidence: Optional[Union[List[EvidenceItem], List[str]]] = None,
        claim_id: Optional[str] = None,
        supplied_evidence_ids: Optional[Union[List[EvidenceItem], List[str]]] = None,
    ) -> Tuple[LLMVerificationDecision, Dict[str, Any]]:
        """Parse and strictly validate structured LLM response.

        Args:
            raw_output: Raw text response or parsed dictionary from LLM.
            supplied_evidence: List of valid EvidenceItem objects or string evidence IDs.
            claim_id: Optional claim identifier for error tracing.
        Returns:
            Tuple of (validated LLMVerificationDecision, diagnostic metadata).

        Raises:
            ResponseParserError: If response fails JSON syntax, schema validation,
                                 cites unsupplied evidence IDs, cites only structural headings,
                                 or cites ineligible source evidence for support.
        """
        if supplied_evidence is None:
            supplied_evidence = supplied_evidence_ids or []
        raw_text_record = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)

        # 1. Parse JSON if string
        if isinstance(raw_output, str):
            clean_str = self._extract_json_substring(raw_output)
            try:
                data = json.loads(clean_str)
            except json.JSONDecodeError as err:
                msg = f"Failed to parse model response as JSON: {err}. Raw output: {raw_output[:200]}"
                logger.warning(msg)
                raise ResponseParserError(
                    VerificationFailure(
                        error_type="invalid_response",
                        message=msg,
                        stage="response_parsing",
                        claim_id=claim_id,
                    )
                ) from err
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            msg = f"Unexpected response type: {type(raw_output)}"
            raise ResponseParserError(
                VerificationFailure(
                    error_type="invalid_response",
                    message=msg,
                    stage="response_parsing",
                    claim_id=claim_id,
                )
            )

        # 2. Schema validation with Pydantic
        try:
            decision = LLMVerificationDecision.model_validate(data)
        except ValidationError as val_err:
            msg = f"Model response failed schema validation: {val_err}"
            logger.warning(msg)
            raise ResponseParserError(
                VerificationFailure(
                    error_type="schema_validation_error",
                    message=msg,
                    stage="schema_validation",
                    claim_id=claim_id,
                )
            ) from val_err

        # Build ID lookup and item lookup
        evidence_by_id: Dict[str, Optional[EvidenceItem]] = {}
        for item in supplied_evidence:
            if isinstance(item, str):
                evidence_by_id[item] = None
            else:
                eid = item.evidence_id or item.chunk_id
                evidence_by_id[eid] = item

        valid_ids: Set[str] = set(evidence_by_id.keys())
        cited_ids = decision.evidence_ids

        # 3. Grounding validation: verify all cited evidence_ids exist in supplied_evidence
        unknown_ids = [eid for eid in cited_ids if eid not in valid_ids]
        if unknown_ids:
            msg = (
                f"Model cited unknown/unsupplied evidence_ids: {unknown_ids}. "
                f"Supplied valid evidence_ids: {sorted(list(valid_ids))}."
            )
            logger.warning(msg)
            raise ResponseParserError(
                VerificationFailure(
                    error_type="unknown_evidence_id",
                    message=msg,
                    stage="evidence_grounding_validation",
                    claim_id=claim_id,
                )
            )

        # 4. Strict support validation: model claiming support must cite eligible substantive evidence
        if decision.verdict == Verdict.SUPPORTED:
            if not cited_ids:
                msg = "Model asserted supported verdict without citing any evidence IDs."
                logger.warning(msg)
                raise ResponseParserError(
                    VerificationFailure(
                        error_type="missing_evidence_ids",
                        message=msg,
                        stage="support_evidence_validation",
                        claim_id=claim_id,
                    )
                )

            # Check if all cited evidence items are structural headings
            cited_items = [evidence_by_id[eid] for eid in cited_ids if evidence_by_id.get(eid) is not None]
            if cited_items and all(e.is_structural or e.evidence_type == "structural" for e in cited_items):
                msg = "Model cited only structural heading evidence for factual support, which is not permitted."
                logger.warning(msg)
                raise ResponseParserError(
                    VerificationFailure(
                        error_type="structural_only_support",
                        message=msg,
                        stage="support_evidence_validation",
                        claim_id=claim_id,
                    )
                )

            # Check if all cited evidence items are marked ineligible (e.g. from an unrelated publication)
            if cited_items and all(not getattr(e, "is_eligible", True) for e in cited_items):
                msg = "Model cited evidence from an ineligible or unverified publication for factual support."
                logger.warning(msg)
                raise ResponseParserError(
                    VerificationFailure(
                        error_type="ineligible_source_evidence",
                        message=msg,
                        stage="support_evidence_validation",
                        claim_id=claim_id,
                    )
                )

        diagnostics: Dict[str, Any] = {
            "raw_response": raw_text_record,
            "parsed_verdict": decision.verdict.value,
            "parsed_confidence": decision.confidence,
            "evidence_ids_cited": cited_ids,
            "evidence_grounding_valid": True,
        }

        return decision, diagnostics

    def _extract_json_substring(self, text: str) -> str:
        """Strip markdown code blocks or surrounding text to isolate the JSON object."""
        cleaned = text.strip()

        # Handle ```json ... ``` or ``` ... ```
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Search for first '{' and matching last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned
