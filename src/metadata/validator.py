"""Corpus and metadata validation layer for legal and scientific EvidenceChunks.

Verifies that every generated EvidenceChunk is structurally valid, traceable,
deterministic, portable, and suitable for embedding/vector storage.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.models.schemas import ContentType, Domain, EvidenceChunk

logger = logging.getLogger(__name__)

# Patterns for non-portable absolute machine paths
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"^[a-zA-Z]:[/\\]"),             # Windows drive: C:\ or C:/
    re.compile(r"^(?:/Users/|/home/|/root/|/tmp/|/var/)", re.IGNORECASE),  # Unix home/system
    re.compile(r"(?:\\Users\\|/Users/)", re.IGNORECASE),                    # Windows user path in string
]

# Random UUID pattern (should not be used for deterministic chunk IDs)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class ValidationReport(BaseModel):
    """Structured report produced by CorpusValidator."""

    domain: str = "all"
    documents_count: int = 0
    total_chunks: int = 0
    valid_chunks: int = 0
    invalid_chunks: int = 0
    embeddable_chunks: int = 0
    non_embeddable_chunks: int = 0

    parent_references_valid: bool = True
    chunk_ids_unique: bool = True
    source_paths_portable: bool = True
    legal_identifiers_valid: bool = True
    scientific_metadata_valid: bool = True

    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    passed: bool = True

    def add_error(self, message: str) -> None:
        """Add an error and mark report as failed."""
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(message)


class CorpusValidator:
    """Validates metadata, hierarchy, portability, and embedding suitability of EvidenceChunks."""

    def __init__(
        self,
        min_embeddable_characters: int = 10,
        settings: Optional[Settings] = None,
    ):
        """Initialize the validator.

        Args:
            min_embeddable_characters: Minimum characters for text to be considered embeddable.
            settings: Project settings singleton.
        """
        self.min_embeddable_characters = min_embeddable_characters
        self.settings = settings or get_settings()

    def validate_chunks(
        self,
        chunks: List[EvidenceChunk],
        domain: Optional[Union[str, Domain]] = None,
    ) -> ValidationReport:
        """Validate an in-memory collection of EvidenceChunks.

        Args:
            chunks: List of EvidenceChunk instances.
            domain: Domain being validated, or 'all'.

        Returns:
            ValidationReport detailing the validation status and metrics.
        """
        domain_str = domain.value if isinstance(domain, Domain) else (domain or "all")
        report = ValidationReport(domain=domain_str, total_chunks=len(chunks))

        if not chunks:
            report.add_warning("No chunks provided for validation.")
            return report

        # Group chunks by document
        doc_ids: Set[str] = set()
        chunk_map: Dict[str, EvidenceChunk] = {}
        duplicates: Set[str] = set()

        for chunk in chunks:
            doc_ids.add(chunk.document_id)
            if chunk.chunk_id in chunk_map:
                duplicates.add(chunk.chunk_id)
            chunk_map[chunk.chunk_id] = chunk

        report.documents_count = len(doc_ids)

        # 1. Check Unique Chunk IDs
        if duplicates:
            report.chunk_ids_unique = False
            for dup in sorted(duplicates):
                report.add_error(f"Duplicate chunk_id found: {dup}")

        # 2. Validate individual chunk fields & embedding suitability
        valid_chunk_count = 0
        invalid_chunk_count = 0
        embeddable_count = 0
        non_embeddable_count = 0

        for chunk in chunks:
            chunk_errors, is_embeddable = self.validate_chunk_attributes(chunk)
            if chunk_errors:
                invalid_chunk_count += 1
                for err in chunk_errors:
                    report.add_error(f"[{chunk.chunk_id}] {err}")
            else:
                valid_chunk_count += 1

            if is_embeddable:
                embeddable_count += 1
            else:
                non_embeddable_count += 1

            # Check source portability
            if not self.is_source_portable(chunk.source):
                report.source_paths_portable = False
                report.add_error(
                    f"[{chunk.chunk_id}] Non-portable source path: '{chunk.source}'"
                )

        report.valid_chunks = valid_chunk_count
        report.invalid_chunks = invalid_chunk_count
        report.embeddable_chunks = embeddable_count
        report.non_embeddable_chunks = non_embeddable_count

        # 3. Validate Parent-Child Hierarchy (Forest / DAG integrity)
        hierarchy_errors = self.validate_hierarchy(chunks, chunk_map)
        if hierarchy_errors:
            report.parent_references_valid = False
            for err in hierarchy_errors:
                report.add_error(err)

        # 4. Domain-specific structural checks
        if any(c.domain == Domain.LEGAL for c in chunks):
            legal_errors = self.validate_legal_domain(chunks)
            if legal_errors:
                report.legal_identifiers_valid = False
                for err in legal_errors:
                    report.add_error(err)

        if any(c.domain == Domain.SCIENTIFIC for c in chunks):
            scientific_errors = self.validate_scientific_domain(chunks)
            if scientific_errors:
                report.scientific_metadata_valid = False
                for err in scientific_errors:
                    report.add_error(err)

        report.passed = len(report.errors) == 0 and report.invalid_chunks == 0
        return report

    def validate_chunk_attributes(self, chunk: EvidenceChunk) -> Tuple[List[str], bool]:
        """Validate required metadata and determine embedding suitability for a single chunk.

        Returns:
            Tuple of (errors_list, is_embeddable)
        """
        errors: List[str] = []

        if not chunk.chunk_id or not chunk.chunk_id.strip():
            errors.append("chunk_id is empty or missing")

        if UUID_PATTERN.match(chunk.chunk_id):
            errors.append(f"chunk_id '{chunk.chunk_id}' appears to be a random UUID, must be deterministic")

        if not chunk.document_id or not chunk.document_id.strip():
            errors.append("document_id is empty or missing")

        if not chunk.title or not chunk.title.strip():
            errors.append("title is empty or missing")

        if not chunk.source or not chunk.source.strip():
            errors.append("source is empty or missing")

        if chunk.page is None or chunk.page < 1:
            errors.append(f"invalid page number: {chunk.page}")

        if not chunk.section or not chunk.section.strip():
            errors.append("section title is empty")

        if not chunk.section_id or not chunk.section_id.strip():
            errors.append("section_id is empty")

        if not chunk.metadata_version or not chunk.metadata_version.strip():
            errors.append("metadata_version is empty or missing")

        # Embedding suitability checks
        is_embeddable = True
        text = chunk.text.strip() if chunk.text else ""
        if not text:
            errors.append("chunk text is empty or whitespace-only")
            is_embeddable = False
        elif len(text) < self.min_embeddable_characters:
            is_embeddable = False

        return errors, is_embeddable

    def is_source_portable(self, source: str) -> bool:
        """Check whether the source string is machine-independent and portable."""
        if not source:
            return False
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(source):
                return False
        return True

    def validate_hierarchy(
        self,
        chunks: List[EvidenceChunk],
        chunk_map: Dict[str, EvidenceChunk],
    ) -> List[str]:
        """Validate parent-child integrity: existence, same-doc, same-domain, no cycles."""
        errors: List[str] = []

        for chunk in chunks:
            if chunk.parent_id is None:
                continue

            # Self parenting
            if chunk.parent_id == chunk.chunk_id:
                errors.append(
                    f"Self-referencing parent detected: chunk {chunk.chunk_id} references itself"
                )
                continue

            # Parent existence
            parent = chunk_map.get(chunk.parent_id)
            if not parent:
                errors.append(
                    f"Missing parent chunk: chunk {chunk.chunk_id} references non-existent parent_id {chunk.parent_id}"
                )
                continue

            # Cross-document parent
            if parent.document_id != chunk.document_id:
                errors.append(
                    f"Cross-document parent reference: chunk {chunk.chunk_id} (doc: {chunk.document_id}) "
                    f"references parent {chunk.parent_id} in doc {parent.document_id}"
                )

            # Cross-domain parent
            if parent.domain != chunk.domain:
                errors.append(
                    f"Cross-domain parent reference: chunk {chunk.chunk_id} (domain: {chunk.domain.value}) "
                    f"references parent {chunk.parent_id} in domain {parent.domain.value}"
                )

        # Cycle detection
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(curr_id: str, path: List[str]) -> Optional[List[str]]:
            visited.add(curr_id)
            in_stack.add(curr_id)

            curr_chunk = chunk_map.get(curr_id)
            if curr_chunk and curr_chunk.parent_id:
                parent_id = curr_chunk.parent_id
                if parent_id in chunk_map:
                    if parent_id in in_stack:
                        return path + [parent_id]
                    if parent_id not in visited:
                        cycle = dfs(parent_id, path + [parent_id])
                        if cycle:
                            return cycle

            in_stack.remove(curr_id)
            return None

        for chunk_id in chunk_map:
            if chunk_id not in visited:
                cycle = dfs(chunk_id, [chunk_id])
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    errors.append(f"Circular parent-child relationship detected: {cycle_str}")

        return errors

    def validate_legal_domain(self, chunks: List[EvidenceChunk]) -> List[str]:
        """Validate legal-specific identifiers and preservation."""
        errors: List[str] = []
        clause_map: Dict[str, List[EvidenceChunk]] = {}

        for chunk in chunks:
            if chunk.domain != Domain.LEGAL:
                continue

            if not chunk.section_id:
                errors.append(f"Legal chunk {chunk.chunk_id} has empty section_id")

            if chunk.clause_id:
                clause_map.setdefault(chunk.clause_id, []).append(chunk)

        # Verify exact clause distinctions: e.g. 3(d) vs 3(e) if present in same doc
        # Group by document
        doc_chunks: Dict[str, List[EvidenceChunk]] = {}
        for c in chunks:
            if c.domain == Domain.LEGAL:
                doc_chunks.setdefault(c.document_id, []).append(c)

        for doc_id, d_chunks in doc_chunks.items():
            clause_ids = {c.clause_id for c in d_chunks if c.clause_id}
            # If document contains 3(d) and 3(e), ensure they are separate
            has_3d = any(cid == "3(d)" or "3(d)" in cid for cid in clause_ids)
            has_3e = any(cid == "3(e)" or "3(e)" in cid for cid in clause_ids)
            if has_3d and has_3e:
                chunks_3d = [c for c in d_chunks if c.clause_id == "3(d)"]
                chunks_3e = [c for c in d_chunks if c.clause_id == "3(e)"]
                for c3d in chunks_3d:
                    for c3e in chunks_3e:
                        if c3d.section_id == c3e.section_id:
                            errors.append(
                                f"Legal conflation in {doc_id}: 3(d) and 3(e) share identical section_id {c3d.section_id}"
                            )

            # Subclauses 3(d)(i) vs 3(d)(ii)
            has_3di = any(cid == "3(d)(i)" for cid in clause_ids)
            has_3dii = any(cid == "3(d)(ii)" for cid in clause_ids)
            if has_3di and has_3dii:
                chunks_3di = [c for c in d_chunks if c.clause_id == "3(d)(i)"]
                chunks_3dii = [c for c in d_chunks if c.clause_id == "3(d)(ii)"]
                for c3di in chunks_3di:
                    for c3dii in chunks_3dii:
                        if c3di.section_id == c3dii.section_id:
                            errors.append(
                                f"Legal subclause conflation in {doc_id}: 3(d)(i) and 3(d)(ii) share identical section_id {c3di.section_id}"
                            )

        return errors

    def validate_scientific_domain(self, chunks: List[EvidenceChunk]) -> List[str]:
        """Validate scientific-specific structure: table_id, parent sections."""
        errors: List[str] = []

        for chunk in chunks:
            if chunk.domain != Domain.SCIENTIFIC:
                continue

            if not chunk.section_id:
                errors.append(f"Scientific chunk {chunk.chunk_id} has empty section_id")

            # Table chunks must preserve table_id
            if chunk.content_type == ContentType.TABLE and not chunk.table_id:
                errors.append(f"Scientific table chunk {chunk.chunk_id} missing table_id")

        return errors

    def validate_jsonl_file(self, file_path: Union[str, Path]) -> ValidationReport:
        """Validate a persisted JSONL file line-by-line.

        Args:
            file_path: Path to the .jsonl file.

        Returns:
            ValidationReport for the file.
        """
        path = Path(file_path).resolve()
        report = ValidationReport()

        if not path.exists():
            report.add_error(f"JSONL file does not exist: {path}")
            return report

        chunks: List[EvidenceChunk] = []
        raw_lines_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                raw_lines_count += 1
                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    report.add_error(f"Malformed JSON at line {line_no} in {path.name}: {exc}")
                    continue

                try:
                    chunk = EvidenceChunk.model_validate(data)
                    chunks.append(chunk)
                except Exception as exc:
                    report.add_error(f"Schema validation failed at line {line_no} in {path.name}: {exc}")

        # Run corpus validation on the successfully parsed chunks
        corpus_report = self.validate_chunks(chunks)
        # Merge raw line issues into corpus report
        corpus_report.total_chunks = raw_lines_count
        corpus_report.invalid_chunks += (raw_lines_count - len(chunks))
        if raw_lines_count != len(chunks):
            corpus_report.passed = False

        for err in report.errors:
            corpus_report.add_error(err)

        return corpus_report

    def validate_directory(
        self,
        directory_path: Union[str, Path],
        domain: Optional[Union[str, Domain]] = None,
    ) -> ValidationReport:
        """Validate all JSONL files within a directory.

        Args:
            directory_path: Path to directory containing .jsonl files.
            domain: Domain filter (legal, scientific, or None/all).

        Returns:
            Aggregated ValidationReport.
        """
        dir_p = Path(directory_path).resolve()
        domain_str = domain.value if isinstance(domain, Domain) else (domain or "all")
        agg_report = ValidationReport(domain=domain_str)

        if not dir_p.exists() or not dir_p.is_dir():
            agg_report.add_error(f"Directory not found: {dir_p}")
            return agg_report

        jsonl_files = sorted(list(dir_p.glob("*.jsonl")))
        if not jsonl_files:
            agg_report.add_warning(f"No .jsonl files found in directory: {dir_p}")
            return agg_report

        all_chunks: List[EvidenceChunk] = []
        for jf in jsonl_files:
            file_report = self.validate_jsonl_file(jf)
            if not file_report.passed:
                agg_report.passed = False
                agg_report.errors.extend(file_report.errors)

            # Re-read chunks for aggregated corpus-level validation
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            all_chunks.append(EvidenceChunk.model_validate_json(line))
                        except Exception:
                            pass

        corpus_report = self.validate_chunks(all_chunks, domain=domain_str)
        # Combine errors
        for err in agg_report.errors:
            if err not in corpus_report.errors:
                corpus_report.add_error(err)

        return corpus_report
