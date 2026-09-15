from typing import List, Dict, Any, Tuple
from src.models import DocumentExtractionResult, RetrievedEvidence, FieldExtractionResult
from runtime_validator.model import ValidationResult, ValidationIssue

class RuntimeValidator:
    """Validates individual document extractions against source evidence chunks."""

    def __init__(self, require_citations: bool = True):
        self.require_citations = require_citations

    def validate(
        self,
        extraction_result: DocumentExtractionResult,
        retrieved_evidence: Dict[str, RetrievedEvidence]
    ) -> ValidationResult:
        issues: List[ValidationIssue] = []

        for field_name, field_res in extraction_result.extractions.items():
            evidence = retrieved_evidence.get(field_name)

            # Rule 1: Confidence Level Check
            if field_res.confidence in ["LOW", "UNSUPPORTED"] and field_res.value is not None:
                issues.append(
                    ValidationIssue(
                        field_name=field_name,
                        issue_type="LOW_CONFIDENCE",
                        description=f"Field '{field_name}' returned value but has '{field_res.confidence}' confidence."
                    )
                )

            # Rule 2: Citation Verification
            if self.require_citations and field_res.value is not None:
                if not field_res.cited_chunk_ids:
                    issues.append(
                        ValidationIssue(
                            field_name=field_name,
                            issue_type="MISSING_GROUNDING",
                            description=f"Field '{field_name}' extracted value without citing evidence chunk IDs."
                        )
                    )
                else:
                    # Verify cited chunk IDs exist in retrieved evidence
                    available_ids = {c.chunk_id for c in evidence.chunks} if evidence else set()
                    invalid_ids = [cid for cid in field_res.cited_chunk_ids if cid not in available_ids]
                    if invalid_ids:
                        issues.append(
                            ValidationIssue(
                                field_name=field_name,
                                issue_type="MISSING_GROUNDING",
                                description=f"Field '{field_name}' cited chunk IDs {invalid_ids} which do not exist in retrieved evidence."
                            )
                        )

            # Rule 3: Direct Grounding Overlap (Deterministic check against source text)
            if field_res.value and evidence and evidence.chunks:
                source_text = " ".join([c.text.lower() for c in evidence.chunks])
                value_str = str(field_res.value).lower()
                
                # Check key token overlap for non-numeric extractions
                value_tokens = [t for t in value_str.split() if len(t) > 3]
                if value_tokens:
                    overlap_count = sum(1 for token in value_tokens if token in source_text)
                    overlap_ratio = overlap_count / len(value_tokens)
                    if overlap_ratio < 0.3:  # Less than 30% word match
                        issues.append(
                            ValidationIssue(
                                field_name=field_name,
                                issue_type="MISSING_GROUNDING",
                                description=f"Extracted value '{field_res.value}' lacks sufficient word overlap with retrieved evidence chunks."
                            )
                        )

        is_valid = len(issues) == 0
        retry_prompt = self._build_retry_prompt(issues) if not is_valid else ""

        return ValidationResult(is_valid=is_valid, issues=issues, retry_prompt=retry_prompt)

    def _build_retry_prompt(self, issues: List[ValidationIssue]) -> str:
        prompt = "Validation failed for the previous extraction. Please correct the following issues:\n"
        for idx, issue in enumerate(issues, 1):
            prompt += f"{idx}. Field [{issue.field_name}] - {issue.issue_type}: {issue.description}\n"
        prompt += "\nRe-evaluate the evidence chunks strictly. If the information is not directly backed by a chunk, mark value as null and confidence as UNSUPPORTED."
        return prompt