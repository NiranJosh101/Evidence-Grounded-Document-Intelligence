from typing import Dict, Any

from anthropic import Anthropic

from config import settings
from src.models import (
    RetrievedEvidence,
    FieldExtractionResult,
    DocumentExtractionResult,
)
from src.prompts import (
    INTERPRETER_SYSTEM_PROMPT,
    FIELD_EXTRACTION_PROMPT,
)


class EvidenceInterpreter:
    """
    Uses Anthropic Claude to interpret field-specific evidence
    and produce structured extraction results.

    This component is responsible only for interpretation.
    Validation and retry logic are handled downstream.
    """

    def __init__(
        self,
        api_key: str = settings.ANTHROPIC_API_KEY,
        model: str = settings.CLAUDE_MODEL,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def interpret_document(
        self,
        doc_id: str,
        namespace: str,
        retrieved_evidence: Dict[str, RetrievedEvidence],
        retry_prompt: str = "",
    ) -> DocumentExtractionResult:
        """
        Interprets the retrieved evidence for every target field
        and compiles the results into a document-level extraction.
        """

        extractions: Dict[str, FieldExtractionResult] = {}
        raw_outputs: Dict[str, Any] = {}

        for field_name, evidence in retrieved_evidence.items():

            extraction = self._interpret_field(
                field_name=field_name,
                evidence=evidence,
                retry_prompt=retry_prompt,
            )

            extractions[field_name] = extraction
            raw_outputs[field_name] = extraction.model_dump()

        return DocumentExtractionResult(
            doc_id=doc_id,
            namespace=namespace,
            extractions=extractions,
            raw_json_output=raw_outputs,
        )

    def _interpret_field(
        self,
        field_name: str,
        evidence: RetrievedEvidence,
        retry_prompt: str = "",
    ) -> FieldExtractionResult:
        """
        Interprets evidence for a single target field.
        """

        if not evidence.chunks:
            return FieldExtractionResult(
                field_name=field_name,
                value=None,
                qualifiers="No evidence chunks were retrieved for this field.",
                confidence="UNSUPPORTED",
                cited_chunk_ids=[],
            )

        evidence_blocks = []

        for chunk in evidence.chunks:
            evidence_blocks.append(
                f"""
--- CHUNK ID: {chunk.chunk_id} | PAGE: {chunk.page_number} ---

{chunk.text}
""".strip()
            )

        formatted_evidence = "\n\n".join(evidence_blocks)

        user_prompt = FIELD_EXTRACTION_PROMPT.format(
            field_name=field_name,
            evidence_text=formatted_evidence,
        )

        # Add validator feedback when this is a retry
        if retry_prompt:
            user_prompt += f"""

--- VALIDATION FEEDBACK ---

{retry_prompt}
"""

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1000,
            temperature=0.0,
            system=INTERPRETER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            output_format=FieldExtractionResult,
        )

        extraction = response.parsed_output

        extraction.cited_chunk_ids = [
            chunk.chunk_id
            for chunk in evidence.chunks
        ]

        return extraction