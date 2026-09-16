from typing import Dict, Any
import json

from openrouter import OpenRouter

from field_aware_rag.config import settings
from field_aware_rag.src.models import (
    RetrievedEvidence,
    FieldExtractionResult,
    DocumentExtractionResult,
)
from field_aware_rag.src.prompts import (
    INTERPRETER_SYSTEM_PROMPT,
    FIELD_EXTRACTION_PROMPT,
)


class EvidenceInterpreter:
    """
    Uses OpenRouter to interpret field-specific evidence
    and produce structured extraction results.

    This component is responsible only for interpretation.
    Validation and retry logic are handled downstream.
    """

    def __init__(
        self,
        api_key: str = settings.OPENROUTER_API_KEY,
        model: str = settings.DEFAULT_MODEL,
    ):
        self.client = OpenRouter(api_key=api_key)
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
        Interprets evidence for a single target field using OpenRouter.
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

        # Convert the Pydantic model into JSON Schema
        schema = FieldExtractionResult.model_json_schema()

        # OpenRouter structured-output request
        response = self.client.chat.send(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": INTERPRETER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "field_extraction_result",
                    "strict": True,
                    "schema": schema,
                },
            },
            stream=False,
        )

        # Get the model's JSON response
        content = response.choices[0].message.content

        print("\n===== OPENROUTER RESPONSE =====")
        print("MODEL:", self.model)
        print("CONTENT:", repr(content))
        print("RESPONSE:", response)
        print("===============================\n")

        if not content:
            raise ValueError(
                f"OpenRouter returned an empty response for field '{field_name}'"
            )

        parsed_output = json.loads(content)

        # Validate against our Pydantic model
        extraction = FieldExtractionResult.model_validate(parsed_output)

        # Attach original chunk IDs for provenance traceability
        extraction.cited_chunk_ids = [
            chunk.chunk_id
            for chunk in evidence.chunks
        ]

        return extraction