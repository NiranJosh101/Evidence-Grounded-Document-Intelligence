from typing import Dict, Tuple

from src.models import TargetSchema, DocumentExtractionResult, RetrievedEvidence
from src.retriever import FieldAwareRetriever
from src.interpreter import EvidenceInterpreter
from runtime_validator.validator import RuntimeValidator, ValidationResult


class IngestionPipeline:
    """
    Orchestrates Field-Aware Retrieval, LLM Interpretation,
    Runtime Validation, and retries.
    """

    def __init__(
        self,
        retriever: FieldAwareRetriever,
        interpreter: EvidenceInterpreter,
        validator: RuntimeValidator,
        max_retries: int = 2,
    ):
        self.retriever = retriever
        self.interpreter = interpreter
        self.validator = validator
        self.max_retries = max_retries

    def process_document(
        self,
        doc_id: str,
        namespace: str,
        schema: TargetSchema,
    ) -> Tuple[DocumentExtractionResult, ValidationResult, int]:
        """
        Runs the complete end-to-end processing pipeline
        for a single document.
        """

        # Step 1: Field-Aware Retrieval
        retrieved_evidence: Dict[str, RetrievedEvidence] = (
            self.retriever.retrieve_for_schema(
                schema=schema,
                namespace=namespace,
            )
        )

        retry_count = 0
        retry_prompt = ""

        while retry_count <= self.max_retries:

            # Step 2: LLM Interpretation
            extraction_result = self.interpreter.interpret_document(
                doc_id=doc_id,
                namespace=namespace,
                retrieved_evidence=retrieved_evidence,
                retry_prompt=retry_prompt,
            )

            # Step 3: Runtime Validation
            val_result = self.validator.validate(
                extraction_result=extraction_result,
                retrieved_evidence=retrieved_evidence,
            )

            # Validation passed
            if val_result.is_valid:
                return extraction_result, val_result, retry_count

            # Validation failed
            retry_count += 1

            # Give the next LLM attempt the validator's feedback
            retry_prompt = val_result.retry_prompt

        # Retries exhausted
        return extraction_result, val_result, retry_count