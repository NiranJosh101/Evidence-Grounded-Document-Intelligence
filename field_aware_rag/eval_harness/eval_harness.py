import json
from typing import List, Dict, Any
from pydantic import BaseModel
from src.models import TargetSchema, DocumentExtractionResult, RetrievedEvidence
from src.pipeline import IngestionPipeline


class EvaluationMetrics(BaseModel):
    retrieval_recall_at_k: float  # Fraction of ground truth evidence text retrieved
    extraction_accuracy: float   # Correct values ratio
    nuance_preservation_score: float  # Preservation of keywords/qualifiers
    overall_score: float


class EvaluationHarness:
    """Runs system benchmarks against a curated Golden Dataset."""

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def run_eval(
        self,
        golden_dataset_path: str,
        schema: TargetSchema
    ) -> EvaluationMetrics:
        with open(golden_dataset_path, "r") as f:
            golden_data = json.load(f)

        total_fields = 0
        retrieved_gt_count = 0
        accurate_extractions = 0
        nuance_matched_count = 0

        for item in golden_data:
            doc_id = item["doc_id"]
            namespace = f"eval_{doc_id}"
            expected_extractions = item["expected_extractions"]

            # Process Golden Document through Ingestion Pipeline
            extraction_result, val_result, _ = self.pipeline.process_document(
                doc_id=doc_id,
                namespace=namespace,
                schema=schema
            )

            # Retrieve evidence map generated during pipeline processing
            retrieved_evidence = self.pipeline.retriever.retrieve_for_schema(schema)

            for field_name, gt_data in expected_extractions.items():
                total_fields += 1
                gt_texts = [gt.lower() for gt in gt_data["ground_truth_chunk_texts"]]

                # 1. Retrieval Recall@K Calculation
                evidence = retrieved_evidence.get(field_name)
                if evidence and evidence.chunks:
                    retrieved_combined_text = " ".join([c.text.lower() for c in evidence.chunks])
                    # Check if ground truth sentences exist in retrieved text
                    matches = sum(1 for gt in gt_texts if any(word in retrieved_combined_text for word in gt.split()[:5]))
                    if matches > 0:
                        retrieved_gt_count += 1

                # 2. Extraction Accuracy & Nuance Scoring
                extracted_res = extraction_result.extractions.get(field_name)
                if extracted_res and extracted_res.value:
                    exp_val = str(gt_data["expected_value"]).lower()
                    act_val = str(extracted_res.value).lower()

                    # Exact value or key string containment match
                    if exp_val in act_val or act_val in exp_val:
                        accurate_extractions += 1

                    # Qualifier/Nuance Keyword Check
                    qualifier_str = str(extracted_res.qualifiers).lower() if extracted_res.qualifiers else ""
                    kw_matches = sum(
                        1 for kw in gt_data.get("expected_qualifier_keywords", [])
                        if kw.lower() in qualifier_str or kw.lower() in act_val
                    )
                    total_kws = len(gt_data.get("expected_qualifier_keywords", []))
                    if total_kws > 0 and (kw_matches / total_kws) >= 0.5:
                        nuance_matched_count += 1

        recall = (retrieved_gt_count / total_fields) if total_fields else 0.0
        accuracy = (accurate_extractions / total_fields) if total_fields else 0.0
        nuance_score = (nuance_matched_count / total_fields) if total_fields else 0.0
        overall = (recall * 0.4) + (accuracy * 0.4) + (nuance_score * 0.2)

        return EvaluationMetrics(
            retrieval_recall_at_k=round(recall, 3),
            extraction_accuracy=round(accuracy, 3),
            nuance_preservation_score=round(nuance_score, 3),
            overall_score=round(overall, 3)
        )