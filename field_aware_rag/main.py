import os

from src.ingest import PDFIngestor
from src.models import (
    TargetSchema,
    DocumentChunk,
)
from src.vector_store import VectorStore
from src.retriever import FieldAwareRetriever
from src.interpreter import EvidenceInterpreter


if __name__ == "__main__":

    # =========================================================
    # 1. Load PDF and extract document chunks
    # =========================================================

    pdf_path = "sample.pdf"
    doc_id = "doc_test_001"
    analysis_id = "analysis_001"

    # Mock chunks used when no PDF is available.
    mock_chunks = [
        DocumentChunk(
            chunk_id="chk_001",
            doc_id=doc_id,
            text=(
                "In fiscal year 2024, our revenue increased by 35% "
                "year-over-year, driven largely by new enterprise "
                "contract wins."
            ),
            page_number=1,
            token_count=22,
        ),
        DocumentChunk(
            chunk_id="chk_002",
            doc_id=doc_id,
            text=(
                "Key operational risks include regulatory compliance "
                "costs and supply chain volatility in Asia-Pacific markets."
            ),
            page_number=2,
            token_count=19,
        ),
        DocumentChunk(
            chunk_id="chk_003",
            doc_id=doc_id,
            text=(
                "Total addressable market for automated document "
                "intelligence is projected to reach $12 billion by 2028."
            ),
            page_number=3,
            token_count=20,
        ),
    ]

    if os.path.exists(pdf_path):

        print(f"\nLoading PDF: {pdf_path}")

        ingestor = PDFIngestor(
            chunk_size=300,
            chunk_overlap=40,
        )

        chunks = ingestor.extract_and_chunk(
            pdf_path=pdf_path,
            doc_id=doc_id,
        )

        print(f"Loaded {len(chunks)} chunks.")

    else:

        chunks = mock_chunks

        print(
            "\nPDF not found. "
            "Using mock document chunks for verification."
        )

    # =========================================================
    # 2. Initialize Pinecone-backed vector store
    # =========================================================

    print("\nInitializing vector store...")

    vector_store = VectorStore()

    # =========================================================
    # 3. Index document chunks
    # =========================================================

    print(
        f"Indexing {len(chunks)} chunks "
        f"into Pinecone namespace '{analysis_id}'..."
    )

    vector_store.index_chunks(
        chunks=chunks,
        namespace=analysis_id,
    )

    print("Indexing complete.")

    # =========================================================
    # 4. Define target extraction fields and query buckets
    # =========================================================

    schema = TargetSchema(
        schema_name="corporate_financial_report_v1",
        field_buckets={

            "company_name": [
                "company name",
                "name of company",
                "company profile",
            ],

            "reporting_period": [
                "reporting period",
                "fiscal year",
                "quarter ended",
                "year ended",
            ],

            "revenue": [
                "total revenue",
                "net revenue",
                "sales",
                "revenue for the period",
            ],

            "revenue_growth": [
                "revenue growth",
                "revenue growth rate",
                "year over year revenue",
                "revenue increased",
                "change in revenue",
            ],

            "growth_drivers": [
                "drivers of revenue growth",
                "reasons for revenue growth",
                "what drove revenue growth",
                "factors contributing to growth",
            ],

            "profitability": [
                "net income",
                "operating income",
                "gross profit",
                "profit margin",
                "operating margin",
            ],

            "management_outlook": [
                "management outlook",
                "future business outlook",
                "forward outlook",
                "expected performance",
                "management expectations",
            ],

            "risks": [
                "risk factors",
                "key risks",
                "business risks",
                "operational challenges",
                "regulatory risks",
            ],

            "guidance": [
                "financial guidance",
                "revenue guidance",
                "earnings guidance",
                "forecast",
                "future guidance",
            ],
        },
    )

    # =========================================================
    # 5. Run field-aware retrieval
    # =========================================================

    print("\nRunning field-aware retrieval...")

    retriever = FieldAwareRetriever(
        vector_store=vector_store,
    )

    evidence_map = retriever.retrieve_for_schema(
        schema=schema,
        namespace=analysis_id,
    )

    # =========================================================
    # 6. Display retrieved evidence
    # =========================================================

    print("\n--- FIELD-AWARE RETRIEVAL RESULTS ---")

    for field, evidence in evidence_map.items():

        print(f"\n[Field: '{field}']")

        print(
            f"  Queries Executed: "
            f"{evidence.query_used}"
        )

        print(
            f"  Retrieved Chunks: "
            f"{len(evidence.chunks)}"
        )

        for idx, (
            chunk,
            score,
        ) in enumerate(
            zip(
                evidence.chunks,
                evidence.similarity_scores or [],
            )
        ):

            print(
                f"    Chunk {idx + 1} "
                f"(Page {chunk.page_number}, "
                f"Score: {score:.3f}): "
                f"{chunk.text[:120]}..."
            )

    # =========================================================
    # 7. Initialize Claude Evidence Interpreter
    # =========================================================

    print(
        "\nInitializing Evidence Interpreter "
        "with Anthropic Claude..."
    )

    interpreter = EvidenceInterpreter()

    # =========================================================
    # 8. Interpret retrieved evidence into structured output
    # =========================================================

    print("\nRunning evidence interpretation...")

    extraction_result = interpreter.interpret_document(
        doc_id=doc_id,
        namespace=analysis_id,
        retrieved_evidence=evidence_map,
    )

    # =========================================================
    # 9. Display structured extraction results
    # =========================================================

    print("\n--- PHASE 3 EXTRACTION OUTPUT ---")

    for field_name, extraction in (
        extraction_result.extractions.items()
    ):

        print(f"\n[Field: '{field_name}']")
        print(f"  Value: {extraction.value}")
        print(f"  Qualifiers: {extraction.qualifiers}")
        print(f"  Confidence: {extraction.confidence}")
        print(
            f"  Cited Chunks: "
            f"{extraction.cited_chunk_ids}"
        )

    print(
        "\nComplete pipeline executed successfully."
    )

