import os

from src.ingest import PDFIngestor
from src.models import TargetSchema


if __name__ == "__main__":
    # 1. Instantiate the PDF ingestor
    ingestor = PDFIngestor(
        chunk_size=300,
        chunk_overlap=40
    )

    # 2. Define the extraction target for corporate financial reports.
    #
    #    Each field has its own query bucket so the retrieval stage
    #    can search specifically for the evidence required to populate
    #    that field.
    sample_schema = TargetSchema(
        schema_name="corporate_financial_report_v1",
        field_buckets={
            "company_name": [
                "company name",
                "name of the company",
                "who is the company",
                "company profile",
            ],

            "reporting_period": [
                "reporting period",
                "fiscal year",
                "fiscal quarter",
                "period ended",
                "quarter ended",
                "year ended",
            ],

            "revenue": [
                "revenue",
                "total revenue",
                "net revenue",
                "sales",
                "revenue for the period",
            ],

            "revenue_growth": [
                "revenue growth",
                "revenue increased",
                "revenue decreased",
                "year over year revenue",
                "revenue growth rate",
                "change in revenue",
            ],

            "profitability": [
                "net income",
                "operating income",
                "gross profit",
                "profit margin",
                "operating margin",
                "profitability",
            ],

            "growth_drivers": [
                "reasons for revenue growth",
                "drivers of growth",
                "growth drivers",
                "what drove revenue growth",
                "factors contributing to growth",
                "reasons for the increase",
            ],

            "market_conditions": [
                "market conditions",
                "industry conditions",
                "market environment",
                "economic environment",
                "industry trends",
                "market trends",
            ],

            "management_outlook": [
                "management outlook",
                "future outlook",
                "business outlook",
                "forward outlook",
                "expected performance",
                "expectations for the future",
                "guidance",
            ],

            "risks": [
                "risk factors",
                "risks",
                "key risks",
                "business risks",
                "market risks",
                "challenges",
                "uncertainties",
            ],

            "guidance": [
                "financial guidance",
                "revenue guidance",
                "earnings guidance",
                "expected revenue",
                "expected earnings",
                "forecast",
                "guidance for the next period",
            ],

            "material_changes": [
                "material changes",
                "significant changes",
                "year over year changes",
                "quarter over quarter changes",
                "major developments",
                "significant developments",
            ],
        },
    )

    print(f"Target Schema Loaded: {sample_schema.schema_name}")
    print(f"Target Fields: {list(sample_schema.field_buckets.keys())}")

    # 3. Parse and chunk the sample financial report.
    #
    #    Replace 'sample.pdf' with a real corporate financial/earnings
    #    report when testing the pipeline.
    pdf_path = "sample.pdf"

    if os.path.exists(pdf_path):
        chunks = ingestor.extract_and_chunk(
            pdf_path=pdf_path,
            doc_id="doc_test_001"
        )

        print(f"\nSuccessfully parsed '{pdf_path}':")
        print(f"Total Chunks Generated: {len(chunks)}")

        if chunks:
            print("\nSample Chunk 0 Metadata:")
            print(chunks[0].model_dump_json(indent=2))

    else:
        print(
            f"\nPlace a corporate financial report named "
            f"'{pdf_path}' in the project root to verify extraction."
        )
