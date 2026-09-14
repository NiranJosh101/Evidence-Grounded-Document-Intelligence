from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an extracted unit of text with provenance metadata."""

    chunk_id: str = Field(
        ...,
        description="Unique hash or ID for the chunk (e.g. doc123_c001)"
    )

    doc_id: str = Field(
        ...,
        description="Parent document identifier"
    )

    text: str = Field(
        ...,
        description="Raw text content of the chunk"
    )

    page_number: int = Field(
        ...,
        description="1-indexed source page number"
    )

    token_count: int = Field(
        ...,
        description="Total token count for this chunk"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as section titles"
    )


class TargetSchema(BaseModel):
    """
    Defines the target information to extract from a corporate
    financial/earnings report and the field-aware query buckets
    used to retrieve supporting evidence.
    """

    schema_name: str = Field(
        default="corporate_financial_report_v1",
        description="Name and version of the target extraction schema"
    )

    field_buckets: Dict[str, List[str]] = Field(
        default_factory=lambda: {

            "company_name": [
                "company name",
                "name of the company",
                "who is the company",
                "company profile"
            ],

            "reporting_period": [
                "reporting period",
                "fiscal year",
                "fiscal quarter",
                "period ended",
                "quarter ended",
                "year ended"
            ],

            "revenue": [
                "revenue",
                "total revenue",
                "net revenue",
                "sales",
                "revenue for the period"
            ],

            "revenue_growth": [
                "revenue growth",
                "revenue increased",
                "revenue decreased",
                "year over year revenue",
                "revenue growth rate",
                "change in revenue"
            ],

            "profitability": [
                "net income",
                "operating income",
                "gross profit",
                "profit margin",
                "operating margin",
                "profitability"
            ],

            "growth_drivers": [
                "reasons for revenue growth",
                "drivers of growth",
                "growth drivers",
                "what drove revenue growth",
                "factors contributing to growth",
                "reasons for the increase"
            ],

            "market_conditions": [
                "market conditions",
                "industry conditions",
                "market environment",
                "economic environment",
                "industry trends",
                "market trends"
            ],

            "management_outlook": [
                "management outlook",
                "future outlook",
                "business outlook",
                "forward outlook",
                "expected performance",
                "expectations for the future",
                "guidance"
            ],

            "risks": [
                "risk factors",
                "risks",
                "key risks",
                "business risks",
                "market risks",
                "challenges",
                "uncertainties"
            ],

            "guidance": [
                "financial guidance",
                "revenue guidance",
                "earnings guidance",
                "expected revenue",
                "expected earnings",
                "forecast",
                "guidance for the next period"
            ],

            "material_changes": [
                "material changes",
                "significant changes",
                "year over year changes",
                "quarter over quarter changes",
                "major developments",
                "significant developments"
            ]
        },
        description=(
            "Maps each target field to domain-specific query variations "
            "used for field-aware evidence retrieval."
        )
    )


class RetrievedEvidence(BaseModel):
    """Links extracted evidence chunks back to specific target fields."""

    field_name: str

    query_used: str

    chunks: List[DocumentChunk]

    similarity_scores: Optional[List[float]] = None