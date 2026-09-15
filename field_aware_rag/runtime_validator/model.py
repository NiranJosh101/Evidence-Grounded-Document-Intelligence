from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from src.models import DocumentExtractionResult, RetrievedEvidence, FieldExtractionResult


class ValidationIssue(BaseModel):
    field_name: str
    issue_type: str = Field(..., description="'MISSING_GROUNDING', 'LOW_CONFIDENCE', or 'SCHEMA_ERROR'")
    description: str


class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    retry_prompt: str = ""