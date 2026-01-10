from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is RAG?")
    k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")

class DocumentSchema(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[DocumentSchema]

class AdvancedQueryResponse(QueryResponse):
    needs_retrieval: bool
    confidence: float

class HealthResponse(BaseModel):
    status: str
    version: str
