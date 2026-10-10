from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    content_type: str | None = Field(None, description="Filter by content type: article, quote, poem")
    use_reranker: bool = True
    use_mmr: bool = True


class SourceDocument(BaseModel):
    title: str
    url: str
    text: str
    content_type: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    is_on_topic: bool = True
    provider: str = ""


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    content_type: str | None = None
    top_k: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[SourceDocument]
    query: str
    total: int


class HealthResponse(BaseModel):
    status: str
    qdrant: str = "unknown"
    llm_provider: str = ""
    collection: str = ""
    points: int | None = None
    reranker: str = ""
    jev: str = "disabled"  # ok | unavailable | disabled


class TopicsResponse(BaseModel):
    topics: list[str]
    total: int
