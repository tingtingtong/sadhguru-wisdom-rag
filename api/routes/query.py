from __future__ import annotations

import asyncio

import logfire
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_provider
from api.schemas import QueryRequest, QueryResponse, SourceDocument
from generation.errors import BUSY_MESSAGE, LLMBusyError
from generation.prompts import build_no_context_response, build_system_prompt, build_user_message, is_no_answer
from generation.providers.base import LLMProvider
from guardrails.input_validator import validate_input
from guardrails.output_validator import validate_output
from guardrails.topic_classifier import classify_query
from retrieval.engine import retrieve
from retrieval.relevance import is_answerable

router = APIRouter(tags=["query"])


@router.post("/query")
async def query(
    request: QueryRequest,
    provider: LLMProvider = Depends(get_provider),
) -> QueryResponse:
    """Full RAG pipeline: guardrails -> retrieve -> generate."""

    with logfire.span("rag_pipeline", question=request.question):
        # Input validation
        is_valid, error = validate_input(request.question)
        if not is_valid:
            return QueryResponse(
                answer=error or "Invalid input.",
                sources=[],
                query=request.question,
                is_on_topic=False,
            )

        # Guardrail and retrieval are independent: run retrieval speculatively while the (Jev)
        # topic check is in flight, and discard it if the question is off-topic.
        (is_on_topic, deflection), results = await asyncio.gather(
            asyncio.to_thread(classify_query, request.question),
            asyncio.to_thread(
                retrieve,
                query=request.question,
                content_type=request.content_type,
                use_reranker=request.use_reranker,
                use_mmr=request.use_mmr,
                use_relevance_gate=False,
            ),
        )
        if not is_on_topic:
            return QueryResponse(
                answer=deflection or "Off-topic question.",
                sources=[],
                query=request.question,
                is_on_topic=False,
            )

        if not results:
            return QueryResponse(
                answer=build_no_context_response(),
                sources=[],
                query=request.question,
                provider=provider.name(),
            )

        # The relevance gate (Jev) and generation (LLM) are independent: run both at once and
        # drop the answer if the gate says the chunks can't answer the question.
        system_prompt = build_system_prompt(results)
        try:
            answerable, answer = await asyncio.gather(
                asyncio.to_thread(is_answerable, request.question, results) if request.use_reranker else _true(),
                provider.generate(system_prompt, build_user_message(request.question)),
            )
        except LLMBusyError:
            logfire.warn("All LLM models rate limited; returning 503")
            return JSONResponse(status_code=503, headers={"Retry-After": "60"}, content={"detail": BUSY_MESSAGE})
        if not answerable or is_no_answer(answer):
            return QueryResponse(
                answer=build_no_context_response(),
                sources=[],
                query=request.question,
                provider=provider.name(),
            )
        answer = validate_output(answer)

        sources = [
            SourceDocument(
                title=r.get("title", ""),
                url=r.get("url", ""),
                text=r.get("text", "")[:300],
                content_type=r.get("content_type", ""),
                score=r.get("rerank_score") or r.get("rrf_score") or r.get("score"),
            )
            for r in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.question,
            provider=provider.name(),
        )


async def _true() -> bool:
    return True
