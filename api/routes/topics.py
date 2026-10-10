from __future__ import annotations

from fastapi import APIRouter

from api.schemas import TopicsResponse
from config.constants import SADHGURU_TOPICS

router = APIRouter(tags=["topics"])


@router.get("/topics")
async def list_topics() -> TopicsResponse:
    return TopicsResponse(topics=sorted(set(SADHGURU_TOPICS)), total=len(set(SADHGURU_TOPICS)))
