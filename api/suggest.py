"""POST /suggest — 대화 요약 갱신 + (필요하면) 답장 제안 3개."""
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.tasks import generate_reply_suggestions

logger = logging.getLogger(__name__)
router = APIRouter()


class SuggestRequest(BaseModel):
    summary: dict
    latest_text: str
    speaker: str
    need_suggestions: bool


@router.post("/suggest")
async def suggest_replies(payload: SuggestRequest):
    logger.info(
        f"[SUGGEST] speaker={payload.speaker}, need={payload.need_suggestions}, "
        f"text={payload.latest_text!r}"
    )
    result = await generate_reply_suggestions(
        prev_summary=payload.summary,
        latest_text=payload.latest_text,
        speaker=payload.speaker,
        need_suggestions=payload.need_suggestions,
    )
    return JSONResponse(content=result)
