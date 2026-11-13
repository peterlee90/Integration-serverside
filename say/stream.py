"""POST /say/stream — '내가 하고 싶은 말': 한국어로 적은 말을 지금 대화에 맞는 영어로 (SSE).

요청: {"text": "한국어 문장", "room_id": "...", "user_id": "...",
       "providers": ["kanana"],      ← 생략하면 config.SAY_PROVIDERS 전부. "다시 만들기"는 한 모델만
       "previous": ["앞서 받은 영어"], ← 다시 만들 때: 이것과 다른 표현으로
       "base_en": "이미 고른 영어 문장"} ← 있으면 그 뒤에 덧붙일 말만 만든다
응답: data: {"type": "say", "provider": "kanana"|"claude", "en", "ko", "done", "error"}  …  data: [DONE]
모델마다 카드가 하나씩 나온다.
"""
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from api.suggest_stream import _sse_response, _to_sse
from config import SUGGEST_TURNS
from llm.say_stream import stream_say
from rooms.state import get_summary, recent_turns

logger = logging.getLogger(__name__)
router = APIRouter()


class SayRequest(BaseModel):
    text: str                         # 사용자가 한국어로 적은(말한) 문장
    room_id: str | None = None
    user_id: str | None = None
    providers: list[str] | None = None
    previous: list[str] = []
    base_en: str = ""


@router.post("/say/stream")
async def say_stream(payload: SayRequest):
    logger.info(
        f"[SAY] room={payload.room_id} user={payload.user_id} providers={payload.providers} "
        f"again={len(payload.previous)} base={payload.base_en!r} text={payload.text!r}"
    )
    if payload.room_id:
        prev = get_summary(payload.room_id, payload.user_id)
        turns = recent_turns(payload.room_id, SUGGEST_TURNS)
    else:
        prev, turns = {"remote": [], "me": []}, []
    events = stream_say(prev, turns, payload.user_id, payload.text.strip(),
                        payload.providers, payload.base_en.strip(), payload.previous)
    return _sse_response(_to_sse(events))
