"""답변 제안 스트리밍 (SSE). 글자가 만들어지는 대로 앱에 보낸다.

POST /suggest/stream  요약 + 답변 제안 3개.  요청 형식은 /suggest 와 같음
POST /suggest/expand  제안 카드를 눌렀을 때, 그 문장 뒤에 이어 붙일 문장

응답은 한 줄에 이벤트 하나:  data: {JSON}\n\n   …   마지막에  data: [DONE]
이벤트 모양은 llm/suggest_stream.py 맨 위 참고.
"""
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.suggest import SuggestRequest, load_context
from config import SUGGEST_TURNS
from llm.suggest_stream import stream_expand, stream_suggestions
from rooms.state import get_summary, recent_turns, set_summary

logger = logging.getLogger(__name__)
router = APIRouter()

# 중간 프록시가 모아서 한꺼번에 보내지 않도록
_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ExpandRequest(BaseModel):
    en: str                           # 고른 카드의 영어 문장
    room_id: str | None = None
    user_id: str | None = None


def _sse_response(events) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=_HEADERS)


async def _to_sse(events, on_event=None):
    """이벤트(dict) 스트림 → SSE 글자. on_event 로 이벤트를 엿볼 수 있다 (요약 저장용)."""
    async for event in events:
        if on_event:
            on_event(event)
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/suggest/stream")
async def suggest_stream(payload: SuggestRequest):
    t0 = time.time()
    logger.info(
        f"[SUGGEST-STREAM] room={payload.room_id} user={payload.user_id} speaker={payload.speaker} "
        f"need={payload.need_suggestions} seq={payload.seq} text={payload.latest_text!r}"
    )
    prev, turns = load_context(payload)

    def save_summary(event: dict):
        """요약이 끝나면 방 상태에 저장."""
        if event["type"] == "summary" and event["final"]:
            if payload.room_id and event["ok"]:
                set_summary(payload.room_id, payload.user_id,
                            {"remote": event["remote"], "me": event["me"]})
            logger.info(f"[SUGGEST-STREAM] 요약 저장 ({time.time() - t0:.2f}s)")

    events = stream_suggestions(prev, turns, payload.user_id, payload.latest_text,
                                payload.speaker, payload.need_suggestions)
    return _sse_response(_to_sse(events, save_summary))


@router.post("/suggest/expand")
async def suggest_expand(payload: ExpandRequest):
    logger.info(f"[EXPAND] room={payload.room_id} user={payload.user_id} en={payload.en!r}")
    if payload.room_id:
        prev = get_summary(payload.room_id, payload.user_id)
        turns = recent_turns(payload.room_id, SUGGEST_TURNS)
    else:
        prev, turns = {"remote": [], "me": []}, []
    return _sse_response(_to_sse(stream_expand(prev, turns, payload.user_id, payload.en.strip())))
