"""POST /suggest — 대화 요약 갱신 + (필요하면) 답변 제안 3개.

room_id + user_id 를 보내면: 요약은 서버의 방 상태에서 읽고/저장, 최근 대화 턴도 서버에서 가져옴.
room_id 를 안 보내면 예전처럼 앱이 보낸 summary 만 사용 (최근 턴 없음).
"""
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SUGGEST_TURNS
from llm.tasks import generate_reply_suggestions
from rooms.state import add_turn, get_summary, recent_turns, set_summary

logger = logging.getLogger(__name__)
router = APIRouter()


class SuggestRequest(BaseModel):
    latest_text: str
    speaker: str                      # "me" | "remote" (요청하는 사람 기준)
    need_suggestions: bool
    room_id: str | None = None
    user_id: str | None = None        # 요청하는 사람(나)의 사용자 id
    summary: dict | None = None       # 예전 방식: 앱이 들고 있던 요약
    seq: int | None = None            # 앱 요청 번호. 그대로 돌려줌 → 늦게 온 옛 응답 무시용


def _recent_without_latest(room_id: str, latest_text: str) -> list:
    """최근 턴. 마지막 턴이 latest_text 와 같으면 빼서 중복을 막는다."""
    turns = recent_turns(room_id, SUGGEST_TURNS + 1)
    if turns and turns[-1].text.strip() == latest_text.strip():
        turns = turns[:-1]
    return turns[-SUGGEST_TURNS:] if SUGGEST_TURNS > 0 else []


@router.post("/suggest")
async def suggest_replies(payload: SuggestRequest):
    t0 = time.time()
    logger.info(
        f"[SUGGEST] room={payload.room_id} user={payload.user_id} speaker={payload.speaker} "
        f"need={payload.need_suggestions} seq={payload.seq} text={payload.latest_text!r}"
    )

    if payload.room_id:
        # 내가 한 말은 여기서도 턴으로 남긴다 (앱 QuickDict 에서 끝난 문장은 번역 서버를 안 거치므로)
        if payload.speaker == "me" and payload.user_id and payload.latest_text.strip():
            add_turn(payload.room_id, payload.user_id, payload.latest_text.strip(), "")
        prev = get_summary(payload.room_id, payload.user_id)
        turns = _recent_without_latest(payload.room_id, payload.latest_text)
    else:
        prev = payload.summary or {"remote": [], "me": []}
        turns = []

    result = await generate_reply_suggestions(
        prev_summary=prev,
        turns=turns,
        user_id=payload.user_id,
        latest_text=payload.latest_text,
        speaker=payload.speaker,
        need_suggestions=payload.need_suggestions,
    )

    if payload.room_id and result["ok"]:
        set_summary(payload.room_id, payload.user_id, result["summary"])

    logger.info(
        f"[SUGGEST] 완료 ok={result['ok']} turns={len(turns)} ({(time.time()-t0):.2f}s) "
        f"summary={result['summary']} suggestions={[s['en'] for s in result['suggestions']]}"
    )
    return JSONResponse(content={
        "summary": result["summary"],
        "suggestions": result["suggestions"],
        "seq": payload.seq,
    })
