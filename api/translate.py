"""POST /translate/stream — 영→한 번역 (SSE 스트리밍).

순서: quick.json 정확 매칭 → (실패 시) 사전 검색 → LLM 스트리밍 (LoRA 어댑터)
번역이 끝나면 방별 상태(rooms/state.py)에 대화 턴으로 저장한다.
"""
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    HISTORY_TURNS, TRANSLATE_MAX_TOKENS, TRANSLATE_MODEL_NAME, TRANSLATE_STOP, TRANSLATE_TEMPERATURE,
)
from lexicon import quick_match
from lexicon.normalize import normalize_text
from lexicon.retrieve import retrieve_hybrid
from llm.client import build_prompt, complete
from llm.prompts import SYSTEM_PROMPT, translation_prompt
from llm.stream_cleaner import TranslationStreamCleaner
from llm.tasks import fix_chinese_part
from rooms.state import add_turn, get_domain, recent_turns, room_count, set_domain

logger = logging.getLogger(__name__)
router = APIRouter()


class TranslationRequest(BaseModel):
    room_id: str
    text: str
    speaker: str | None = None   # 말한 사람(사용자 id). 앱이 아직 안 보내면 "unknown"


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


_DONE = _sse("[DONE]")


def _sse_response(events) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream")


async def _fixed_events(*events: str):
    for e in events:
        yield e


@router.post("/translate/stream")
async def translate_stream(payload: TranslationRequest):
    t0 = time.time()
    room_id, speaker = payload.room_id, payload.speaker
    user_input = payload.text.strip()
    logger.info(f"[REQ] room={room_id} speaker={speaker} text={user_input!r}")

    if not user_input:
        return _sse_response(_fixed_events(_DONE))

    # 1) quick.json 정확 매칭
    norm_input = normalize_text(user_input)
    logger.info(f"[NORM] {user_input!r} → {norm_input!r}")

    if norm_input in quick_match.EXACT_MATCH_DICT:
        cached = quick_match.EXACT_MATCH_DICT[norm_input]
        add_turn(room_id, speaker, user_input, cached)
        logger.info(f"[QUICK-HIT] {norm_input!r} → {cached!r} ({(time.time()-t0)*1000:.1f}ms)")
        return _sse_response(_fixed_events(_sse(cached), _DONE))

    logger.info(f"[QUICK-MISS] {norm_input!r} → LLM 추론 경로 진입")

    # 2) 사전 검색 (도메인은 방마다 따로)
    matches, domain = retrieve_hybrid(user_input, get_domain(room_id))
    set_domain(room_id, domain)
    logger.info(f"[RETRIEVE] matches={len(matches)}, domain={domain}")

    # 3) 프롬프트 → 4) 스트리밍
    prompt = build_prompt(_build_messages(room_id, user_input, matches))
    return _sse_response(_stream_translation(room_id, speaker, user_input, prompt, t0))


def _build_messages(room_id: str, user_input: str, matches: list) -> list:
    """시스템 + 방의 최근 턴들 + 이번 문장. user 는 전부 학습 때 모양('번역할 문장: ...')으로."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in recent_turns(room_id, HISTORY_TURNS):
        if not turn.translation:
            continue
        messages.append({"role": "user", "content": translation_prompt(turn.text, [])})
        messages.append({"role": "assistant", "content": turn.translation})
    messages.append({"role": "user", "content": translation_prompt(user_input, matches)})
    return messages


async def _stream_translation(room_id: str, speaker: str | None, user_input: str, prompt: str, t0: float):
    llm_t0 = time.time()
    logger.info(f"[LLM] 스트림 시작 (model={TRANSLATE_MODEL_NAME})")

    stream = await complete(
        prompt,
        temperature=TRANSLATE_TEMPERATURE,
        max_tokens=TRANSLATE_MAX_TOKENS,
        stream=True,
        stop=TRANSLATE_STOP,
        model=TRANSLATE_MODEL_NAME,
    )
    cleaner = TranslationStreamCleaner(fix_chinese_part)

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            async for piece in cleaner.feed(chunk.choices[0].text):
                yield _sse(piece)
    async for piece in cleaner.flush():
        yield _sse(piece)
    yield _DONE

    translation = cleaner.text
    add_turn(room_id, speaker, user_input, translation)
    logger.info(
        f"[LLM] 완료: {translation!r} "
        f"(llm={(time.time()-llm_t0):.2f}s, total={(time.time()-t0):.2f}s, rooms={room_count()})"
    )
