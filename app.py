import sys
import re
import time
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ============================================================
# 내부 모듈
# ============================================================
import nlp_db
from nlp_db import init_exact_matches, init_vector_db, retrieve_hybrid
from llm_service import (
    SYSTEM_PROMPT,
    get_user_prompt,
    fix_chinese_part,
    generate_reply_suggestions,
    LLM_CLIENT,
    MODEL_NAME,
    tokenizer,
)

if sys.stdin:
    sys.stdin.reconfigure(encoding="utf-8", errors="ignore")
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

ROOM_HISTORIES = defaultdict(lambda: deque(maxlen=3))
LAST_ACTIVE_DOMAIN = "league"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("===== 서버 시작 =====")
    init_vector_db()
    init_exact_matches()

    logger.info(f"EXACT_MATCH_DICT 크기: {len(nlp_db.EXACT_MATCH_DICT)}")
    yield
    logger.info("===== 서버 종료 =====")


app = FastAPI(lifespan=lifespan)


class TranslationRequest(BaseModel):
    room_id: str
    text: str


class SuggestRequest(BaseModel):
    summary: dict
    latest_text: str
    speaker: str
    need_suggestions: bool


# ============================================================
# /translate/stream
# ============================================================
@app.post("/translate/stream")
async def translate_stream(payload: TranslationRequest):
    global LAST_ACTIVE_DOMAIN

    t0 = time.time()
    room_id = payload.room_id
    user_input = payload.text.strip()

    logger.info(f"[REQ] room={room_id} text={user_input!r}")

    if not user_input:
        async def empty_resp():
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_resp(), media_type="text/event-stream")

    # ---- 1) quick.json 매칭 (★ nlp_db.normalize_text 사용) ----
    norm_input = nlp_db.normalize_text(user_input)
    logger.info(f"[NORM] {user_input!r} → {norm_input!r}")

    if norm_input in nlp_db.EXACT_MATCH_DICT:
        cached = nlp_db.EXACT_MATCH_DICT[norm_input]
        logger.info(f"[QUICK-HIT] {norm_input!r} → {cached!r} ({(time.time()-t0)*1000:.1f}ms)")

        async def fast_response():
            yield f"data: {cached}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(fast_response(), media_type="text/event-stream")

    logger.info(f"[QUICK-MISS] {norm_input!r} → LLM 추론 경로 진입")

    # ---- 2) 벡터 검색 ----
    resolved_matches, LAST_ACTIVE_DOMAIN = retrieve_hybrid(user_input, LAST_ACTIVE_DOMAIN)
    logger.info(f"[RETRIEVE] matches={len(resolved_matches)}, domain={LAST_ACTIVE_DOMAIN}")

    # ---- 3) LLM 프롬프트 구성 ----
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in ROOM_HISTORIES[room_id]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": get_user_prompt(user_input, resolved_matches)})

    raw_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    # ---- 4) SSE 스트리밍 ----
    async def event_generator():
        llm_t0 = time.time()
        logger.info("[LLM] 스트림 시작")

        response_stream = await LLM_CLIENT.completions.create(
            model=MODEL_NAME,
            prompt=raw_prompt,
            temperature=0.1,
            max_tokens=100,
            extra_body={"enable_thinking": False},
            stream=True,
        )

        full_chunks = []
        is_thinking = False
        buffer = ""
        chinese_buffer = ""

        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].text:
                buffer += chunk.choices[0].text

                if "<think>" in buffer and not is_thinking:
                    is_thinking = True

                if is_thinking:
                    if "</think>" in buffer:
                        is_thinking = False
                        buffer = buffer.split("</think>", 1)[-1]
                    else:
                        continue

                if not is_thinking and buffer:
                    if buffer.endswith("<") or buffer.endswith("<th"):
                        continue

                    if chinese_buffer or re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", buffer):
                        chinese_buffer += buffer
                        buffer = ""
                        match_non = re.search(r"[가-힣a-zA-Z]", chinese_buffer)
                        if match_non:
                            idx = match_non.start()
                            ch_part, kor_part = chinese_buffer[:idx], chinese_buffer[idx:]

                            if ch_part.strip():
                                prefix = "".join(full_chunks).strip()
                                korean_fix = await fix_chinese_part(prefix, ch_part)
                                yield f"data: {korean_fix} \n\n"
                                full_chunks.append(f"{korean_fix} ")

                            yield f"data: {kor_part}\n\n"
                            full_chunks.append(kor_part)
                            chinese_buffer = ""
                    else:
                        yield f"data: {buffer}\n\n"
                        full_chunks.append(buffer)
                        buffer = ""

        if chinese_buffer.strip():
            korean_fix = await fix_chinese_part("".join(full_chunks).strip(), chinese_buffer)
            yield f"data: {korean_fix} \n\n"
            full_chunks.append(f"{korean_fix} ")
        elif chinese_buffer:
            yield f"data: {chinese_buffer}\n\n"
            full_chunks.append(chinese_buffer)

        yield "data: [DONE]\n\n"

        clean_translation = "".join(full_chunks).strip()
        ROOM_HISTORIES[room_id].append({"user": user_input, "assistant": clean_translation})

        logger.info(
            f"[LLM] 완료: {clean_translation!r} "
            f"(llm={(time.time()-llm_t0):.2f}s, total={(time.time()-t0):.2f}s)"
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# /suggest
# ============================================================
@app.post("/suggest")
async def suggest_replies(payload: SuggestRequest):
    logger.info(f"[SUGGEST] speaker={payload.speaker}, need={payload.need_suggestions}, text={payload.latest_text!r}")
    result = await generate_reply_suggestions(
        prev_summary=payload.summary,
        latest_text=payload.latest_text,
        speaker=payload.speaker,
        need_suggestions=payload.need_suggestions,
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)
