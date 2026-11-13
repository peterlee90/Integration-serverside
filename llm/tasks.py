"""LLM 보조 작업: 중국어 섞임 보정, 대화 요약 + 답변 제안."""
import asyncio
import json
import logging
import re
import time

from config import (
    FIX_MAX_TOKENS, FIX_TEMPERATURE, SUGGEST_JSON_SCHEMA,
    SUGGEST_MAX_TOKENS, SUGGEST_TEMPERATURE, SUMMARY_MAX_KEYWORDS, SUMMARY_MAX_TOKENS,
)
from llm.client import build_prompt, complete_text
from llm.prompts import chinese_fix_prompt
from llm.suggest_prompts import (
    SUGGESTIONS_SCHEMA, SUMMARY_SCHEMA, format_turns, suggestion_prompt, summary_prompt,
)

logger = logging.getLogger(__name__)

# 모델이 제안을 못 만들었을 때 쓰는 기본 답장
FALLBACK_SUGGESTIONS = [
    {"en": "I agree.", "ko": "동의해."},
    {"en": "What do you mean?", "ko": "무슨 뜻이야?"},
    {"en": "Let's change the subject.", "ko": "주제를 바꾸자."},
]


def _fallback_suggestions() -> list:
    return [dict(s) for s in FALLBACK_SUGGESTIONS]


def _user_prompt(content: str) -> str:
    return build_prompt([{"role": "user", "content": content}])


async def fix_chinese_part(prefix_text: str, chinese_text: str) -> str:
    prompt = _user_prompt(chinese_fix_prompt(prefix_text, chinese_text))
    return await complete_text(prompt, temperature=FIX_TEMPERATURE, max_tokens=FIX_MAX_TOKENS)


def _parse_json(text: str) -> dict:
    """모델 출력에서 JSON 부분만 꺼낸다 (```json 감싸기, 앞뒤 잡담 허용)."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"JSON 없음: {text[:80]!r}")
    return json.loads(match.group(0))


def _clean_summary(raw, prev: dict) -> dict:
    """요약 형식 보정 + 사람별 키워드 개수 제한 (넘치면 오래된 것부터 버림)."""
    if not isinstance(raw, dict):
        return prev
    out = {}
    for key in ("remote", "me"):
        value = raw.get(key, [])
        words = [str(w).strip() for w in value if str(w).strip()] if isinstance(value, list) else []
        out[key] = words[-SUMMARY_MAX_KEYWORDS:]
    return out


def _clean_suggestions(raw) -> list:
    """제안 3개 형식 보정. 모자라면 기본 답장으로 채움."""
    items = []
    for s in raw if isinstance(raw, list) else []:
        if isinstance(s, dict) and str(s.get("en", "")).strip():
            items.append({"en": str(s["en"]).strip(), "ko": str(s.get("ko", "")).strip()})
    for fb in _fallback_suggestions():
        if len(items) >= 3:
            break
        items.append(fb)
    return items[:3]


async def _ask_json(name: str, content: str, max_tokens: int, schema: dict) -> dict:
    """LLM 한 번 호출 → JSON. 걸린 시간과 출력 길이를 로그로 남긴다 (느린 쪽 찾기용)."""
    t0 = time.time()
    text = await complete_text(
        _user_prompt(content), temperature=SUGGEST_TEMPERATURE, max_tokens=max_tokens,
        json_schema=schema if SUGGEST_JSON_SCHEMA else None,
    )
    logger.info(f"[SUGGEST] {name} LLM {(time.time() - t0):.2f}s, 출력 {len(text)}자")
    return _parse_json(text)


async def _make_summary(prev: dict, turns_text: str, latest_text: str, speaker: str) -> tuple:
    """(요약, 성공 여부). 실패하면 이전 요약 유지."""
    try:
        result = await _ask_json("요약", summary_prompt(prev, turns_text, latest_text, speaker),
                                 SUMMARY_MAX_TOKENS, SUMMARY_SCHEMA)
        return _clean_summary(result.get("summary"), prev), True
    except Exception as e:
        logger.warning(f"[SUGGEST] 요약 실패 → 이전 요약 유지: {type(e).__name__}: {e}")
        return prev, False


async def _make_suggestions(prev: dict, turns_text: str, latest_text: str, speaker: str) -> list:
    """답장 3개. 실패하면 기본 답장."""
    try:
        result = await _ask_json("제안", suggestion_prompt(prev, turns_text, latest_text, speaker),
                                 SUGGEST_MAX_TOKENS, SUGGESTIONS_SCHEMA)
        return _clean_suggestions(result.get("suggestions"))
    except Exception as e:
        logger.warning(f"[SUGGEST] 제안 실패 → 기본 답장: {type(e).__name__}: {e}")
        return _fallback_suggestions()


async def generate_reply_suggestions(
    prev_summary: dict, turns: list, user_id: str | None,
    latest_text: str, speaker: str, need_suggestions: bool,
) -> dict:
    """요약을 갱신하고, need_suggestions 면 답장 3개도 만든다.

    요약과 제안은 LLM 호출을 따로 해서 동시에 돌린다 (한 번에 길게 생성하는 것보다 빨리 끝남).
    제안은 이전 요약 + 최근 대화를 보고 만든다. "ok" 는 요약 성공 여부.
    """
    turns_text = format_turns(turns, user_id)
    args = (prev_summary, turns_text, latest_text, speaker)
    if need_suggestions:
        (summary, ok), suggestions = await asyncio.gather(
            _make_summary(*args), _make_suggestions(*args))
    else:
        (summary, ok), suggestions = await _make_summary(*args), []
    return {"summary": summary, "suggestions": suggestions, "ok": ok}
