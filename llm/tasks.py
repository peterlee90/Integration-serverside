"""LLM 보조 작업: 중국어 섞임 보정, 대화 요약 + 답변 제안."""
import json
import logging
import re

from config import (
    FIX_MAX_TOKENS, FIX_TEMPERATURE, SUGGEST_JSON_SCHEMA,
    SUGGEST_MAX_TOKENS, SUGGEST_TEMPERATURE, SUMMARY_MAX_KEYWORDS, SUMMARY_MAX_TOKENS,
)
from llm.client import build_prompt, complete_text
from llm.prompts import chinese_fix_prompt
from llm.suggest_prompts import (
    SUMMARY_ONLY_SCHEMA, WITH_SUGGESTIONS_SCHEMA, format_turns, suggestion_prompt,
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
        words = [str(w).strip() for w in raw.get(key, []) if str(w).strip()]
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


async def generate_reply_suggestions(
    prev_summary: dict, turns: list, user_id: str | None,
    latest_text: str, speaker: str, need_suggestions: bool,
) -> dict:
    """요약을 갱신하고, need_suggestions 면 답장 3개도 만든다. 실패하면 이전 요약 유지."""
    content = suggestion_prompt(prev_summary, format_turns(turns, user_id),
                                latest_text, speaker, need_suggestions)
    prompt = _user_prompt(content)
    max_tokens = SUGGEST_MAX_TOKENS if need_suggestions else SUMMARY_MAX_TOKENS
    schema = None
    if SUGGEST_JSON_SCHEMA:
        schema = WITH_SUGGESTIONS_SCHEMA if need_suggestions else SUMMARY_ONLY_SCHEMA

    try:
        text = await complete_text(prompt, temperature=SUGGEST_TEMPERATURE,
                                   max_tokens=max_tokens, json_schema=schema)
        result = _parse_json(text)
        return {
            "summary": _clean_summary(result.get("summary"), prev_summary),
            "suggestions": _clean_suggestions(result.get("suggestions")) if need_suggestions else [],
            "ok": True,
        }
    except Exception as e:
        logger.warning(f"[SUGGEST] 실패 → 이전 요약 유지: {type(e).__name__}: {e}")
        return {
            "summary": prev_summary,
            "suggestions": _fallback_suggestions() if need_suggestions else [],
            "ok": False,
        }
