"""LLM 보조 작업: 중국어 섞임 보정, 대화 요약 + 답장 제안."""
import json
import logging

from config import (
    FIX_MAX_TOKENS, FIX_TEMPERATURE,
    SUGGEST_MAX_TOKENS, SUGGEST_TEMPERATURE, SUMMARY_MAX_TOKENS,
)
from llm.client import build_prompt, complete_text
from llm.prompts import chinese_fix_prompt, suggestion_prompt

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


async def generate_reply_suggestions(
    prev_summary: dict, latest_text: str, speaker: str, need_suggestions: bool
) -> dict:
    """요약을 갱신하고, need_suggestions 면 답장 3개도 만든다. 실패하면 기존 요약 유지."""
    prompt = _user_prompt(suggestion_prompt(prev_summary, latest_text, speaker, need_suggestions))
    max_tokens = SUGGEST_MAX_TOKENS if need_suggestions else SUMMARY_MAX_TOKENS

    try:
        text = await complete_text(prompt, temperature=SUGGEST_TEMPERATURE, max_tokens=max_tokens)
        result = json.loads(text)

        # 출력 형식 보정
        if not need_suggestions:
            result["suggestions"] = []
        elif "suggestions" not in result:
            result["suggestions"] = _fallback_suggestions()
        return result

    except Exception as e:
        logger.warning(f"Suggestion Error: {e}")
        return {
            "summary": prev_summary,
            "suggestions": _fallback_suggestions() if need_suggestions else [],
        }
