"""'내가 하고 싶은 말' 스트리밍: 같은 프롬프트를 Kanana 와 Claude 에 동시에 보내고 오는 대로 내보낸다.

이벤트 (dict):
  {"type": "say", "provider": "kanana" | "claude", "en": 지금까지, "ko": 지금까지,
   "done": 끝났는지, "error": 실패 이유(성공이면 "")}
"""
import logging
import time

from config import SAY_MAX_TOKENS, SAY_PROVIDERS, SAY_TEMPERATURE
from llm.claude_client import claude_ready, stream_claude
from llm.partial_json import partial_string
from llm.say_prompts import say_prompt
from llm.suggest_prompts import ONE_SUGGESTION_SCHEMA, format_turns
from llm.suggest_stream import _merge, _stream_text

logger = logging.getLogger(__name__)


def _text_stream(provider: str, content: str):
    """provider 별 LLM 스트림 (둘 다 '지금까지 쌓인 전체 글자'를 돌려줌)."""
    if provider == "claude":
        if not claude_ready():
            raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 없음")
        return stream_claude(content, SAY_MAX_TOKENS, SAY_TEMPERATURE)
    return _stream_text(content, SAY_MAX_TOKENS, ONE_SUGGESTION_SCHEMA)


async def _say_worker(queue, provider: str, content: str):
    t0, first, en, ko, error = time.time(), None, "", "", ""
    event = {"type": "say", "provider": provider}
    try:
        async for text in _text_stream(provider, content):
            new = partial_string(text, "en"), partial_string(text, "ko")
            if new == (en, ko):
                continue
            en, ko = new
            if first is None and en:
                first = time.time() - t0
            await queue.put({**event, "en": en, "ko": ko, "done": False, "error": ""})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning(f"[SAY] {provider} 실패: {error}")
    await queue.put({**event, "en": en.strip(), "ko": ko.strip(), "done": True, "error": error})
    logger.info(f"[SAY] {provider} 첫 글자 {first or 0:.2f}s, 끝 {time.time() - t0:.2f}s: {en.strip()!r}")


def stream_say(prev: dict, turns: list, user_id: str | None, ko_text: str,
               providers: list | None = None, base_en: str = "", previous: list | None = None):
    """한국어 문장 → provider 마다 영어 카드 1개씩 (이벤트 스트림).

    providers: 이번에 쓸 모델만 (없으면 config.SAY_PROVIDERS 전부). "다시 만들기"는 한 모델만 보냄
    base_en / previous: llm/say_prompts.py 참고
    """
    wanted = [p for p in SAY_PROVIDERS if not providers or p in providers]
    content = say_prompt(prev, format_turns(turns, user_id), ko_text, base_en, previous)
    return _merge([lambda q, p=p: _say_worker(q, p, content) for p in wanted])
