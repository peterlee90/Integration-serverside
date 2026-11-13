"""요약·답변 제안·이어 말하기를 스트리밍으로 만든다 (글자가 오는 대로 이벤트를 내보냄).

LLM 호출 여러 개(요약 1 + 제안 3)를 동시에 돌리고, 각자 나오는 대로 한 줄로 합쳐서 돌려준다.
이벤트 (dict):
  {"type": "suggestion", "i": 0~2, "en": 지금까지, "ko": 지금까지, "done": 끝났는지}
  {"type": "summary", "remote": [...], "me": [...], "final": 끝났는지, "ok": 성공 여부}
  {"type": "more", "en": 지금까지, "ko": 지금까지, "done": 끝났는지}      ← 이어 말하기
"""
import asyncio
import logging
import time

from config import (
    EXPAND_MAX_TOKENS, SUGGEST_JSON_SCHEMA, SUGGEST_MAX_TOKENS, SUGGEST_TEMPERATURE,
    SUMMARY_MAX_TOKENS,
)
from llm.client import build_prompt, complete
from llm.partial_json import partial_list, partial_string
from llm.suggest_prompts import (
    DIRECTIONS, ONE_SUGGESTION_SCHEMA, SUMMARY_SCHEMA,
    expand_prompt, format_turns, suggestion_prompt, summary_prompt,
)
from llm.tasks import FALLBACK_SUGGESTIONS, _clean_summary, _parse_json

logger = logging.getLogger(__name__)


async def _stream_text(content: str, max_tokens: int, schema: dict):
    """LLM 스트림. 토막이 올 때마다 '지금까지 쌓인 전체 글자'를 돌려준다."""
    stream = await complete(
        build_prompt([{"role": "user", "content": content}]),
        temperature=SUGGEST_TEMPERATURE, max_tokens=max_tokens, stream=True,
        json_schema=schema if SUGGEST_JSON_SCHEMA else None,
    )
    text = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            text += chunk.choices[0].text
            yield text


async def _pair_worker(queue, name: str, event: dict, content: str, max_tokens: int,
                       fallback: dict | None):
    """{"en", "ko"} 한 쌍을 스트리밍한다 (제안 카드 1개, 이어 말하기 공통)."""
    t0, first, en, ko = time.time(), None, "", ""
    try:
        async for text in _stream_text(content, max_tokens, ONE_SUGGESTION_SCHEMA):
            new = partial_string(text, "en"), partial_string(text, "ko")
            if new == (en, ko):
                continue
            en, ko = new
            if first is None and en:
                first = time.time() - t0
            await queue.put({**event, "en": en, "ko": ko, "done": False})
    except Exception as e:
        logger.warning(f"[SUGGEST] {name} 실패: {type(e).__name__}: {e}")
    if not en.strip() and fallback:
        en, ko = fallback["en"], fallback["ko"]
    await queue.put({**event, "en": en.strip(), "ko": ko.strip(), "done": True})
    logger.info(f"[SUGGEST] {name} 첫 글자 {first or 0:.2f}s, 끝 {time.time() - t0:.2f}s: {en.strip()!r}")


async def _summary_worker(queue, prev: dict, turns_text: str, latest_text: str, speaker: str):
    """요약을 스트리밍한다. 키워드가 하나 완성될 때마다 내보내고, 끝나면 final."""
    t0, shown, text, ok = time.time(), None, "", False
    summary = prev
    try:
        async for text in _stream_text(summary_prompt(prev, turns_text, latest_text, speaker),
                                       SUMMARY_MAX_TOKENS, SUMMARY_SCHEMA):
            now = (partial_list(text, "remote"), partial_list(text, "me"))
            if now != shown and any(now):
                shown = now
                await queue.put({"type": "summary", "remote": now[0], "me": now[1],
                                 "final": False, "ok": True})
        summary, ok = _clean_summary(_parse_json(text).get("summary"), prev), True
    except Exception as e:
        logger.warning(f"[SUGGEST] 요약 실패 → 이전 요약 유지: {type(e).__name__}: {e}")
    await queue.put({"type": "summary", "remote": summary.get("remote", []),
                     "me": summary.get("me", []), "final": True, "ok": ok})
    logger.info(f"[SUGGEST] 요약 끝 {time.time() - t0:.2f}s ok={ok}: {summary}")


async def _merge(workers: list):
    """worker(queue) 들을 동시에 돌리고 이벤트를 나오는 순서대로 돌려준다.
    받는 쪽이 중간에 끊으면(앱이 요청 취소) 남은 LLM 호출도 같이 멈춘다."""
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [asyncio.create_task(w(queue)) for w in workers]

    async def _close_when_done():
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.put(None)

    closer = asyncio.create_task(_close_when_done())
    try:
        while (event := await queue.get()) is not None:
            yield event
    finally:
        for t in tasks + [closer]:
            t.cancel()


def stream_suggestions(prev: dict, turns: list, user_id: str | None, latest_text: str,
                       speaker: str, need_suggestions: bool):
    """요약(+ need_suggestions 면 제안 3개) 이벤트 스트림."""
    turns_text = format_turns(turns, user_id)
    workers = [lambda q: _summary_worker(q, prev, turns_text, latest_text, speaker)]
    if need_suggestions:
        for i, direction in enumerate(DIRECTIONS):
            content = suggestion_prompt(prev, turns_text, latest_text, speaker, direction)
            workers.append(lambda q, i=i, content=content: _pair_worker(
                q, f"제안{i + 1}", {"type": "suggestion", "i": i}, content,
                SUGGEST_MAX_TOKENS // len(DIRECTIONS), FALLBACK_SUGGESTIONS[i]))
    return _merge(workers)


def stream_expand(prev: dict, turns: list, user_id: str | None, first_en: str):
    """고른 카드 문장 뒤에 이어 붙일 문장 이벤트 스트림. 실패하면 빈 문장으로 끝남."""
    content = expand_prompt(prev, format_turns(turns, user_id), first_en)
    return _merge([lambda q: _pair_worker(q, "이어 말하기", {"type": "more"}, content,
                                          EXPAND_MAX_TOKENS, None)])
