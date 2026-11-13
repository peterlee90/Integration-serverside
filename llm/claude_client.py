"""Claude API 스트리밍 호출 (Messages API, SSE).

키는 서버 환경변수 ANTHROPIC_API_KEY 에서만 읽는다. 코드·앱·채팅에 키를 적지 말 것.
"""
import json
import os

import httpx

from config import CLAUDE_API_URL, CLAUDE_API_VERSION, CLAUDE_MODEL, CLAUDE_TIMEOUT_SEC


def claude_ready() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


async def stream_claude(content: str, max_tokens: int, temperature: float):
    """토막이 올 때마다 '지금까지 쌓인 전체 글자'를 돌려준다 (llm/suggest_stream._stream_text 와 같은 모양)."""
    headers = {
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "messages": [{"role": "user", "content": content}],
    }
    text = ""
    async with httpx.AsyncClient(timeout=CLAUDE_TIMEOUT_SEC) as client:
        async with client.stream("POST", CLAUDE_API_URL, headers=headers, json=body) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"Claude API {response.status_code}: {detail}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if event.get("type") == "error":
                    raise RuntimeError(f"Claude API 오류: {event.get('error')}")
                if event.get("type") != "content_block_delta":
                    continue
                piece = event.get("delta", {}).get("text", "")
                if piece:
                    text += piece
                    yield text
