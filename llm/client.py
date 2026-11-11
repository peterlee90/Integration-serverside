"""LLM(vLLM OpenAI 호환 서버) 연결과 호출 헬퍼."""
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME

LLM_CLIENT = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def build_prompt(messages: list) -> str:
    """채팅 메시지 목록 → 모델 입력 문자열 (thinking 끔)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


async def complete(prompt: str, *, temperature: float, max_tokens: int, stream: bool = False):
    """completions 호출. stream=True 면 비동기 스트림을 돌려준다."""
    stream_arg = {"stream": True} if stream else {}
    return await LLM_CLIENT.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
        **stream_arg,
    )


async def complete_text(prompt: str, *, temperature: float, max_tokens: int) -> str:
    """스트리밍 없이 호출하고 결과 텍스트만 돌려준다."""
    response = await complete(prompt, temperature=temperature, max_tokens=max_tokens)
    return response.choices[0].text.strip()
