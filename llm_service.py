import json
import textwrap
from openai import AsyncOpenAI
from transformers import AutoTokenizer

LLM_CLIENT = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL_NAME = "Qwen/Qwen3-8B-FP8"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SYSTEM_PROMPT = textwrap.dedent(r"""\
    # Role
    EN-KR Translator.
    # Rules
    [CRITICAL] Apply [Glossary] definitions.
""")

def get_user_prompt(sentence, resolved_matches):
    base_prompt = f"원문: {sentence.strip()}\n번역:"
    if not resolved_matches: return base_prompt
    dict_lines = [f"- {m['term']}: {m['meaning']}" for m in resolved_matches]
    return f"[사전]\n" + "\n".join(dict_lines) + f"\n{base_prompt}"

async def fix_chinese_part(prefix_text, chinese_text):
    prompt = f"앞문장 '{prefix_text}'에 자연스럽게 이어지도록, 중국어 '{chinese_text}'를 한국어로 번역해. 번역된 결과만 출력해."
    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    response = await LLM_CLIENT.completions.create(
        model=MODEL_NAME, prompt=raw_prompt, temperature=0.1, max_tokens=50, extra_body={"enable_thinking": False}
    )
    return response.choices[0].text.strip()

async def generate_reply_suggestions(history):
    # 대화 이력이 전혀 없을 때 → 첫인사 제안
    if not history:
        return {
            "summary": "대화 시작 전",
            "suggestions": [
                {"en": "Hi, how are you?", "ko": "안녕, 잘 지내?"},
                {"en": "Nice to meet you!", "ko": "만나서 반가워!"}
            ]
        }

    # 영어 대화 이력을 텍스트로 변환 (User/Bot 포맷)
    history_text = "\n".join([
        f"User: {t['user']}\nBot: {t['assistant']}" for t in history
    ])

    prompt = f"""
You are analyzing a short English conversation. Do the following:
1. Write a one-sentence summary of the conversation in Korean.
2. Suggest 2 likely next replies that the User might say in English, along with their Korean translations.

Conversation history:
{history_text}

Output strictly as JSON:
{{
  "summary": "한국어 요약 문장",
  "suggestions": [
    {{"en": "English reply 1", "ko": "한국어 번역 1"}},
    {{"en": "English reply 2", "ko": "한국어 번역 2"}}
  ]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    try:
        response = await LLM_CLIENT.completions.create(
            model=MODEL_NAME,
            prompt=raw_prompt,
            temperature=0.3,
            max_tokens=200,
            extra_body={"enable_thinking": False}
        )
        return json.loads(response.choices[0].text.strip())
    except:
        # LLM 실패 시 안전장치
        return {
            "summary": "대화 내용을 요약할 수 없습니다.",
            "suggestions": [
                {"en": "I'm good, you?", "ko": "잘 지내, 너는?"},
                {"en": "Could you tell me more?", "ko": "좀 더 자세히 설명해줄래?"}
            ]
        }
