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
    if not history: return ["💡 알겠습니다.", "💡 더 자세히 설명해주세요."]
    
    # 히스토리 텍스트화
    history_text = "\n".join([f"User: {t['user']}\nBot: {t['assistant']}" for t in history])
    
    prompt = f"""
    아래 대화 내역의 핵심 키워드를 요약하고, 사용자가 봇에게 이어서 할 법한 짧은 답변 제안 2가지를 작성해.
    [대화 내역]
    {history_text}
    
    출력 형식은 반드시 아래와 같은 JSON 배열이어야 해:
    ["💡 제안내용1", "💡 제안내용2"]
    """
    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    try:
        response = await LLM_CLIENT.completions.create(
            model=MODEL_NAME, prompt=raw_prompt, temperature=0.3, max_tokens=100, extra_body={"enable_thinking": False}
        )
        return json.loads(response.choices[0].text.strip())
    except:
        return ["💡 알겠습니다.", "💡 잘 이해가 안 가요."]
