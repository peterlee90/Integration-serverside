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

# 🔄 롤링 요약 및 조건부 제안 생성으로 전면 수정
async def generate_reply_suggestions(prev_summary: dict, latest_text: str, speaker: str, need_suggestions: bool):
    speaker_label = "Me" if speaker == "me" else "Remote"
    
    # 1. 공통 프롬프트 (요약 갱신)
    prompt = f"""[Current Status]
Remote: {json.dumps(prev_summary.get("remote", []), ensure_ascii=False)}
Me: {json.dumps(prev_summary.get("me", []), ensure_ascii=False)}

[Latest Message ({speaker_label})]
"{latest_text}"

Task:
1. Update the [Current Status] based on the [Latest Message] using VERY short Korean keywords. Do not hallucinate.
"""

    # 2. 제안 요청 여부에 따른 출력 스키마 분기
    if need_suggestions:
        prompt += """2. Suggest 3 natural English replies for 'Me' to respond to 'Remote', along with Korean translations.

Output STRICTLY as JSON:
{
  "summary": {"remote": ["keyword1"], "me": ["keyword2"]},
  "suggestions": [
    {"en": "reply 1", "ko": "번역 1"},
    {"en": "reply 2", "ko": "번역 2"},
    {"en": "reply 3", "ko": "번역 3"}
  ]
}"""
    else:
        prompt += """
Output STRICTLY as JSON:
{
  "summary": {"remote": ["keyword1"], "me": ["keyword2"]}
}"""

    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    try:
        response = await LLM_CLIENT.completions.create(
            model=MODEL_NAME,
            prompt=raw_prompt,
            temperature=0.2, # 환각 방지를 위해 낮춤
            max_tokens=250 if need_suggestions else 100, 
            extra_body={"enable_thinking": False}
        )
        
        result_json = json.loads(response.choices[0].text.strip())
        
        # 방어 코드 (포맷팅 보장)
        if not need_suggestions:
            result_json["suggestions"] = []
        elif need_suggestions and "suggestions" not in result_json:
            result_json["suggestions"] = [
                {"en": "I agree.", "ko": "동의해."},
                {"en": "What do you mean?", "ko": "무슨 뜻이야?"},
                {"en": "Let's change the subject.", "ko": "주제를 바꾸자."}
            ]
            
        return result_json
        
    except Exception as e:
        print(f"Suggestion Error: {e}")
        # 에러 시 기존 요약 유지 및 안전한 폴백 반환
        fallback = {"summary": prev_summary, "suggestions": []}
        if need_suggestions:
            fallback["suggestions"] = [
                {"en": "I agree.", "ko": "동의해."},
                {"en": "What do you mean?", "ko": "무슨 뜻이야?"},
                {"en": "Let's change the subject.", "ko": "주제를 바꾸자."}
            ]
        return fallback
