"""LLM 에 보내는 프롬프트 모음.

문구를 바꾸면 모델 출력이 달라지니, 바꾼 뒤엔 꼭 실제 문장으로 확인.
"""
import json

# 번역 규칙. 출력은 한 줄 "번역1 / 번역2" (줄바꿈에서 생성이 끊김 → config.TRANSLATE_STOP)
SYSTEM_PROMPT = (
    "너는 영어를 한국어로 옮기는 실시간 자막 번역기다.\n"
    "규칙:\n"
    "1. 자연스러운 한국어 번역 2개를 한 줄에 ' / '로 구분해서 출력한다.\n"
    "2. 번역 말고는 아무것도 쓰지 않는다. (원문, 설명, 참고, 따옴표, 번호, 줄바꿈 금지)\n"
    "3. [사전]이 주어지면 그 단어는 반드시 사전의 뜻으로 번역한다.\n"
    "\n"
    "예시\n"
    "입력: You gotta be kidding me.\n"
    "출력: 농담이시죠? / 설마 진심이에요?"
)


def translation_prompt(sentence: str, matches: list) -> str:
    """번역 요청 본문: (사전 블록) + 원문. 모델이 따라 쓰지 않게 '원문:/번역:' 표시는 안 붙인다."""
    sentence = sentence.strip()
    if not matches:
        return sentence
    lines = [f"- {m['term']}: {m['meaning']}" for m in matches]
    return "[사전]\n" + "\n".join(lines) + f"\n\n{sentence}"


def chinese_fix_prompt(prefix_text: str, chinese_text: str) -> str:
    """번역 중 섞여 나온 중국어를 앞 문장에 이어지게 한국어로 바꾸는 요청."""
    return (
        f"앞문장 '{prefix_text}'에 자연스럽게 이어지도록, "
        f"중국어 '{chinese_text}'를 한국어로 번역해. 번역된 결과만 출력해."
    )


_SUMMARY_ONLY_SCHEMA = """
Output STRICTLY as JSON:
{
  "summary": {"remote": ["keyword1"], "me": ["keyword2"]}
}"""

_WITH_SUGGESTIONS_SCHEMA = """2. Suggest 3 natural English replies for 'Me' to respond to 'Remote', along with Korean translations.

Output STRICTLY as JSON:
{
  "summary": {"remote": ["keyword1"], "me": ["keyword2"]},
  "suggestions": [
    {"en": "reply 1", "ko": "번역 1"},
    {"en": "reply 2", "ko": "번역 2"},
    {"en": "reply 3", "ko": "번역 3"}
  ]
}"""


def suggestion_prompt(prev_summary: dict, latest_text: str, speaker: str, need_suggestions: bool) -> str:
    """롤링 요약 갱신 (+ 답장 제안) 요청."""
    speaker_label = "Me" if speaker == "me" else "Remote"
    remote = json.dumps(prev_summary.get("remote", []), ensure_ascii=False)
    me = json.dumps(prev_summary.get("me", []), ensure_ascii=False)

    prompt = f"""[Current Status]
Remote: {remote}
Me: {me}

[Latest Message ({speaker_label})]
"{latest_text}"

Task:
1. Update the [Current Status] based on the [Latest Message] using VERY short Korean keywords. Do not hallucinate.
"""
    return prompt + (_WITH_SUGGESTIONS_SCHEMA if need_suggestions else _SUMMARY_ONLY_SCHEMA)
