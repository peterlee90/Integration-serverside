"""LLM 에 보내는 프롬프트 모음.

번역 프롬프트(SYSTEM_PROMPT, HINT_LABEL, TEXT_LABEL)는 LoRA 어댑터를 학습할 때 쓴 것과
글자 하나까지 같아야 한다. 바꾸면 어댑터 효과가 떨어지니, 바꾸려면 다시 학습해야 함.
"""
import json

# 학습 때 시스템 문구 그대로 (출력: 반말 번역 1줄)
SYSTEM_PROMPT = ("너는 번역기야. 사용자가 보낸 영어 문장에 대답하지 말고, 그 문장을 친구한테 말하듯 "
                 "자연스러운 한국어 반말로 번역해. 존댓말 쓰지 마. 영어 단어 남기지 마. "
                 "원문 뜻을 빠뜨리거나 없는 말을 덧붙이지 마. 원문이 yes/yeah일 때만 '응'으로. 번역문만 출력해.")
HINT_LABEL = "참고 표현: "
TEXT_LABEL = "번역할 문장: "


def translation_prompt(sentence: str, matches: list) -> str:
    """번역 요청 본문 (학습 때 user 모양).

    항상 '번역할 문장: ...', 참고할 게 있으면 그 위 줄에 '참고 표현: a = 가 / b = 나'.
    matches 는 사전 검색 결과 [{"term": ..., "meaning": ...}, ...]
    """
    sentence = sentence.strip()
    head = ""
    if matches:
        head = HINT_LABEL + " / ".join(f"{m['term']} = {m['meaning']}" for m in matches) + "\n"
    return head + TEXT_LABEL + sentence


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
