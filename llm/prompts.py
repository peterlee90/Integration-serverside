"""LLM 에 보내는 프롬프트 모음.

번역 프롬프트(SYSTEM_PROMPT, HINT_LABEL, TEXT_LABEL)는 LoRA 어댑터를 학습할 때 쓴 것과
글자 하나까지 같아야 한다. 바꾸면 어댑터 효과가 떨어지니, 바꾸려면 다시 학습해야 함.
"""
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
