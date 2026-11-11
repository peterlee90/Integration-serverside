"""LLM 에 보내는 프롬프트 모음.

문구를 한 글자라도 바꾸면 모델 출력이 달라질 수 있으니 주의.
"""
import json
import textwrap

# 주의: r"""\ 로 시작해서 실제 값은 '\' + 줄바꿈으로 시작하고, 들여쓰기도 안 지워진다.
#       지금 이 상태로 잘 동작하고 있어서 원본 그대로 둠.
SYSTEM_PROMPT = textwrap.dedent(r"""\
    # Role
    EN-KR Translator.
    # Rules
    [CRITICAL] Apply [Glossary] definitions.
""")


def translation_prompt(sentence: str, matches: list) -> str:
    """번역 요청 본문. 사전 매칭이 있으면 [사전] 블록을 앞에 붙인다."""
    base = f"원문: {sentence.strip()}\n번역:"
    if not matches:
        return base
    lines = [f"- {m['term']}: {m['meaning']}" for m in matches]
    return "[사전]\n" + "\n".join(lines) + f"\n{base}"


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
