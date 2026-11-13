"""'내가 하고 싶은 말' 프롬프트: 사용자가 한국어로 적은 말을 지금 대화에 맞는 영어로 바꾼다."""
import json

from config import SAY_MAX_SENTENCES, SUGGEST_MAX_WORDS


def say_prompt(summary: dict, turns_text: str, ko_text: str,
               base_en: str = "", previous: list | None = None) -> str:
    """base_en: 이미 말하기로 한 영어 문장 (있으면 그 뒤에 덧붙일 말만 쓴다)
    previous: 앞서 만들어 준 영어 문장들 (다시 요청했을 때 다른 표현을 쓰게 한다)"""
    remote = json.dumps(summary.get("remote", []), ensure_ascii=False)
    me = json.dumps(summary.get("me", []), ensure_ascii=False)

    if base_en:
        want = f""""Me" is going to say this sentence first:
"{base_en}"

Right after it, "Me" wants to add this (written in Korean):
"{ko_text}"

Task: Write only the part that "Me" adds, in English. Do NOT repeat the first sentence."""
    else:
        want = f""""Me" wants to say this next (written in Korean):
"{ko_text}"

Task: Write what "Me" should say in English."""

    again = ""
    if previous:
        tried = "\n".join(f'- "{p}"' for p in previous[-3:])
        again = f"\n- \"Me\" did not like these earlier versions. Say it in a clearly different way:\n{tried}"

    return f"""You help "Me" (a Korean speaker) in a live English conversation with "Remote".

[Summary so far (Korean keywords)]
Remote: {remote}
Me: {me}

[Recent conversation]
{turns_text}

{want}
- Keep the meaning and feeling of the Korean. Do not translate word by word: say it the way a native speaker would in this conversation.
- Do not add facts or opinions that are not in the Korean.
- {SAY_MAX_SENTENCES} sentences at most, each short and easy to say aloud (max {SUGGEST_MAX_WORDS} words), casual and natural.
- "en": the English to say. "ko": what that English means, in natural casual Korean (반말), so "Me" can check it.{again}

Output ONLY this JSON, nothing else:
{{"en": "...", "ko": "..."}}"""
