"""요약 프롬프트 + 답변 제안 프롬프트 (LLM 호출을 둘로 나눠 동시에 돌린다).

요약:  입력 = 지금까지의 요약 + 최근 대화 + 방금 한 말 → {"summary": {"remote": [...], "me": [...]}}
제안:  입력 = 지금까지의 요약 + 최근 대화 + 방금 한 말 → {"suggestions": [{"en", "ko"} x3]}
"""
import json

from config import SUGGEST_MAX_WORDS, SUMMARY_MAX_KEYWORDS

# SGLang 구조화 출력용 JSON 스키마 (config.SUGGEST_JSON_SCHEMA 가 True 일 때만 사용)
_KEYWORDS = {"type": "array", "maxItems": SUMMARY_MAX_KEYWORDS, "items": {"type": "string"}}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "object",
            "properties": {"remote": _KEYWORDS, "me": _KEYWORDS},
            "required": ["remote", "me"],
        },
    },
    "required": ["summary"],
}
SUGGESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"en": {"type": "string"}, "ko": {"type": "string"}},
                "required": ["en", "ko"],
            },
        },
    },
    "required": ["suggestions"],
}


def format_turns(turns: list, user_id: str | None) -> str:
    """대화 턴 → 'Me: ...' / 'Remote: ...' 줄들. 화자를 모르면 'Speaker'."""
    lines = []
    for t in turns:
        if user_id and t.speaker == user_id:
            who = "Me"
        elif user_id and t.speaker != "unknown":
            who = "Remote"
        else:
            who = "Speaker"
        lines.append(f"{who}: {t.text}")
    return "\n".join(lines) if lines else "(none)"


def _dump(words: list) -> str:
    return json.dumps(words, ensure_ascii=False)


def summary_prompt(summary: dict, turns_text: str, latest_text: str, speaker: str) -> str:
    """요약 갱신 프롬프트. 키워드를 한국어로 받기 위해 지시문도 한국어로 쓴다."""
    who = "Me" if speaker == "me" else "Remote"
    return f"""영어 대화를 한국어 키워드로 정리한다. "Me" 는 나, "Remote" 는 상대다.

[지금까지의 요약]
Remote: {_dump(summary.get("remote", []))}
Me: {_dump(summary.get("me", []))}

[최근 대화]
{turns_text}

[방금 한 말 ({who})]
"{latest_text}"

할 일: 방금 한 말까지 반영해서 요약을 다시 쓴다.
- 키워드는 반드시 한국어로 쓴다. 영어 문장을 그대로 옮기지 말고 뜻을 한국어로 줄인다. (사람 이름, 게임 용어 같은 고유명사만 그대로)
- 키워드 하나는 1~4 단어. 사람마다 최대 {SUMMARY_MAX_KEYWORDS}개. 넘치면 덜 중요한 옛 키워드를 뺀다.
- 계획, 감정, 질문, 이름처럼 중요한 내용만 남긴다. 인사나 맞장구("hi", "what's up", "yeah")는 넣지 않는다.
- 대화에 없는 내용은 만들지 않는다. 넣을 것이 없으면 빈 목록 [] 으로 둔다.

예시 (형식 참고용, 내용은 베끼지 말 것)
Remote 가 "Hi, what's up? I'm so tired, I worked late." 라고 하고 Me 가 "Do you want to play later?" 라고 했으면
{{"summary": {{"remote": ["피곤함", "늦게까지 일함"], "me": ["나중에 게임 제안"]}}}}

아래 형식의 JSON 만 출력한다:
{{"summary": {{"remote": ["키워드"], "me": ["키워드"]}}}}"""


def suggestion_prompt(summary: dict, turns_text: str, latest_text: str, speaker: str) -> str:
    """답변 제안 3개 프롬프트 (요약은 읽기만 하고 고치지 않는다)."""
    who = "Me" if speaker == "me" else "Remote"
    return f"""You help "Me" in a live English conversation with "Remote".

[Summary so far (Korean keywords)]
Remote: {_dump(summary.get("remote", []))}
Me: {_dump(summary.get("me", []))}

[Recent conversation]
{turns_text}

[Latest message ({who})]
"{latest_text}"

Task: Suggest 3 replies that "Me" could say next to "Remote", based on the whole conversation and especially the latest message.
- Make the 3 replies go in different directions: (a) agree or answer positively, (b) ask a follow-up question, (c) a different opinion, a decline, or a light change of topic.
- If the latest message is a question, all 3 replies must answer it.
- Each reply: ONE short spoken sentence, max {SUGGEST_MAX_WORDS} words, casual and natural, matching the mood of the conversation.
- "ko": natural Korean meaning in casual speech (반말).

Output ONLY this JSON:
{{"suggestions": [{{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}]}}"""

Output ONLY this JSON:
{{"summary": {{"remote": ["키워드"], "me": ["키워드"]}}, "suggestions": [{{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}]}}"""
