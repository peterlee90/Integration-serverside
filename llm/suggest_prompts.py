"""요약 갱신 + 답변 제안 프롬프트.

입력: 지금까지의 요약(보는 사람 기준 remote/me) + 방의 최근 대화 턴 + 마지막 메시지
출력: JSON {"summary": {...}, "suggestions": [{"en", "ko"} x3]}
"""
import json

from config import SUGGEST_MAX_WORDS, SUMMARY_MAX_KEYWORDS

# SGLang 구조화 출력용 JSON 스키마 (config.SUGGEST_JSON_SCHEMA 가 True 일 때만 사용)
_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "remote": {"type": "array", "items": {"type": "string"}},
        "me": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["remote", "me"],
}
SUMMARY_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"summary": _SUMMARY_SCHEMA},
    "required": ["summary"],
}
WITH_SUGGESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _SUMMARY_SCHEMA,
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
    "required": ["summary", "suggestions"],
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


def suggestion_prompt(summary: dict, turns_text: str, latest_text: str, speaker: str,
                      need_suggestions: bool) -> str:
    speaker_label = "Me" if speaker == "me" else "Remote"
    remote = json.dumps(summary.get("remote", []), ensure_ascii=False)
    me = json.dumps(summary.get("me", []), ensure_ascii=False)

    prompt = f"""You help "Me" in a live English conversation with "Remote".

[Summary so far]
Remote: {remote}
Me: {me}

[Recent conversation]
{turns_text}

[Latest message ({speaker_label})]
"{latest_text}"

Task:
1. Update the summary using VERY short Korean keywords (max {SUMMARY_MAX_KEYWORDS} per person). Keep important facts (plans, feelings, questions, names). Drop minor old ones. Do not invent anything.
"""
    if not need_suggestions:
        return prompt + """
Output ONLY this JSON:
{"summary": {"remote": ["키워드"], "me": ["키워드"]}}"""

    return prompt + f"""2. Suggest 3 replies that "Me" could say next to "Remote", based on the whole conversation and especially the latest message.
- Make the 3 replies go in different directions: (a) agree or answer positively, (b) ask a follow-up question, (c) a different opinion, a decline, or a light change of topic.
- If the latest message is a question, all 3 replies must answer it.
- Each reply: ONE short spoken sentence, max {SUGGEST_MAX_WORDS} words, casual and natural, matching the mood of the conversation.
- "ko": natural Korean meaning in casual speech (반말).

Output ONLY this JSON:
{{"summary": {{"remote": ["키워드"], "me": ["키워드"]}}, "suggestions": [{{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}, {{"en": "...", "ko": "..."}}]}}"""
