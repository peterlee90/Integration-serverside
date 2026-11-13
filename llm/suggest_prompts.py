"""요약 프롬프트 + 답변 제안 프롬프트 (LLM 호출을 둘로 나눠 동시에 돌린다).

요약:  입력 = 지금까지의 요약 + 최근 대화 + 방금 한 말 → {"summary": {"remote": [...], "me": [...]}}
제안:  입력 = 같은 것 + 답변 방향 하나 → {"en", "ko"} 한 개. 방향 3개를 따로 호출해서 동시에 돌린다.
이어 말하기: 입력 = 요약 + 최근 대화 + 고른 카드 문장 → 뒤에 붙일 문장 {"en", "ko"}
"""
import json

from config import EXPAND_MAX_SENTENCES, SUGGEST_MAX_WORDS, SUMMARY_MAX_KEYWORDS

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
ONE_SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {"en": {"type": "string"}, "ko": {"type": "string"}},
    "required": ["en", "ko"],
}

# 답변 3개의 방향. 방향마다 LLM 을 따로 불러서 서로 다른 답이 나오게 한다.
DIRECTIONS = [
    "agree, or give a positive answer",
    "react in a few words, then ask a follow-up question",
    "give a different opinion, a polite decline, or lightly change the topic",
]


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
- 키워드는 반드시 한국어로 쓴다. 영어 문장을 그대로 옮기지 말고 뜻을 한국어로 줄인다. 예: "I'm so hungry" → "배고픔". (사람 이름, 게임 용어 같은 고유명사만 그대로)
- 키워드 하나는 1~4 단어. 사람마다 최대 {SUMMARY_MAX_KEYWORDS}개. 넘치면 덜 중요한 옛 키워드를 뺀다.
- 계획, 감정, 질문, 이름처럼 중요한 내용만 남긴다. 인사나 맞장구("hi", "what's up", "yeah")는 넣지 않는다.
- 위 대화와 요약에 실제로 나온 내용만 쓴다. 없는 내용은 만들지 않는다. 넣을 것이 없으면 빈 목록 [] 으로 둔다.

아래 형식의 JSON 만 출력한다:
{{"summary": {{"remote": ["키워드"], "me": ["키워드"]}}}}"""


def suggestion_prompt(summary: dict, turns_text: str, latest_text: str, speaker: str,
                      direction: str) -> str:
    """답변 제안 1개 프롬프트 (요약은 읽기만 하고 고치지 않는다)."""
    who = "Me" if speaker == "me" else "Remote"
    return f"""You help "Me" in a live English conversation with "Remote".

[Summary so far (Korean keywords)]
Remote: {_dump(summary.get("remote", []))}
Me: {_dump(summary.get("me", []))}

[Recent conversation]
{turns_text}

[Latest message ({who})]
"{latest_text}"

Task: Write ONE reply that "Me" could say next to "Remote", based on the whole conversation and especially the latest message.
- Direction of this reply: {direction}.
- If the latest message is a question, the reply must answer it (still in that direction).
- ONE short spoken sentence, max {SUGGEST_MAX_WORDS} words, casual and natural, matching the mood of the conversation.
- "ko": natural Korean meaning in casual speech (반말).

Output ONLY this JSON:
{{"en": "...", "ko": "..."}}"""


def expand_prompt(summary: dict, turns_text: str, first_en: str) -> str:
    """고른 카드 문장 뒤에 이어 붙일 문장 프롬프트 (카드 문장 자체는 다시 쓰지 않는다)."""
    return f"""You help "Me" in a live English conversation with "Remote".

[Summary so far (Korean keywords)]
Remote: {_dump(summary.get("remote", []))}
Me: {_dump(summary.get("me", []))}

[Recent conversation]
{turns_text}

"Me" is going to say this sentence next:
"{first_en}"

Task: Write what "Me" says right after that sentence, so the answer is not too short.
- Write {EXPAND_MAX_SENTENCES} more sentences at most. Add a reason, a detail, or a question back to "Remote".
- Do NOT repeat or rewrite the first sentence. Do not invent facts that contradict the conversation.
- Each sentence: short and easy to say aloud, max {SUGGEST_MAX_WORDS} words, casual and natural.
- "en": only the new sentences. "ko": their natural Korean meaning in casual speech (반말).

Output ONLY this JSON:
{{"en": "...", "ko": "..."}}"""
