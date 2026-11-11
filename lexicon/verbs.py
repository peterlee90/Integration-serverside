"""동사 활용형 처리: 불규칙 동사 표, 활용형 목록, 구동사 정규식."""
import logging
import os
import re

logger = logging.getLogger(__name__)

# 원형 → {원형, 과거형, 과거분사 ...}  (verbs.txt: "go | went | gone")
IRREGULAR_VERBS = {}

# 구동사 단어 사이에 끼어들 수 있는 단어 수 (예: "pick it up")
_GAP = r"(?:\s+\w+){0,4}"


def load_irregular_verbs(path: str):
    if not os.path.exists(path):
        logger.warning(f"[동사 로드 실패] {path} 없음.")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                IRREGULAR_VERBS[parts[0].lower()] = set(parts)
    logger.info("[동사 로드 성공] verbs.txt 적용 완료.")


def get_verb_forms(verb: str) -> set:
    """역색인용: 동사의 가능한 활용형 전부."""
    if verb in IRREGULAR_VERBS:
        return set(IRREGULAR_VERBS[verb])
    forms = {verb}
    if verb.endswith("e"):
        forms.update([verb + "s", verb[:-1] + "ed", verb[:-1] + "ing"])
    elif verb.endswith("y"):
        forms.update([verb[:-1] + "ies", verb[:-1] + "ied", verb + "ing"])
    else:
        forms.update([verb + "s", verb + "es", verb + "ed", verb + "ing",
                      verb + verb[-1] + "ed", verb + verb[-1] + "ing"])
    return forms


def get_verb_pattern(verb: str) -> str:
    """정규식용: 동사 활용형에 매칭되는 패턴."""
    if verb in IRREGULAR_VERBS:
        return rf"(?:{'|'.join(IRREGULAR_VERBS[verb])})"
    if verb.endswith("e"):
        return rf"{re.escape(verb[:-1])}(?:e|es|ed|ing)"
    if verb.endswith("y"):
        return rf"(?:{re.escape(verb[:-1])}(?:y|ies|ied)|{re.escape(verb)}ing)"
    return rf"{re.escape(verb)}(?:s|es|ed|ing|{verb[-1]}ed|{verb[-1]}ing)?"


def build_phrasal_regex(term: str, pos: str) -> str:
    """사전 표제어가 문장에 들어 있는지 찾는 정규식.

    동사가 아니면 단어 그대로, 동사면 활용형 허용.
    여러 단어 동사는 첫 단어나 마지막 단어 중 하나가 활용되고, 사이에 단어가 끼어도 됨.
    """
    if pos != "verb":
        return rf"\b{re.escape(term)}\b"

    words = term.split()
    if len(words) == 1:
        return rf"\b(?:{get_verb_pattern(words[0])})\b"

    first, last = words[0], words[-1]
    first_inflected = rf"\b(?:{get_verb_pattern(first)}){_GAP}\s+{re.escape(last)}\b"
    last_inflected = rf"\b{re.escape(first)}{_GAP}\s+(?:{get_verb_pattern(last)})\b"
    return rf"(?:{first_inflected}|{last_inflected})"
