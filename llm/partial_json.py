"""아직 다 안 온 JSON 글자에서 값을 미리 꺼낸다 (스트리밍 표시용).

LLM 이 {"en": "I usually rel   ← 여기까지만 왔어도 "I usually rel" 을 돌려준다.
"""
import json
import re

_STR_BODY = r'((?:[^"\\]|\\.)*)'                 # 따옴표 안쪽 (닫는 따옴표는 없어도 됨)
_CUT_ESCAPE = re.compile(r'\\(u[0-9a-fA-F]{0,3})?$')   # 끝에서 잘린 \ 또는 \u12


def _unescape(raw: str) -> str:
    raw = _CUT_ESCAPE.sub("", raw)
    try:
        return json.loads(f'"{raw}"', strict=False)
    except ValueError:
        return raw


def partial_string(text: str, key: str) -> str:
    """ "key": "값..." 의 값. 아직 안 나왔으면 빈 문자열. 닫는 따옴표가 없어도 지금까지 온 만큼."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"{_STR_BODY}', text, flags=re.S)
    return _unescape(match.group(1)) if match else ""


def partial_list(text: str, key: str) -> list:
    """ "key": ["a", "b", "c   의 항목들. 따옴표가 닫힌 항목만 돌려준다 (위 예: ["a", "b"])."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[([^\]]*)', text, flags=re.S)
    if not match:
        return []
    items = re.findall(rf'"{_STR_BODY}"', match.group(1), flags=re.S)
    return [w for w in (_unescape(i).strip() for i in items) if w]
