"""quick.json 매칭용 문장 정규화.

로드할 때와 조회할 때 반드시 같은 normalize_text() 를 써야 매칭된다.
"""
import re

import contractions

# STT 가 아포스트로피를 빼먹은 축약형 복원 (위에서부터 순서대로 적용)
_STT_FIXES = [
    (re.compile(pattern), replacement)
    for pattern, replacement in [
        (r"\bim\b", "i am"),
        (r"\bive\b", "i have"),
        (r"\bdont\b", "do not"),
        (r"\bcant\b", "cannot"),
        (r"\bcouldnt\b", "could not"),
        (r"\bwouldnt\b", "would not"),
        (r"\bshouldnt\b", "should not"),
        (r"\bwont\b", "will not"),
        (r"\barent\b", "are not"),
        (r"\bisnt\b", "is not"),
        (r"\bwasnt\b", "was not"),
        (r"\bwerent\b", "were not"),
        (r"\bdidnt\b", "did not"),
        (r"\bdoesnt\b", "does not"),
        (r"\bhasnt\b", "has not"),
        (r"\bhavent\b", "have not"),
        (r"\bhadnt\b", "had not"),
        (r"\btheyre\b", "they are"),
        (r"\byoure\b", "you are"),
        (r"\bthats\b", "that is"),
        (r"\bwhats\b", "what is"),
        (r"\bhowre\b", "how are"),
    ]
]

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """소문자화 → 축약형 펼치기 → 영문/숫자/공백만 남기기 → 공백 정리."""
    if not s:
        return ""

    s = s.lower().strip()

    for pattern, replacement in _STT_FIXES:
        s = pattern.sub(replacement, s)

    # 나머지 축약형은 contractions 라이브러리로
    try:
        s = contractions.fix(s)
    except Exception:
        pass

    s = _NON_ALNUM.sub("", s)
    return _SPACES.sub(" ", s).strip()
