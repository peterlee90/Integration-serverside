"""quick.json: 자주 쓰는 문장 → 정해 둔 번역 (LLM 을 거치지 않음).

키는 normalize_text() 로 정규화해서 저장한다.
"""
import json
import logging
import os

from config import DATA_DIR, QUICK_MATCH_FILE
from lexicon.normalize import normalize_text

logger = logging.getLogger(__name__)

EXACT_MATCH_DICT = {}

# 시작할 때 매칭이 잘 되는지 확인해 보는 문장들
_SELF_CHECK_SAMPLES = ["I get it", "I'm sorry", "Im sorry", "thank you", "hello"]


def init_exact_matches():
    json_path = os.path.join(DATA_DIR, QUICK_MATCH_FILE)
    logger.info(f"[quick.json] 경로: {os.path.abspath(json_path)}")

    if not os.path.exists(json_path):
        logger.error(f"[quick.json] 파일 없음: {json_path}")
        EXACT_MATCH_DICT.clear()
        return

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    normalized, skipped = _normalize_keys(loaded)
    EXACT_MATCH_DICT.clear()
    EXACT_MATCH_DICT.update(normalized)

    logger.info(
        f"[quick.json] 원본 {len(loaded)}개 → 정규화 {len(EXACT_MATCH_DICT)}개 "
        f"(skip {len(skipped)}개)"
    )
    logger.info(f"  원본 키 샘플: {list(loaded.keys())[:5]}")
    logger.info(f"  정규화 키 샘플: {list(EXACT_MATCH_DICT.keys())[:5]}")
    _self_check()


def _normalize_keys(loaded: dict):
    """키를 정규화한 새 dict 와, 정규화 후 빈 문자열이 돼서 버린 키 목록."""
    result, skipped = {}, []
    for key, value in loaded.items():
        nk = normalize_text(key)
        if not nk:
            skipped.append(key)
            continue
        if nk in result and result[nk] != value:
            logger.warning(
                f"[quick.json] 키 충돌: {nk!r} (기존 {result[nk]!r} vs 신규 {value!r}) → 신규 유지"
            )
        result[nk] = value
    return result, skipped


def _self_check():
    for sample in _SELF_CHECK_SAMPLES:
        n = normalize_text(sample)
        logger.info(f"  [자가진단] {sample!r} → {n!r} → 매칭: {n in EXACT_MATCH_DICT}")
