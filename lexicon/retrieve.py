"""문장에 들어 있는 사전 표제어 찾기 (역색인 + 정규식 → 벡터 검색으로 뜻 고르기)."""
import re
from collections import defaultdict

from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import COLLECTION_NAME, RETRIEVE_TOP_K, SEARCH_LIMIT
from lexicon.store import INVERTED_INDEX, TERM_REGEX_CACHE, embed_model, qdrant


def retrieve_hybrid(user_input: str, last_active_domain: str, top_k: int = RETRIEVE_TOP_K):
    """(매칭된 표제어 목록[점수순, 최대 top_k], 이번 턴의 도메인) 을 돌려준다."""
    present_terms = _find_present_terms(user_input)
    if not present_terms:
        return [], last_active_domain

    term_groups = _search_meanings(user_input, present_terms)

    resolved = []
    for term, items in term_groups.items():
        best = _pick_meaning(items, last_active_domain)
        resolved.append({
            "term": term, "domain": best["domain"], "pos": best["pos"],
            "meaning": best["meaning"], "score": best["score"],
        })

    domains = [m["domain"] for m in resolved]
    new_domain = max(set(domains), key=domains.count) if domains else last_active_domain
    return sorted(resolved, key=lambda x: x["score"], reverse=True)[:top_k], new_domain


def _find_present_terms(user_input: str) -> set:
    """문장에 실제로 들어 있는 표제어. 더 긴 표제어에 포함된 짧은 건 뺀다."""
    clean = re.sub(r"[^a-z0-9\s]", "", user_input.lower()).strip()

    candidates = set()
    for w in set(re.findall(r"\b\w+\b", clean)):
        candidates.update(INVERTED_INDEX.get(w, ()))

    found = {t for t in candidates if TERM_REGEX_CACHE[t].search(clean)}
    return {t for t in found if not any(t != other and t in other for other in found)}


def _search_meanings(user_input: str, terms: set) -> dict:
    """표제어별로 후보 뜻들을 문장과의 유사도 점수와 함께 모은다."""
    term_filter = Filter(must=[FieldCondition(key="term", match=MatchAny(any=list(terms)))])
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_model.encode(user_input).tolist(),
        query_filter=term_filter,
        limit=SEARCH_LIMIT,
    )

    groups = defaultdict(list)
    for hit in result.points:
        p = hit.payload
        groups[p["term"]].append({
            "domain": p["domain"], "meaning": p["meaning"],
            "pos": p["pos"], "score": hit.score,
        })
    return groups


def _pick_meaning(items: list, last_active_domain: str) -> dict:
    """직전 도메인과 같은 뜻이 있으면 그중 최고점, 없으면 전체 최고점."""
    same_domain = [x for x in items if x["domain"] == last_active_domain]
    return max(same_domain or items, key=lambda x: x["score"])
