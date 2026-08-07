import os
import re
import glob
import json
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "hybrid_dictionary"

EXACT_MATCH_DICT = {}
IRREGULAR_VERBS = {}
INVERTED_INDEX = defaultdict(set)
TERM_REGEX_CACHE = {}

# 📌 1. 절대 경로 대신 상대 경로(현재 폴더 안의 data 폴더) 사용
DATA_DIR = "data" 

def init_exact_matches():
    global EXACT_MATCH_DICT
    json_path = os.path.join(DATA_DIR, "quick.json")
    
    # 📌 2. 경로 추적 로그 추가
    print(f"🔍 [디버그] 하드매칭(quick.json) 찾는 경로: {os.path.abspath(json_path)}")
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            EXACT_MATCH_DICT = json.load(f)
        print(f"✅ [하드매칭 로드 성공]: {len(EXACT_MATCH_DICT)}개 단어 장전 완료.")
    else:
        print(f"❌ [하드매칭 로드 실패]: 해당 경로에 quick.json 파일이 없습니다!")

def get_verb_forms(verb):
    if verb in IRREGULAR_VERBS: return set(IRREGULAR_VERBS[verb])
    forms = {verb}
    if verb.endswith('e'):
        forms.update([verb + 's', verb[:-1] + 'ed', verb[:-1] + 'ing'])
    elif verb.endswith('y'):
        forms.update([verb[:-1] + 'ies', verb[:-1] + 'ied', verb + 'ing'])
    else:
        forms.update([verb + 's', verb + 'es', verb + 'ed', verb + 'ing', verb + verb[-1] + 'ed', verb + verb[-1] + 'ing'])
    return forms

def get_verb_pattern(verb):
    if verb in IRREGULAR_VERBS: return rf"(?:{'|'.join(IRREGULAR_VERBS[verb])})"
    if verb.endswith('e'): return rf"{re.escape(verb[:-1])}(?:e|es|ed|ing)"
    elif verb.endswith('y'): return rf"(?:{re.escape(verb[:-1])}(?:y|ies|ied)|{re.escape(verb)}ing)"
    else: return rf"{re.escape(verb)}(?:s|es|ed|ing|{verb[-1]}ed|{verb[-1]}ing)?"

def build_phrasal_regex(term, pos):
    words = term.split()
    if pos != "verb": return rf"\b{re.escape(term)}\b"
    gap = r"(?:\s+\w+){0,4}"
    if len(words) == 1: return rf"\b(?:{get_verb_pattern(words[0])})\b"
    first, last = words[0], words[-1]
    return rf"(?:\b(?:{get_verb_pattern(first)}){gap}\s+{re.escape(last)}\b|\b{re.escape(first)}{gap}\s+(?:{get_verb_pattern(last)})\b)"

def init_vector_db():
    print(f"🔍 [디버그] 사전(TXT) 찾는 폴더: {os.path.abspath(DATA_DIR)}")
    
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )
    
    verbs_path = os.path.join(DATA_DIR, "verbs.txt")
    if os.path.exists(verbs_path):
        with open(verbs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2: IRREGULAR_VERBS[parts[0].lower()] = set(parts)
        print(f"✅ [동사 로드 성공]: verbs.txt 적용 완료.")
    else:
        print(f"⚠️ [동사 로드 실패]: {verbs_path} 없음.")

    if not os.path.exists(DATA_DIR): 
        print(f"❌ [사전 로드 실패]: {DATA_DIR} 폴더 자체가 없습니다!")
        return
        
    files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.txt")) if "verbs.txt" not in f]
    print(f"🔍 [디버그] 찾은 사전 파일들: {files}")
    
    points, point_id, loaded_data = [], 0, []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6: loaded_data.append((parts[0], parts[1].lower(), parts[2], parts[3], parts[4], parts[5]))
                elif len(parts) == 5: loaded_data.append((parts[0], parts[1].lower(), parts[2], parts[3], parts[4], "번역 없음"))

    texts_to_embed, payloads = [], []
    for dom, term, pos, mean, en_ex, kr_ex in loaded_data:
        if term not in TERM_REGEX_CACHE:
            TERM_REGEX_CACHE[term] = re.compile(build_phrasal_regex(term, pos), re.IGNORECASE)
            for w in term.split():
                if pos == "verb":
                    for form in get_verb_forms(w): INVERTED_INDEX[form.lower()].add(term)
                else: INVERTED_INDEX[w.lower()].add(term)
        texts_to_embed.append(f"[{term}] {en_ex}")
        payloads.append({"domain": dom, "term": term, "pos": pos, "meaning": mean, "en_example": en_ex, "kr_example": kr_ex})
        
    if texts_to_embed:
        vectors = embed_model.encode(texts_to_embed, batch_size=32).tolist()
        for i, (vector, payload) in enumerate(zip(vectors, payloads)):
            points.append(PointStruct(id=point_id + i, vector=vector, payload=payload))
        point_id += len(vectors)
    
    if points: 
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ [사전 벡터 DB 완료]: 총 {len(points)}개 인코딩 완료.")

def retrieve_hybrid(user_input, last_active_domain, top_k=4):
    clean_input = re.sub(r'[^a-z0-9\s]', '', user_input.lower()).strip()
    candidate_terms = set()
    for w in set(re.findall(r'\b\w+\b', clean_input)):
        if w in INVERTED_INDEX: candidate_terms.update(INVERTED_INDEX[w])
            
    raw_present_terms = set()
    for term in candidate_terms:
        if TERM_REGEX_CACHE[term].search(clean_input): raw_present_terms.add(term)

    present_terms = set(raw_present_terms)
    for t1 in raw_present_terms:
        for t2 in raw_present_terms:
            if t1 != t2 and t1 in t2 and t1 in present_terms: present_terms.remove(t1)

    if not present_terms: return [], last_active_domain

    term_filter = Filter(must=[FieldCondition(key="term", match=MatchAny(any=list(present_terms)))])
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=embed_model.encode(user_input).tolist(), 
        query_filter=term_filter, limit=100
    )

    term_groups = defaultdict(list)
    for hit in search_result.points:
        p = hit.payload
        term_groups[p['term']].append({"domain": p['domain'], "meaning": p['meaning'], "pos": p['pos'], "score": hit.score})

    resolved_matches, current_turn_domains = [], []
    for term, items in term_groups.items():
        if len(items) == 1: best_match = items[0]
        else:
            domain_matched = [x for x in items if x['domain'] == last_active_domain]
            best_match = max(domain_matched, key=lambda x: x['score']) if domain_matched else max(items, key=lambda x: x['score'])
        resolved_matches.append({"term": term, "domain": best_match["domain"], "pos": best_match["pos"], "meaning": best_match["meaning"], "score": best_match["score"]})
        current_turn_domains.append(best_match["domain"])

    new_active_domain = max(set(current_turn_domains), key=current_turn_domains.count) if current_turn_domains else last_active_domain
    return sorted(resolved_matches, key=lambda x: x['score'], reverse=True)[:top_k], new_active_domain
