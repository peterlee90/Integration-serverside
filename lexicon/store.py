"""사전(TXT) → 벡터 DB + 역색인 구축.

사전 파일 형식 (한 줄): 도메인 | 표제어 | 품사 | 뜻 | 영어 예문 | 한국어 예문(생략 가능)
"""
import glob
import logging
import os
import re
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DATA_DIR, EMBED_BATCH_SIZE, EMBED_DEVICE, EMBED_MODEL_NAME, VERBS_FILE
from lexicon.verbs import build_phrasal_regex, get_verb_forms, load_irregular_verbs

logger = logging.getLogger(__name__)

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
qdrant = QdrantClient(":memory:")

INVERTED_INDEX = defaultdict(set)   # 단어(활용형) → 그 단어가 들어간 표제어들
TERM_REGEX_CACHE = {}               # 표제어 → 문장에서 찾는 정규식

_NO_KR_EXAMPLE = "번역 없음"


def init_vector_db():
    logger.info(f"[사전] 폴더: {os.path.abspath(DATA_DIR)}")

    _reset_collection()
    load_irregular_verbs(os.path.join(DATA_DIR, VERBS_FILE))

    if not os.path.exists(DATA_DIR):
        logger.error(f"[사전 로드 실패] {DATA_DIR} 폴더 자체가 없습니다!")
        return

    files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.txt")) if VERBS_FILE not in f]
    logger.info(f"[사전] 찾은 파일들: {files}")

    entries = _read_entries(files)
    texts, payloads = [], []
    for dom, term, pos, mean, en_ex, kr_ex in entries:
        _index_term(term, pos)
        texts.append(f"[{term}] {en_ex}")
        payloads.append({"domain": dom, "term": term, "pos": pos, "meaning": mean,
                         "en_example": en_ex, "kr_example": kr_ex})

    if not texts:
        return
    vectors = embed_model.encode(texts, batch_size=EMBED_BATCH_SIZE).tolist()
    points = [PointStruct(id=i, vector=v, payload=p)
              for i, (v, p) in enumerate(zip(vectors, payloads))]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"[사전 벡터 DB 완료] 총 {len(points)}개 인코딩 완료.")


def _reset_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embed_model.get_sentence_embedding_dimension(),
                                    distance=Distance.COSINE),
    )


def _read_entries(files: list) -> list:
    """사전 파일들을 읽어 (도메인, 표제어(소문자), 품사, 뜻, 영예문, 한예문) 목록으로."""
    entries = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6:
                    entries.append((parts[0], parts[1].lower(), *parts[2:6]))
                elif len(parts) == 5:
                    entries.append((parts[0], parts[1].lower(), *parts[2:5], _NO_KR_EXAMPLE))
    return entries


def _index_term(term: str, pos: str):
    """표제어 정규식을 만들고 역색인에 등록 (같은 표제어는 처음 한 번만)."""
    if term in TERM_REGEX_CACHE:
        return
    TERM_REGEX_CACHE[term] = re.compile(build_phrasal_regex(term, pos), re.IGNORECASE)
    for w in term.split():
        if pos == "verb":
            for form in get_verb_forms(w):
                INVERTED_INDEX[form.lower()].add(term)
        else:
            INVERTED_INDEX[w.lower()].add(term)
