"""서버 전역 설정값. 숫자·경로·모델 이름은 전부 여기서 바꾼다."""

# ---- 서버 ----
HOST = "0.0.0.0"
PORT = 8081

# ---- 데이터 파일 (실행 위치 기준 상대 경로) ----
DATA_DIR = "data"
QUICK_MATCH_FILE = "quick.json"
VERBS_FILE = "verbs.txt"

# ---- LLM (SGLang OpenAI 호환 서버) ----
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
MODEL_NAME = "kakaocorp/kanana-1.5-8b-instruct-2505"   # 토크나이저 + 요청 둘 다 (어댑터 붙이면 나눔)

# ---- 임베딩 / 벡터 DB ----
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cuda"
COLLECTION_NAME = "hybrid_dictionary"
EMBED_BATCH_SIZE = 32
SEARCH_LIMIT = 100
RETRIEVE_TOP_K = 4

# ---- 번역 ----
HISTORY_TURNS = 3            # 방마다 기억하는 이전 번역 수
DEFAULT_DOMAIN = "league"    # 처음 시작할 때의 사전 도메인
TRANSLATE_TEMPERATURE = 0.1
TRANSLATE_MAX_TOKENS = 100

# ---- 중국어 섞임 보정 ----
FIX_TEMPERATURE = 0.1
FIX_MAX_TOKENS = 50

# ---- 요약 / 답장 제안 ----
SUGGEST_TEMPERATURE = 0.2    # 환각 방지를 위해 낮춤
SUGGEST_MAX_TOKENS = 250     # 제안 포함
SUMMARY_MAX_TOKENS = 100     # 요약만
