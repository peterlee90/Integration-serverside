"""서버 전역 설정값. 숫자·경로·모델 이름은 전부 여기서 바꾼다."""

# ---- 서버 ----
HOST = "0.0.0.0"
PORT = 8081

# ---- 데이터 파일 (실행 위치 기준 상대 경로) ----
DATA_DIR = "data"
QUICK_MATCH_FILE = "quick.json"
VERBS_FILE = "verbs.txt"

# ---- LLM (SGLang OpenAI 호환 서버, 같은 Pod 안) ----
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
MODEL_NAME = "kakaocorp/kanana-1.5-8b-instruct-2505"   # 베이스: 토크나이저 + 요약·제안·중국어 보정

# 번역에 쓸 LoRA 어댑터 이름 (SGLang --lora-paths 의 "이름=경로" 에서 이름 부분)
# "" 로 두면 어댑터 없이 베이스로 번역
TRANSLATE_ADAPTER = "daily"
TRANSLATE_MODEL_NAME = f"{MODEL_NAME}:{TRANSLATE_ADAPTER}" if TRANSLATE_ADAPTER else MODEL_NAME

# ---- 임베딩 / 벡터 DB ----
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cuda"
COLLECTION_NAME = "hybrid_dictionary"
EMBED_BATCH_SIZE = 32
SEARCH_LIMIT = 100
RETRIEVE_TOP_K = 4

# ---- 방별 대화 상태 (rooms/state.py) ----
ROOM_MAX_TURNS = 30              # 방마다 보관하는 최근 대화 턴 수 (양쪽 화자 합쳐서)
ROOM_IDLE_TTL_SEC = 3 * 60 * 60  # 이 시간 동안 요청 없는 방은 메모리에서 지움 (초)
DEFAULT_DOMAIN = "league"        # 방이 처음 생길 때의 사전 도메인

# ---- 번역 ----
HISTORY_TURNS = 3            # 번역 문맥으로 넣는 이전 턴 수 (0 이면 문맥 안 넣음 = 학습 때와 같은 조건)
TRANSLATE_TEMPERATURE = 0.1
TRANSLATE_MAX_TOKENS = 60      # 번역 1줄이면 충분
TRANSLATE_STOP = ["\n"]        # 첫 줄만 받고 끊음 (학습·평가 때와 같음)

# ---- 중국어 섞임 보정 ----
FIX_TEMPERATURE = 0.1
FIX_MAX_TOKENS = 50

# ---- 요약 / 답장 제안 ----
SUGGEST_TEMPERATURE = 0.2    # 환각 방지를 위해 낮춤
SUGGEST_MAX_TOKENS = 300     # 요약 + 제안 3개
SUMMARY_MAX_TOKENS = 120     # 요약만
SUGGEST_TURNS = 6            # 제안 만들 때 넣는 최근 대화 턴 수 (양쪽 화자 합쳐서)
SUMMARY_MAX_KEYWORDS = 8     # 요약 키워드 사람별 최대 개수
SUGGEST_MAX_WORDS = 12       # 제안 한 문장 최대 단어 수
SUGGEST_JSON_SCHEMA = True   # SGLang 구조화 출력으로 JSON 강제 (에러 나면 False 로)
