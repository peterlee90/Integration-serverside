"""번역 서버 진입점: 로깅·인코딩 설정, 시작 시 사전 로드, 라우터 등록."""
import logging
import sys
from contextlib import asynccontextmanager

# 로깅은 내부 모듈(모델 로딩) import 전에 설정해야 형식이 적용된다
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from api.suggest import router as suggest_router  # noqa: E402
from api.translate import router as translate_router  # noqa: E402
from config import HOST, PORT  # noqa: E402
from lexicon import quick_match  # noqa: E402
from lexicon.store import init_vector_db  # noqa: E402

logger = logging.getLogger("app")

if sys.stdin:
    sys.stdin.reconfigure(encoding="utf-8", errors="ignore")
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("===== 서버 시작 =====")
    init_vector_db()
    quick_match.init_exact_matches()
    logger.info(f"EXACT_MATCH_DICT 크기: {len(quick_match.EXACT_MATCH_DICT)}")
    yield
    logger.info("===== 서버 종료 =====")


app = FastAPI(lifespan=lifespan)
app.include_router(translate_router)
app.include_router(suggest_router)


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
