from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import logging
import os

from backend.router import chat_router, user_router
from backend.database.config import engine
from backend.database import models
from ai_engine.rag.knowledge_base import load_knowledge_base


# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def run_migrations() -> None:
    """Alembic 마이그레이션을 head까지 적용한다.

    - 새 DB: upgrade가 전체 스키마를 생성
    - 기존(pre-alembic) DB: 이미 테이블이 있고 alembic_version이 없으면 stamp로 채택
      (create_all로 만들어졌던 기존 개발 DB를 깨지 않고 alembic 관리로 편입)
    """
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import inspect

    cfg = Config(os.path.join(ROOT_DIR, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT_DIR, "migrations"))

    tables = set(inspect(engine).get_table_names())
    if "alembic_version" not in tables and "user" in tables:
        logger.info("기존 DB 감지 — 현재 스키마를 alembic head로 stamp")
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


# ─── 시작 시 ChromaDB 미리 로드 (첫 요청 지연 방지) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()

    try:
        logger.info("ChromaDB warming up...")
        load_knowledge_base()
        logger.info("AI engine ready")
    except Exception as e:
        logger.warning(f"AI engine warmup skipped: {e}")
    yield


app = FastAPI(
    lifespan=lifespan,
)

# CORS 설정: 프론트엔드 포트(3000) 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth state 저장용 세션 미들웨어 (authlib이 사용). SECRET_KEY로 서명.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["SECRET_KEY"],
    same_site="lax",
    https_only=False,  # 배포(HTTPS) 시 True 권장
)

# static 폴더 설정: 이미지 저장 및 서빙용
# backend/static 폴더를 절대 경로로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 라우터 등록
app.include_router(chat_router.router)
app.include_router(user_router.router)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Cloud Diagram API is running"}

# 실행 가이드 (루트 폴더에서 실행 시):
# uvicorn backend.main:app --reload
