from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import os

from backend.router import chat_router
from backend.database.config import Base, engine
from backend.database import models
from ai_engine.rag.knowledge_base import load_knowledge_base

load_dotenv()

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── 시작 시 ChromaDB 미리 로드 (첫 요청 지연 방지) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

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

# static 폴더 설정: 이미지 저장 및 서빙용
# backend/static 폴더를 절대 경로로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 라우터 등록
app.include_router(chat_router.router)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Cloud Diagram API is running"}

# 실행 가이드 (루트 폴더에서 실행 시):
# uvicorn backend.main:app --reload
