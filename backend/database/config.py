import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# DB 연결 문자열은 DATABASE_URL 환경변수로 지정한다.
# 미설정 시 로컬 개발용 SQLite로 폴백한다.
#   PostgreSQL 예: postgresql://user:password@localhost:5432/cloudchatbot
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./backend/database/test.db"
)

# check_same_thread 옵션은 SQLite 전용이다.
_connect_args = (
    {"check_same_thread": False}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,  # 끊긴 커넥션 자동 감지 (Postgres 등 네트워크 DB에 유용)
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
