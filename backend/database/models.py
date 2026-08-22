from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON

from backend.database.config import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    requirements = Column(Text, nullable=False)
    response_message = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)
    terraform_code = Column(JSON, nullable=True)
    validation_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    # OAuth(소셜) 가입 유저는 비밀번호가 없으므로 nullable.
    password = Column(String, nullable=True)
    # 가입 경로: "local"(이메일/비밀번호) | "github" | ...
    provider = Column(String, nullable=False, default="local")
    created_at = Column(DateTime, nullable=False, default=datetime.now)