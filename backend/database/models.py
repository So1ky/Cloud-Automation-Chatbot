from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON

from backend.database.config import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    # 대화 묶음 ID (첫 턴의 id와 동일, 후속 턴은 첫 턴의 값을 상속)
    conversation_id = Column(Integer, nullable=True, index=True)
    requirements = Column(Text, nullable=False)
    response_message = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)
    terraform_code = Column(JSON, nullable=True)
    validation_summary = Column(Text, nullable=True)
    cost_estimate = Column(JSON, nullable=True)
    # 이 턴에서 확정된 아키텍처 YAML (다음 턴의 previous_yaml로 사용)
    yaml_output = Column(Text, nullable=True)
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