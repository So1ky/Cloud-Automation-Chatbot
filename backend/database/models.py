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
    created_at = Column(DateTime, nullable=False, default=datetime.now)


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)