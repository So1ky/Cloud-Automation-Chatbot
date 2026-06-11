from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime

from backend.database.config import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    requirements = Column(Text, nullable=False)
    response_message = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


