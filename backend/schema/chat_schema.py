from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    requirements: str
    # 있으면 해당 대화에 이어붙이는 후속 턴 (이전 설계를 문맥으로 반영)
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    status: str
    chat_id: int
    # 대화 묶음 ID (첫 턴이면 chat_id와 동일)
    conversation_id: int
    image_url: str
    message: str
    terraform_code: Optional[dict] = None
    validation_summary: Optional[str] = None
    cost_estimate: Optional[dict] = None


class ChatHistoryItemResponse(BaseModel):
    id: int
    conversation_id: int
    requirements: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    id: int
    conversation_id: int
    requirements: str
    response_message: str
    image_url: Optional[str]
    terraform_code: Optional[dict] = None
    validation_summary: Optional[str] = None
    cost_estimate: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True
