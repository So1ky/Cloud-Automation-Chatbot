from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    requirements: str

class ChatResponse(BaseModel):
    status: str
    chat_id: int
    image_url: str
    message: str
    terraform_code: Optional[dict] = None
    validation_summary: Optional[str] = None
    cost_estimate: Optional[dict] = None


class ChatHistoryItemResponse(BaseModel):
    id: int
    requirements: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    id: int
    requirements: str
    response_message: str
    image_url: Optional[str]
    terraform_code: Optional[dict] = None
    validation_summary: Optional[str] = None
    cost_estimate: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True
